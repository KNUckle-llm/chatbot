from urllib.parse import urlparse

from dotenv import load_dotenv
import os
import logging
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
import json
from datetime import datetime
from prompt_comparison.prompt_variants import PROMPT_VARIANTS, system_message
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, trim_messages
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import pymysql
import asyncio


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Global resources (pre-loaded for performance)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


def get_current_korean_date():
    now = datetime.now()
    weekdays = ['월', '화', '수', '목', '금', '토', '일']
    weekday = weekdays[now.weekday()]
    return f"{now.year}년 {now.month}월 {now.day}일 {weekday}요일"

def create_dynamic_system_message():
    current_date = get_current_korean_date()
    date_header = f"""## 📅 현재 시스템 정보
**현재 날짜**: {current_date}
**기준 시간**: 표준시 (UTC)

"""
    return date_header + system_message


# Original prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", create_dynamic_system_message()),
    ("placeholder", "{memory}"),
    # ("user", "🔍 검색된 문서:\n{context}"),
    ("human", "{input}"),
])

parser = StrOutputParser()
trimmer = trim_messages(max_tokens=5000, token_counter=llm, strategy="last", include_system=True, start_on="human")

# Session store
retriever = None
rag_chain = None
history_store = {}

# 더미 유저 ID
DUMMY_USER_ID = "knuckle"


def create_session_record(session_id: str, user_id: str = DUMMY_USER_ID):
    """session 테이블에 세션 레코드 생성"""
    try:
        mysql_url = os.getenv("DATABASE_URL")
        parsed_url = urlparse(mysql_url)

        connection = pymysql.connect(
            host=parsed_url.hostname,
            port=parsed_url.port or 3306,
            user=parsed_url.username,
            password=parsed_url.password,
            database=parsed_url.path[1:],
            autocommit=True
        )

        cursor = connection.cursor()

        # 세션이 이미 존재하는지 확인
        cursor.execute("SELECT session_id FROM session WHERE session_id = %s", (session_id,))
        if cursor.fetchone():
            cursor.close()
            connection.close()
            return

        # 새 세션 생성
        cursor.execute(
            "INSERT INTO session (session_id, user_id, started_at) VALUES (%s, %s, %s)",
            (session_id, user_id, datetime.now())
        )

        cursor.close()
        connection.close()
        logger.info(f"세션 생성: {session_id}")

    except Exception as e:
        logger.error(f"세션 생성 실패: {e}")


def init_session(session_id: str):
    """기존 함수에 세션 테이블 저장 로직 추가"""
    # 1. session 테이블에 레코드 생성
    create_session_record(session_id)

    # 2. 기존 로직
    if session_id not in history_store:
        mysql_url = os.getenv("DATABASE_URL")
        history_store[session_id] = SQLChatMessageHistory(
            connection=mysql_url,
            table_name="chat_history",
            session_id=session_id,
            session_id_field_name="session_id",
        )
        history_store[session_id].add_message(SystemMessage(content=system_message.strip()))


def get_session_history(session_id: str):
    return history_store[session_id]


# Global variables
retriever = None
rag_chain = None
prompt_chains = {}  # 각 프롬프트별 체인 저장


def test_db_connection() -> bool:
    """데이터베이스 연결 및 문서 존재 여부 테스트"""
    try:
        db_path = "./ChromaDB/knu_chroma_db_all"
        if not os.path.exists(db_path):
            logger.error(f"ChromaDB 경로가 존재하지 않습니다: {db_path}")
            return False

        db = Chroma(persist_directory=db_path, embedding_function=hf_embeddings)

        try:
            collection = db._collection
            if collection is None:
                logger.error("Chroma collection이 None입니다")
                return False

            count = collection.count()
            logger.info(f"데이터베이스의 총 문서 수: {count}")

            if count == 0:
                logger.warning("데이터베이스에 문서가 없습니다")
                return False

        except Exception as e:
            logger.error(f"컬렉션 접근 오류: {e}")
            return False

        try:
            results = db.get(limit=3, include=["documents", "metadatas"])
            logger.info(f"샘플 문서 {len(results.get('documents', []))}개 확인됨")
            return True
        except Exception as e:
            logger.error(f"샘플 문서 조회 오류: {e}")
            return False

    except Exception as e:
        logger.error(f"데이터베이스 연결 테스트 실패: {e}")
        return False


def rebuild_chain(selected_dept: str = None):
    global retriever, rag_chain

    # 사용자가 부서를 선택한 경우: 해당 부서 전용 DB만 불러와 retriever 구성(빠름)

    # 부서를 선택하지 않은 경우: 모든 부서의 문서가 저장된 DB를 로드하여 retriever 구성(느림)
    db_path = f"./ChromaDB/knu_chroma_db_all"
    db = Chroma(persist_directory=db_path, embedding_function=hf_embeddings)

    """
    벡터 거리 기반 의미 검색
    질문 -> MultiQuery(3개의 서브 질문)
    서브 질문 1 -> ChromaDB에서 mmr방식(다양하고도 유사한 문서를 찾는 방식)으로 검색하여 10개 후보 중 5개를 선택
    서브 질문 2 -> ChromaDB에서 mmr방식으로 검색하여 10개 후보 중 5개를 선택
    서브 질문 3 -> ChromaDB에서 mmr방식으로 검색하여 10개 후보 중 5개를 선택

    키워드 기반 재정렬 - Document 리스트 상태에서 작동함
    청크 N개를 대상으로 BM25(키워드(=질문에서 나온 단어)가 문서 안에 얼마나 잘 등장하는지 관련성을 수치화하는 방식)

    앙상블 -> top 3개 선택
    -> LLM context에 top3 청크 3개 삽입 -> 답변 생성
    """
    # 전체 문서 로딩 후 BM25 구성용 문서 리스트 확보
    try:
        results = db.get(include=["documents", "metadatas"])
        docs = [Document(page_content=doc, metadata=meta) for doc, meta in
                zip(results["documents"], results["metadatas"])]

        # Chroma + BM25 혼합 검색기 구성
        chroma_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
        )

        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 3
        base_retriever = EnsembleRetriever(retrievers=[chroma_retriever, bm25_retriever])
        retriever = MultiQueryRetriever.from_llm(base_retriever, llm=llm)

        # 최종 RAG 체인
        rag_chain_core = {
                             "memory": trimmer,  # trimmer 적용
                             "context": retriever,
                             "input": RunnablePassthrough(),
                             # "selected_dept": RunnablePassthrough()
                         } | prompt_template | llm | parser

        # 최종 RAG 체인
        rag_chain = RunnableWithMessageHistory(
            rag_chain_core,
            get_session_history,
            input_messages_key="input",
            history_messages_key="memory"
        )

    except Exception as e:
        logger.error(f"RAG 체인 구성 실패: {e}")
        
    return rag_chain


def answer_question(question: str, session_id: str, selected_dept: str = 'ALL(전체)') -> str:
    if not session_id:
        return "⚠️ 세션을 먼저 생성하거나 선택해주세요."

    init_session(session_id)

    # rag_chain의 입력 형식에 맞춰 dict 구성
    chain_input = {
        "memory": None,  # RunnableWithMessageHistory가 알아서 처리
        "context": None,  # retriever가 알아서 처리
        "input": question,
    }

    # RAG 체인 호출 → 답변 생성(자동으로 history에 기록됨)
    answer = rag_chain.invoke(
        chain_input,
        config={"configurable": {"session_id": session_id}}
    )
    return answer


# 🚀 새로운 스트리밍 응답 함수들

# 🚀 새로운 스트리밍 응답 함수들
async def streaming_answer_question(question: str, session_id: str, selected_dept: str = 'ALL(전체)') -> AsyncGenerator[
    str, None]:
    """스트리밍 방식으로 질문에 대한 답변 생성"""
    try:
        if not session_id:
            yield "⚠️ 세션을 먼저 생성하거나 선택해주세요."
            return

        if rag_chain is None:
            logger.info("RAG 체인이 없습니다. 구성 중...")
            if not rebuild_chain():
                yield "⚠️ RAG 시스템 초기화에 실패했습니다. 문서 데이터베이스를 확인해주세요."
                return

        init_session(session_id)

        if not question.strip():
            yield "⚠️ 유효한 질문을 입력해주세요."
            return

        logger.info(f"스트리밍 질문 처리 중: {question[:50]}...")

        # rag_chain의 입력 형식에 맞춰 dict 구성
        chain_input = {
            "memory": None,  # RunnableWithMessageHistory가 알아서 처리
            "context": None,  # retriever가 알아서 처리
            "input": question,
        }

        # 방법 1: sync stream을 async로 변환
        try:
            # sync stream 사용
            for chunk in rag_chain.stream(
                    chain_input,
                    config={"configurable": {"session_id": session_id}}
            ):
                if chunk:
                    yield chunk
                    # 약간의 딜레이로 더 자연스러운 스트리밍 효과
                    await asyncio.sleep(0.01)

        except Exception as stream_error:
            logger.warning(f"stream 메서드 실패, 대체 방법 시도: {stream_error}")

            # 방법 2: LLM만 스트리밍하고 나머지는 일반 처리
            try:
                # 컨텍스트 먼저 검색
                retrieved_docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                # 히스토리 가져오기
                history = get_session_history(session_id)
                memory_messages = history.messages

                # 프롬프트 구성
                formatted_prompt = prompt_template.format_messages(
                    context=context,
                    input=question,
                    memory=memory_messages
                )

                # LLM만 스트리밍
                async for chunk in llm.astream(formatted_prompt):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
                        await asyncio.sleep(0.01)
                    elif isinstance(chunk, str):
                        yield chunk
                        await asyncio.sleep(0.01)

                # 히스토리에 수동으로 추가
                from langchain_core.messages import HumanMessage, AIMessage
                history.add_message(HumanMessage(content=question))
                # 전체 응답은 나중에 end 시점에서 추가

            except Exception as llm_error:
                logger.error(f"LLM 스트리밍도 실패: {llm_error}")
                # 방법 3: 일반 답변을 청크로 나누어 스트리밍 효과
                try:
                    full_answer = answer_question(question, session_id)
                    words = full_answer.split()

                    current_chunk = ""
                    for i, word in enumerate(words):
                        current_chunk += word + " "
                        # 몇 단어마다 청크로 전송
                        if (i + 1) % 3 == 0 or i == len(words) - 1:
                            yield current_chunk
                            current_chunk = ""
                            await asyncio.sleep(0.05)

                except Exception as fallback_error:
                    logger.error(f"모든 스트리밍 방법 실패: {fallback_error}")
                    yield f"⚠️ 스트리밍 처리 중 오류가 발생했습니다: {str(fallback_error)}"

        logger.info("✅ 스트리밍 답변 생성 완료")

    except Exception as e:
        logger.error(f"스트리밍 답변 생성 오류: {e}")
        yield f"⚠️ 시스템 오류가 발생했습니다: {str(e)}"


async def generate_streaming_response(question: str, session_id: str, prompt_variant: str = "user_focused") -> \
AsyncGenerator[str, None]:
    """SSE 형식으로 스트리밍 응답 생성"""
    try:
        # 응답 시작 신호
        yield f"data: {json.dumps({'type': 'start',  'timestamp': datetime.now().isoformat()})}\n\n"

        # 스트리밍 답변 생성
        full_response = ""
        async for chunk in streaming_answer_question(question, session_id):
            if chunk:
                full_response += chunk

                # 각 청크를 SSE 형식으로 전송
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

                # 약간의 딜레이 (선택사항)
                await asyncio.sleep(0.01)

        # 응답 완료 신호
        yield f"data: {json.dumps({'type': 'end', 'full_response': full_response, 'timestamp': datetime.now().isoformat()})}\n\n"

    except Exception as e:
        logger.error(f"SSE 스트리밍 응답 생성 오류: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


def initialize_rag_system() -> bool:
    """RAG 시스템 초기화"""
    logger.info("RAG 시스템 초기화 중...")
    try:
        success = rebuild_chain()
        if success:
            logger.info("✅ RAG 시스템 초기화 성공")
        else:
            logger.error("❌ RAG 시스템 초기화 실패")

        return success
    except Exception as e:
        logger.error(f"초기화 중 오류: {e}")
        return False


if __name__ == "__main__":
    print('실행됨')