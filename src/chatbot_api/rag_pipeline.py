from urllib.parse import urlparse

from dotenv import load_dotenv
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from prompt_comparison.prompt_variants import PROMPT_VARIANTS
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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Global resources (pre-loaded for performance)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Original system message
system_message = """Hello! I'm your Kongju National University information assistant. 🏫

I'm here to help you find what you need using official KNU documents and resources.

I'll do my best to:
- Give you clear, useful answers
- Explain things step-by-step when needed
- Let you know if I need more information
- Point you to the right sources

**Please respond in Korean language.**

🔍 Based on the official documents I have access to:
{context}

📝 Your question: {input}

📋 Here's what I found:
[Answer with helpful explanations]

📚 **Reference Information:**
- Document: [filename]
- Source Department: [department]
- Link: [URL if available]

Need more specific information? Feel free to ask follow-up questions!"""

# Original prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("placeholder", "{memory}"),
    ("user", "🔍 검색된 문서:\n{context}"),
    ("human", "{input}"),
])

parser = StrOutputParser()
trimmer = trim_messages(max_tokens=500, token_counter=llm, strategy="last", include_system=True, start_on="human")

# Session store
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


def safe_str_conversion(obj: Any) -> str:
    """객체를 안전하게 문자열로 변환"""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        try:
            items = []
            for k, v in obj.items():
                if v is not None and str(v).strip():
                    items.append(f"{k}: {v}")
            return ", ".join(items) if items else ""
        except:
            return str(obj)
    try:
        return str(obj)
    except:
        return "[변환 불가능한 객체]"


def create_safe_document(content: Any, metadata: Any = None) -> Document:
    """안전한 Document 객체 생성"""
    safe_content = safe_str_conversion(content)
    if not safe_content.strip():
        safe_content = "[빈 문서]"

    safe_metadata = {}
    if metadata:
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                safe_key = safe_str_conversion(key)
                safe_value = safe_str_conversion(value)
                if safe_key and safe_value:
                    safe_metadata[safe_key] = safe_value
        else:
            safe_metadata["source"] = safe_str_conversion(metadata)

    return Document(page_content=safe_content, metadata=safe_metadata)


def format_context(docs: List[Document]) -> str:
    """검색된 문서들을 안전하게 포맷팅"""
    if not docs:
        return "검색된 문서가 없습니다."

    try:
        formatted_docs = []
        for i, doc in enumerate(docs[:3]):
            try:
                content = safe_str_conversion(doc.page_content)
                if len(content) > 500:
                    content = content[:500] + "..."

                metadata_str = ""
                if hasattr(doc, 'metadata') and doc.metadata:
                    meta_parts = []
                    for key, value in doc.metadata.items():
                        safe_value = safe_str_conversion(value)
                        if safe_value.strip():
                            meta_parts.append(f"{key}: {safe_value}")

                    if meta_parts:
                        metadata_str = f" [출처: {', '.join(meta_parts)}]"

                formatted_docs.append(f"문서 {i + 1}: {content}{metadata_str}")

            except Exception as e:
                logger.error(f"문서 {i + 1} 포맷팅 오류: {e}")
                formatted_docs.append(f"문서 {i + 1}: [포맷팅 오류]")

        return "\n\n".join(formatted_docs)

    except Exception as e:
        logger.error(f"문서 포맷팅 전체 오류: {e}")
        return "문서 포맷팅 중 오류가 발생했습니다."


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


def create_prompt_chain(prompt_variant_key: str):
    """특정 프롬프트 변형을 위한 체인 생성"""
    try:
        if prompt_variant_key not in PROMPT_VARIANTS:
            raise ValueError(f"Unknown prompt variant: {prompt_variant_key}")

        variant = PROMPT_VARIANTS[prompt_variant_key]

        # 프롬프트 템플릿 생성
        simple_prompt = ChatPromptTemplate.from_messages([
            ("user", variant["template"])
        ])

        # 컨텍스트 생성 함수
        def create_context_for_prompt(inputs):
            try:
                if isinstance(inputs, dict):
                    question = inputs.get("input", "")
                else:
                    question = str(inputs)

                search_results = retriever.invoke(question)
                context = format_context(search_results)

                return {
                    "input": question,
                    "context": context
                }
            except Exception as e:
                logger.error(f"컨텍스트 생성 오류 ({prompt_variant_key}): {e}")
                return {
                    "input": inputs.get("input", "") if isinstance(inputs, dict) else str(inputs),
                    "context": "검색 중 오류가 발생했습니다."
                }

        # 체인 구성
        context_chain = RunnableLambda(create_context_for_prompt)
        chain = context_chain | simple_prompt | llm | parser

        return chain

    except Exception as e:
        logger.error(f"프롬프트 체인 생성 실패 ({prompt_variant_key}): {e}")
        return None


def rebuild_chain(selected_dept: str = None) -> bool:
    """RAG 체인 재구성 - 모든 프롬프트 변형 포함"""
    global retriever, rag_chain, prompt_chains

    try:
        logger.info("RAG 체인 구성 시작...")

        if not test_db_connection():
            logger.error("데이터베이스 연결 실패")
            return False

        db_path = "./ChromaDB/knu_chroma_db_all"
        db = Chroma(persist_directory=db_path, embedding_function=hf_embeddings)

        try:
            results = db.get(include=["documents", "metadatas"])
            logger.info(f"DB에서 {len(results.get('documents', []))}개 문서 조회")
        except Exception as e:
            logger.error(f"문서 조회 실패: {e}")
            return False

        if not results.get("documents"):
            logger.error("조회된 문서가 없습니다")
            return False

        docs = []
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for i in range(len(documents)):
            try:
                doc_content = documents[i] if i < len(documents) else ""
                doc_metadata = metadatas[i] if i < len(metadatas) else {}

                if not safe_str_conversion(doc_content).strip():
                    continue

                safe_doc = create_safe_document(doc_content, doc_metadata)
                docs.append(safe_doc)

            except Exception as e:
                logger.warning(f"문서 {i} 처리 오류: {e}")
                continue

        logger.info(f"유효한 문서 {len(docs)}개 생성")

        if not docs:
            logger.error("유효한 문서가 없습니다")
            return False

        # 부서별 필터링
        if selected_dept:
            filtered_docs = []
            for doc in docs:
                try:
                    metadata_str = safe_str_conversion(doc.metadata).lower()
                    if selected_dept.lower() in metadata_str:
                        filtered_docs.append(doc)
                except:
                    continue

            if filtered_docs:
                docs = filtered_docs
                logger.info(f"부서 '{selected_dept}'로 필터링: {len(docs)}개 문서")

        # Vector Retriever 생성
        try:
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            logger.info("Vector retriever 생성 완료")
        except Exception as e:
            logger.error(f"Retriever 생성 실패: {e}")
            return False

        # 각 프롬프트 변형에 대한 체인 생성
        prompt_chains = {}
        for variant_key in PROMPT_VARIANTS.keys():
            try:
                chain = create_prompt_chain(variant_key)
                if chain:
                    prompt_chains[variant_key] = chain
                    logger.info(f"프롬프트 체인 생성 완료: {variant_key}")
                else:
                    logger.warning(f"프롬프트 체인 생성 실패: {variant_key}")
            except Exception as e:
                logger.error(f"프롬프트 체인 생성 오류 ({variant_key}): {e}")

        # 원본 체인도 유지
        def create_simple_context(inputs):
            try:
                if isinstance(inputs, dict):
                    question = inputs.get("input", "")
                else:
                    question = str(inputs)

                search_results = retriever.invoke(question)
                context = format_context(search_results)

                return {
                    "input": question,
                    "context": context
                }
            except Exception as e:
                logger.error(f"컨텍스트 생성 오류: {e}")
                return {
                    "input": inputs.get("input", "") if isinstance(inputs, dict) else str(inputs),
                    "context": "검색 중 오류가 발생했습니다."
                }

        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "🔍 검색된 문서:\n{context}"),
            ("human", "{input}"),
        ])

        context_chain = RunnableLambda(create_simple_context)
        simple_chain = context_chain | simple_prompt | llm | parser

        rag_chain = RunnableWithMessageHistory(
            simple_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="memory"
        )

        logger.info(f"RAG 체인 구성 완료 (프롬프트 변형 {len(prompt_chains)}개)")
        return True

    except Exception as e:
        logger.error(f"RAG 체인 구성 실패: {e}")
        return False


def answer_with_prompt_variant(question: str, prompt_variant_key: str) -> str:
    """특정 프롬프트 변형으로 질문에 답변"""
    try:
        if prompt_variant_key not in prompt_chains:
            return f"⚠️ 프롬프트 변형 '{prompt_variant_key}'을 찾을 수 없습니다."

        safe_question = safe_str_conversion(question)
        if not safe_question.strip():
            return "⚠️ 유효한 질문을 입력해주세요."

        logger.info(f"프롬프트 변형 '{prompt_variant_key}'로 답변 생성: {safe_question[:50]}...")

        chain = prompt_chains[prompt_variant_key]
        response = chain.invoke({"input": safe_question})

        return safe_str_conversion(response)

    except Exception as e:
        logger.error(f"프롬프트 변형 답변 생성 실패 ({prompt_variant_key}): {e}")
        return f"⚠️ 답변 생성 중 오류가 발생했습니다: {str(e)}"


def compare_prompt_responses(question: str, prompt_variants: List[str] = None) -> Dict[str, Any]:
    """여러 프롬프트 변형으로 같은 질문에 대한 답변 비교"""
    try:
        if prompt_variants is None:
            prompt_variants = list(PROMPT_VARIANTS.keys())

        # 유효하지 않은 프롬프트 변형 필터링
        valid_variants = [v for v in prompt_variants if v in PROMPT_VARIANTS]

        if not valid_variants:
            return {
                "error": "유효한 프롬프트 변형이 없습니다.",
                "available_variants": list(PROMPT_VARIANTS.keys())
            }

        logger.info(f"프롬프트 비교 시작: {question[:50]}... (변형 {len(valid_variants)}개)")

        # RAG 체인이 없으면 초기화
        if not prompt_chains:
            if not rebuild_chain():
                return {"error": "RAG 시스템 초기화에 실패했습니다."}

        comparison_results = {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "responses": {},
            "summary": {
                "total_variants": len(valid_variants),
                "successful_responses": 0,
                "failed_responses": 0
            }
        }

        # 각 프롬프트 변형으로 답변 생성
        for variant_key in valid_variants:
            try:
                logger.info(f"프롬프트 변형 처리 중: {variant_key}")

                start_time = datetime.now()
                response = answer_with_prompt_variant(question, variant_key)
                end_time = datetime.now()

                response_time = (end_time - start_time).total_seconds()

                comparison_results["responses"][variant_key] = {
                    "name": PROMPT_VARIANTS[variant_key]["name"],
                    "response": response,
                    "response_time": response_time,
                    "success": not response.startswith("⚠️"),
                    "character_count": len(response),
                    "timestamp": start_time.isoformat()
                }

                if not response.startswith("⚠️"):
                    comparison_results["summary"]["successful_responses"] += 1
                else:
                    comparison_results["summary"]["failed_responses"] += 1

            except Exception as e:
                logger.error(f"프롬프트 변형 '{variant_key}' 처리 실패: {e}")
                comparison_results["responses"][variant_key] = {
                    "name": PROMPT_VARIANTS[variant_key]["name"],
                    "response": f"⚠️ 오류: {str(e)}",
                    "response_time": 0,
                    "success": False,
                    "character_count": 0,
                    "timestamp": datetime.now().isoformat()
                }
                comparison_results["summary"]["failed_responses"] += 1

        # 성공률 계산
        total = comparison_results["summary"]["total_variants"]
        successful = comparison_results["summary"]["successful_responses"]
        comparison_results["summary"]["success_rate"] = (successful / total) * 100 if total > 0 else 0

        logger.info(f"프롬프트 비교 완료: {successful}/{total} 성공")
        return comparison_results

    except Exception as e:
        logger.error(f"프롬프트 비교 실패: {e}")
        return {"error": f"프롬프트 비교 중 오류가 발생했습니다: {str(e)}"}


def format_comparison_results(comparison_results: Dict[str, Any]) -> str:
    """비교 결과를 읽기 쉬운 형태로 포맷팅"""
    try:
        if "error" in comparison_results:
            return f"❌ 오류: {comparison_results['error']}"

        output = []
        output.append("=" * 80)
        output.append(f"📊 프롬프트 비교 결과")
        output.append("=" * 80)
        output.append(f"🔍 질문: {comparison_results['question']}")
        output.append(f"⏰ 실행 시간: {comparison_results['timestamp']}")
        output.append(
            f"📈 성공률: {comparison_results['summary']['success_rate']:.1f}% ({comparison_results['summary']['successful_responses']}/{comparison_results['summary']['total_variants']})")
        output.append("")

        # 각 프롬프트 변형 결과 출력
        for variant_key, result in comparison_results["responses"].items():
            output.append("-" * 60)
            output.append(f"🏷️  {result['name']} ({variant_key})")
            output.append("-" * 60)
            output.append(f"✅ 성공: {'예' if result['success'] else '아니오'}")
            output.append(f"⏱️  응답 시간: {result['response_time']:.2f}초")
            output.append(f"📝 문자 수: {result['character_count']:,}자")
            output.append("")
            output.append("📋 응답 내용:")
            output.append(result['response'])
            output.append("")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"비교 결과 포맷팅 실패: {e}")
        return f"❌ 결과 포맷팅 중 오류가 발생했습니다: {str(e)}"


def save_comparison_results(comparison_results: Dict[str, Any], filename: str = None) -> str:
    """비교 결과를 JSON 파일로 저장"""
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prompt_comparison_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)

        logger.info(f"비교 결과 저장 완료: {filename}")
        return filename

    except Exception as e:
        logger.error(f"비교 결과 저장 실패: {e}")
        return f"❌ 저장 실패: {str(e)}"


# 기존 함수들 유지
def answer_question(question: str, session_id: str) -> str:
    """원래 방식으로 질문에 대한 답변 생성"""
    try:
        if not session_id:
            return "⚠️ 세션을 먼저 생성하거나 선택해주세요."

        if rag_chain is None:
            logger.info("RAG 체인이 없습니다. 구성 중...")
            if not rebuild_chain():
                return "⚠️ RAG 시스템 초기화에 실패했습니다. 문서 데이터베이스를 확인해주세요."

        init_session(session_id)

        safe_question = safe_str_conversion(question)
        if not safe_question.strip():
            return "⚠️ 유효한 질문을 입력해주세요."

        logger.info(f"질문 처리 중: {safe_question[:50]}...")

        try:
            response = rag_chain.invoke(
                {"input": safe_question},
                config={"configurable": {"session_id": session_id}}
            )

            logger.info("✅ 답변 생성 성공")
            return safe_str_conversion(response)

        except Exception as chain_error:
            logger.error(f"체인 실행 중 오류: {chain_error}")

            # 폴백: 간단한 방식으로 답변 생성
            logger.info("폴백 모드로 답변 생성 시도")
            try:
                search_results = retriever.invoke(safe_question)
                context = format_context(search_results)

                simple_prompt = f"""당신은 공주대학교 정보 안내 AI입니다.
아래 문서를 바탕으로 질문에 답해주세요.

검색된 문서:
{context}

질문: {safe_question}

답변:"""

                response = llm.invoke(simple_prompt)
                logger.info("✅ 폴백 모드 답변 생성 성공")
                return safe_str_conversion(response.content if hasattr(response, 'content') else response)

            except Exception as fallback_error:
                logger.error(f"폴백 모드도 실패: {fallback_error}")
                return f"⚠️ 답변 생성 중 오류가 발생했습니다. 검색된 문서는 {len(search_results) if 'search_results' in locals() else 0}개입니다."

    except Exception as e:
        logger.error(f"전체 프로세스 오류: {e}")
        return f"⚠️ 시스템 오류가 발생했습니다: {str(e)}"


def debug_search(query: str, max_results: int = 3) -> List[Document]:
    """검색 기능 디버깅"""
    try:
        db_path = "./ChromaDB/knu_chroma_db_all"
        db = Chroma(persist_directory=db_path, embedding_function=hf_embeddings)

        safe_query = safe_str_conversion(query)
        logger.info(f"디버그 검색: '{safe_query}'")

        vector_results = db.similarity_search(safe_query, k=max_results)
        logger.info(f"검색 결과: {len(vector_results)}개")

        for i, doc in enumerate(vector_results):
            try:
                content = safe_str_conversion(doc.page_content)[:100]
                metadata = safe_str_conversion(doc.metadata)
                logger.info(f"결과 {i + 1}: {content}...")
                logger.info(f"메타데이터: {metadata}")
            except Exception as e:
                logger.error(f"결과 {i + 1} 처리 오류: {e}")

        return vector_results

    except Exception as e:
        logger.error(f"디버그 검색 실패: {e}")
        return []



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


# 새로운 사용 예시 함수들
def demo_prompt_comparison():
    """프롬프트 비교 데모"""
    print("🚀 프롬프트 비교 시스템 데모 시작")

    # 시스템 초기화
    if not initialize_rag_system():
        print("❌ 시스템 초기화 실패")
        return

    # 테스트 질문들
    test_questions = [
        "공주대 공과대학 학과 목록",
        "소프트웨어학과와 컴퓨터공학과의 차이점",
        "공주대 천안캠퍼스의 공학관은 어디어디가 있는지?",
        "신소재공학부의 전공은 무엇이 있는지?",
        "소프트웨어학과 졸업하고 싶어"
    ]

    for question in test_questions:
        print(f"\n🔍 질문: {question}")
        print("=" * 50)

        # 모든 프롬프트 변형으로 비교
        results = compare_prompt_responses(question)

        # 결과 출력
        formatted_results = format_comparison_results(results)
        print(formatted_results)

        # 결과 저장
        filename = save_comparison_results(results)
        print(f"📁 결과 저장: {filename}")

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    demo_prompt_comparison()