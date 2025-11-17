from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from src.agent.state import CustomState
from src.agent.utils import (
    initialize_components,
    detect_language,
)
from src.agent.prompts import GRADE_PROMPT, HITL_PROMPT, SYSTEM_PROMPT
from ..core.logger import get_logger

logger = get_logger(__name__)
model, store, retriever_tool = initialize_components()


def language_detection_node(state: CustomState):
    logger.info("🔹 [language_detection_node] 시작")
    message = str(state.get("messages")[-1].content)
    return {
        "language": detect_language(message)
    }

def route_before_retrieval_node(state: CustomState) -> Literal["retrieve", "rewrite_question"]:
    """
    1. 질문 명확성 평가
    2. 불명확하면 rewrite_question(HITL)로 이동
    3. 명확하면 retrieve 경로로 진행 (Tool 호출/문서 존재 여부는 나중에 판단)
    """
    logger.info("🔹 [route_before_retrieval_node] 시작")
    message = state.get("messages")[-1]
    question_text = str(message.content).strip()
    
    # 1. 질문 명확성 평가
    if not question_text:
        logger.info("질문이 비어있음 → rewrite_question")
        return "rewrite_question"
    
    # LLM에게 질문 평가
    eval_prompt = f"""
    사용자가 보낸 질문을 평가하세요.
    공주대학교 SW 사업단 학생들(컴퓨터공학과, 소프트웨어학과, 인공지능학부, 스마트정보기술공학과)을 위한 챗봇입니다.
    벡터 DB에는 학과정보 (학과별 교과과정표, 학과 교수님 정보, 학과별 공지사항, 학과별 자료/서식, 규정집, SW사업단 소식, SW사업단 혜택, SW사업단 소식, SW사업단 공지사항, TOPCIT같은 대회 일정 등)가 들어 있습니다.
    - 질문이 대부분 이해 가능하고 핵심 정보가 포함되어 있다면 'yes'
    - 질문이 불분명하거나 필요한 정보가 거의 없으면 'no'

    질문이 'no'일 경우, 추가로 어떤 정보를 포함하면 충분할지 간단히 안내만 하세요.
    예시처럼 엄격한 형식이나 세부 템플릿을 요구하지 마세요.

    질문:
    {question_text}
    """
    eval_response = model.invoke([SystemMessage(content=eval_prompt)])
    logger.info(f"LLM 응답: {eval_response.content}")
    
    unclear = "no" in str(eval_response.content).lower()
    logger.info(f"질문 모호 여부 판단: {unclear}")
    
    if unclear:
        logger.info("→ rewrite_question 경로 선택")
        state["unclear_reason"] = str(eval_response.content)
        return "rewrite_question"
    
    # 명확하면 바로 retrieve 경로
    logger.info("→ retrieve 경로 선택")
    return "retrieve"
    

def collect_documents_node(state: CustomState):
    logger.info("🔹 [collect_documents_node] 시작")

    # ToolNode 메시지 필터링
    tool_msgs = [
        msg for msg in state.get("messages", [])
        if getattr(msg, "role", None) == "tool"
    ]

    # 검색 결과 없음 → 질문이 너무 모호한 경우
    if not tool_msgs:
        logger.info("No tool outputs found. Redirecting to rewrite_question.")
        return {"next_node": "rewrite_question"}

    collected = []

    for msg in tool_msgs:
        try:
            # retriever_tool 이 반환한 리스트 그대로 있음
            docs = msg.content   # 이미 [{"content":..., "metadata":...}, ...]
            
            # 리스트인지 확인
            if isinstance(docs, list):
                collected.extend(docs)
            else:
                logger.warning("Tool output is not a list. Skipping.")
        except Exception as e:
            logger.error(f"Failed to parse tool output: {e}")

    # 최대 3개만 유지 (retriever 기본 k=3이지만 혹시 중복도 대비)
    collected = collected[:3]

    logger.info(f"Collected {len(collected)} documents.")

    return {"documents": collected}



def rewrite_question_node(state: CustomState):
    logger.info("🔹 [rewrite_question_node] 시작")
    logger.info("Rewriting question for HITL...")
    language = state.get("language")
    unclear_info = state.get("unclear_reason", "")
    
    prompt = HITL_PROMPT.format(language=language)
    # 역질문 안내 메시지 추가
    if unclear_info:
        prompt += f"\n\n질문이 모호한 이유 : {unclear_info}"

    response = model.invoke([{"role": "system", "content": prompt}])
    return {"messages": [response]}


def generation_node(state: CustomState):
    logger.info("🔹 [generation_node] 시작")

    # 언어 정보 가져오기
    language = state.get("language")
    user_message = state["messages"][-1].content

    # 문서 가져오기
    documents = state.get("documents", [])

    # 언어 안내 메시지
    language_message = (
        f"Answer the question in {language}. "
        "If en, answer in English; if ko, answer in Korean.\n\n"
    )

    # 문서 포맷팅
    formatted_docs = ""
    for idx, doc in enumerate(documents, start=1):
        content = doc.get("content", "")
        url = doc.get("metadata", {}).get("url", "URL 없음")
        formatted_docs += (
            f"   문서 {idx}:\n"
            f"       내용: {content}\n"
            f"       출처: {url}\n\n"
        )

    # SYSTEM_PROMPT에 문서와 사용자 질문 삽입
    system_message = SYSTEM_PROMPT.format(
        documents=formatted_docs,
        input=user_message
    )

    # LLM 호출
    final_message = language_message + system_message
    response = model.invoke([SystemMessage(content=final_message)])

    return {"messages": state["messages"] + [response]}



def summarization_node(state: CustomState):
    logger.info("🔹 [summarization_node] 시작")
    summarization = state.get("summarization")
    logger.info(f"🔹 이전 요약 길이: {len(summarization)}")

    if summarization:
        summary_message = (
            "This is a summary of the conversation to date:\n\n"
            f"{summarization}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state.get("messages") + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but the 8 most recent messages
    delete_messages = [
        RemoveMessage(id=msg.id) for msg in state.get("messages")[:-8]
    ]
    return {
        "summarization": str(response.content).strip(),
        "messages": delete_messages
    }
