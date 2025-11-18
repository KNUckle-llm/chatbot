from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage

from src.agent.state import CustomState
from src.agent.utils import (
    initialize_components,
    detect_language,
)
from src.agent.prompts import GRADE_PROMPT, HITL_PROMPT, SYSTEM_PROMPT
from ..core.logger import get_logger

logger = get_logger(__name__)
model, store, retriever_tool_structured, RetrieverToolNode = initialize_components()


def language_detection_node(state: CustomState):
    logger.info("🔹 [language_detection_node] 시작")
    user_message = state.get("messages")[-1]
    state.set("language", detect_language(str(user_message.content)))
    return state

def route_before_retrieval_node(state: CustomState) -> Literal["retrieve", "rewrite_question"]:
    logger.info("🔹 [route_before_retrieval_node] 시작")
    user_message = state.get("messages")[-1]
    question_text = str(user_message.content).strip()
    
    if not question_text:
        logger.info("질문이 비어있음 → rewrite_question")
        state.set("unclear_reason", "질문이 비어있습니다. 명확하게 질문해 주세요.")
        return "rewrite_question"
    
    # LLM에게 질문 평가
    eval_prompt = f"""
    사용자가 보낸 질문을 평가하여 'yes' 또는 'no'로 판단하세요.  
    - 'yes'는 질문이 충분히 구체적이거나 일부 정보가 부족해도 답변 가능할 때.  
    - 'no'는 질문이 챗봇 사용자, 챗봇 제공 정보와 전혀 관련이 없거나 불명확하고 질문이 너무 짧을 때.
         이 경우, 들어온 질문인 "{question_text}"에 기반하여 아래 정보를 근거로 역질문을 생성합니다.
    
    챗봇의 사용자 : 공주대학교 SW 사업단 주관학과 학생들(컴퓨터공학과, 소프트웨어학과, 인공지능학부, 스마트정보기술공학과)
    챗봇이 제공하는 정보 : 학과 정보(학과별 교과과정표, 학과별 교수님 정보, 학과별 공지사항, 학과별 자료/서식, SW사업단 소식, SW사업단 혜택, SW사업단 공지사항, SW사업단 대회정보)
    """
    eval_response = model.invoke([SystemMessage(content=eval_prompt)])
    logger.info(f"LLM 응답: {eval_response.content}")
    
    # 평가 결과를 AIMessage로 state에 기록
    state.set("messages", state.get("messages") + [AIMessage(content=eval_response.content)])
    
    unclear = "no" in str(eval_response.content).lower()
    logger.info(f"질문 모호 여부 판단: {unclear}")
    
    if unclear:
        logger.info("→ rewrite_question 경로 선택")
        state.set("unclear_reason", str(eval_response.content))
        return "rewrite_question"
    
    # 명확하면 바로 retrieve 경로
    logger.info("→ retrieve 경로 선택")
    return "retrieve"
    

def collect_documents_node(state: CustomState):
    logger.info("🔹 [collect_documents_node] 시작")

    tool_msgs = [msg for msg in state.get("messages", []) if getattr(msg, "role", None) == "tool"]

    # 검색 결과 없음
    if not tool_msgs:
        logger.info("No tool outputs found. Redirecting to rewrite_question.")
        state.set("no_docs", True)  # 검색 결과 없음을 표시
        return state

    collected = []
    for msg in tool_msgs:
        try:
            docs = msg.content
            if isinstance(docs, list):
                collected.extend(docs)
        except Exception as e:
            logger.error(f"Failed to parse tool output: {e}")

    state.set("documents", collected[:3])
    state.set("no_docs", False)  # 문서 있음 표시
    logger.info(f"Collected {len(collected)} documents.")
    return state



def rewrite_question_node(state: CustomState):
    logger.info("🔹 [rewrite_question_node] HITL 시작")
    unclear_info = state.get("unclear_reason", "질문을 명확하게 해주세요.")

    # LLM 호출 없이 바로 AIMessage 생성
    response_message = AIMessage(content=f"{unclear_info}")
    messages = state.get("messages") + [response_message]
    state.set("messages", messages)
    
    return state


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

    system_message = SYSTEM_PROMPT.format(documents=formatted_docs, input=user_message)
    final_message = language_message + system_message
    response = model.invoke([SystemMessage(content=final_message)])

    messages = state.get("messages") + [AIMessage(content=response.content)]
    state.set("messages", messages)
    return state



def summarization_node(state: CustomState):
    logger.info("🔹 [summarization_node] 시작")
    summarization = state.get("summarization") or ""
    messages = state.get("messages")

    if summarization:
        summary_message = f"This is a summary of the conversation to date:\n{summarization}\nExtend considering new messages above:"
    else:
        summary_message = "Create a summary of the conversation above:"

    response = model.invoke(messages + [HumanMessage(content=summary_message)])
    state.set("summarization", str(response.content).strip())
    state.set("messages", messages[-8:])  # 최신 8개만 유지
    return state
