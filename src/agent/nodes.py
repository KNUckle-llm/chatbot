from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ..core.logger import get_logger
from .state import CustomState
from .utils import initialize_components, detect_language
from .prompts import SYSTEM_PROMPT, HITL_PROMPT

logger = get_logger(__name__)
model, store, retriever_tool_structured, RetrieverToolNode = initialize_components()


# ---------------------------
# 노드 정의
# ---------------------------

def language_detection_node(state: CustomState):
    logger.info("🔹 [language_detection_node] 실행")
    user_message = state.get("messages")[-1]
    state.set("language", detect_language(str(user_message.content)))
    return state

def route_before_retrieval_node(state: CustomState) -> bool:
    """질문이 모호하면 True, 명확하면 False (rewrite_question vs retrieve)"""
    messages = state.get("messages", [])
    user_message = next((msg for msg in reversed(messages) if getattr(msg, "role", None) == "user"), None)
    question_text = str(user_message.content).strip() if user_message else ""
    if not question_text:
        state.set("unclear_reason", "질문이 비어있습니다. 명확하게 질문해 주세요.")
        return True
    
    # LLM에게 질문 평가
    eval_prompt = f"""
    사용자가 보낸 질문을 평가하여 반드시 'yes' 또는 'no'로 판단하세요. 
    - 'yes'는 질문이 충분히 구체적이거나 일부 정보가 부족해도 답변 가능할 때.  
    - 'no'는 질문이 챗봇 사용자, 챗봇 제공 정보와 전혀 관련이 없거나 불명확하고 질문이 너무 짧을 때.
         이 경우, 들어온 질문인 "{question_text}"에 기반하여 아래 정보를 근거로 역질문을 생성합니다.
    
    챗봇의 사용자 : 공주대학교 SW 사업단 주관학과 학생들(컴퓨터공학과, 소프트웨어학과, 인공지능학부, 스마트정보기술공학과)
    챗봇이 제공하는 정보 : 학과 정보(학과별 교과과정표, 학과별 교수님 정보, 학과별 공지사항, 학과별 자료/서식, SW사업단 소식, SW사업단 혜택, SW사업단 공지사항, SW사업단 대회정보)
    """
    try:
        eval_response = model.invoke([SystemMessage(content=eval_prompt)])
        unclear = "no" in str(eval_response.content).lower()
        if unclear:
            state.set("unclear_reason", "질문이 명확하지 않습니다.")
        return unclear
    except Exception as e:
        logger.error(f"LLM 평가 중 예외 발생: {e}")
        state.set("unclear_reason", "질문 평가 실패")
        return True


def collect_documents_node(state: CustomState) -> bool:
    """문서 존재 여부 반환(True=없음 → rewrite, False=있음 → generate)"""
    tool_msgs = [msg for msg in state.get("messages", []) if getattr(msg, "role", None) == "tool"]
    if not tool_msgs:
        state.set("documents", [])
        return True
    collected = []
    for msg in tool_msgs:
        docs = getattr(msg, "content", [])
        if isinstance(docs, list):
            collected.extend(docs)
    state.set("documents", collected[:3])
    return len(collected) == 0



def rewrite_question_node(state: CustomState):
    logger.info("🔹 [rewrite_question_node] HITL")
    unclear_info = state.get("unclear_reason", "질문을 명확하게 해주세요.")
    response_message = AIMessage(content=unclear_info)
    state.set("messages", state.get("messages") + [response_message])
    return state


def generation_node(state: CustomState):
    logger.info("🔹 [generation_node] 실행")
    messages = state.get("messages")
    user_message_obj = next((msg for msg in reversed(messages) if getattr(msg, "role", None) == "user"), None)
    if not user_message_obj:
        return state
    user_message = user_message_obj.content
    language = state.get("language")
    documents = state.get("documents", [])

    formatted_docs = ""
    for idx, doc in enumerate(documents, start=1):
        content = doc.get("content", "")
        url = doc.get("metadata", {}).get("url", "URL 없음")
        formatted_docs += f"문서 {idx}:\n  내용: {content}\n  출처: {url}\n\n"

    system_message = SYSTEM_PROMPT.format(documents=formatted_docs, input=user_message)
    final_message = f"Answer in {language}.\n\n" + system_message
    response = model.invoke([SystemMessage(content=final_message)])
    state.set("messages", messages + [AIMessage(content=response.content)])
    return state



def summarization_node(state: CustomState):
    logger.info("🔹 [summarization_node] 실행")
    messages = state.get("messages")
    summarization = state.get("summarization") or ""
    summary_message = (
        f"This is a summary of the conversation to date:\n{summarization}\nExtend with new messages above:"
        if summarization else
        "Create a summary of the conversation above:"
    )
    response = model.invoke(messages + [HumanMessage(content=summary_message)])
    state.set("summarization", str(response.content).strip())
    return state

