from typing import List, Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage

from src.agent.state import CustomState
from src.agent.utils import (
    initialize_components,
    detect_language,
)
from src.agent.prompts import GRADE_PROMPT, HITL_PROMPT, SYSTEM_PROMPT
from ..core.logger import get_logger

logger = get_logger(__name__)
model, store, retriever_tool_structured, retriever_node_instance = initialize_components()


def language_detection_node(state: CustomState):
    logger.info("Detecting language...")
    message = str(state.get("messages")[-1].content)
    return {
        "language": detect_language(message)
    }


def generate_query_or_response_node(state: CustomState):
    """
    초기 LLM 메시지 생성 (단순 복사)
    """
    logger.info("Generating initial LLM message...")
    last_msg = state.get("messages")[-1]
    content = getattr(last_msg, "content", str(last_msg))
    response = HumanMessage(content=content)
    return {"messages": [response]}


def route_before_retrieval_node(state: CustomState) -> Literal["retrieve", "rewrite_question"]:
    """
    테스트용: 항상 retrieve 단계로 진행
    목적: 이 메시지를 기반으로 검색 필요 여부 판단
    rewrite 역질문 할지를 여기서 구현
    """
    return "retrieve"


def collect_documents_node(state: CustomState):
    """
    state['documents']에 저장된 문서들을 포맷
    generate 노드에서 바로 사용할 수 있도록 반환
    """
    documents: List[dict] = state.get("documents") or []
    if not documents:
        logger.warning("collect_documents_node: documents 비어 있음")
        return {"formatted_documents": ""}

    formatted_docs = []
    for idx, doc in enumerate(documents[:3], start=1):  # 최대 3개
        content = doc.get("content", "")
        meta  = doc.get("metadata", {})
        source = meta.get("url", "N/A")
        formatted_docs.append(
            f"문서 {idx}:\n"
            f"    내용: {content}\n"
            f"    출처: {source}\n"
        )
    return {"formatted_documents": "\n".join(formatted_docs)}


def rewrite_question_node(state: CustomState):
    logger.info("Rewriting question for HITL...")
    language = state.get("language", "ko")
    prompt = HITL_PROMPT.format(language=language)
    response = model.invoke([SystemMessage(content=prompt)])
    return {"messages": [response]}


def generation_node(state: CustomState):
    """
    state['messages']와 state['documents'] 기반으로 답변 생성
    """
    logger.info("Generating answer...")
    language = state.get("language")
    user_message = state.get("messages")[-1].content
    documents = state.get("documents") or []

    if documents:
        formatted_docs = collect_documents_node(state)["formatted_documents"]
        system_prompt = SYSTEM_PROMPT.format(input=user_message, documents=formatted_docs)
    else:
        summarization = state.get("summarization")
        system_prompt = (
            "Answer based on conversation history and summary.\n"
            f"Conversation Summary: {summarization}\n"
            f"Messages: {state.get('messages')}\n"
        )

    message = f"Answer the question in {language}. If en, use English; if ko, use Korean.\n" + system_prompt
    response = model.invoke([SystemMessage(content=message)])
    return {"messages": state.get("messages") + [response]}


def summarization_node(state: CustomState):
    """
    대화 요약 생성 및 오래된 메시지 삭제
    """
    logger.info("Summarizing conversation...")
    summarization = state.get("summarization")
    if summarization:
        summary_msg = (
            "This is a summary of the conversation to date:\n\n"
            f"{summarization}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_msg = "Create a summary of the conversation above:"

    messages = state.get("messages") + [HumanMessage(content=summary_msg)]
    response = model.invoke(messages)

    # 오래된 메시지 삭제 (8개 제외)
    delete_messages = [RemoveMessage(id=msg.id) for msg in state.get("messages")[:-8]]
    return {
        "summarization": str(response.content).strip(),
        "messages": delete_messages
    }