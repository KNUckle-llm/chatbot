from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from src.agent.state import CustomState
from src.agent.utils import initialize_components, detect_language
from src.agent.prompts import HITL_PROMPT, SYSTEM_PROMPT
from ..core.logger import get_logger

logger = get_logger(__name__)
model, store, retriever_tool = initialize_components()


def language_detection_node(state: CustomState):
    last_msg = state.get("messages")[-1]
    text = str(last_msg.content)
    state["language"] = detect_language(text)
    logger.info(f"Detected language: {state['language']}")
    return {"language": state["language"]}


def generate_query_or_response_node(state: CustomState):
    last_msg = state.get("messages")[-1]
    prompt = (
        f"질문: {last_msg.content}\n"
        "이 질문이 적절하면 'yes', 부적절하면 'no'로 판단하세요."
        "'no'라면, 질문을 기반으로 아래 내용을 근거하여 역질문을 하여 사용자의 질문을 유도합니다."
        f"{HITL_PROMPT.format(language=state.get('language', 'ko'))}\n\n"
    )
    response = model.invoke([SystemMessage(content=prompt)])
    eval_text = response.content.strip().lower()
    state["last_question_evaluation"] = eval_text
    state["question_appropriate"] = eval_text.startswith("yes")
    if not state["question_appropriate"]:
        state["question_reason"] = eval_text
    logger.info(f"Question appropriateness: {state['question_appropriate']}")
    return {"messages": state.get("messages")}


def route_before_retrieval_node(state: CustomState) -> Literal["retrieve", "rewrite_question"]:
    return "retrieve" if state.get("question_appropriate") else "rewrite_question"


def collect_documents_node(state: CustomState, max_docs: int = 3):
    """질문별 ToolNode 결과만 documents로 저장"""
    logger.info("Collecting documents for current question...")
    new_docs = [
        {"content": str(msg.content).strip()}
        for msg in state.get("messages", [])
        if (getattr(msg, "role", None) or getattr(msg, "type", None)) == "tool"
        and getattr(msg, "content", None)
    ][:max_docs]
    state["documents"] = new_docs
    logger.info(f"Collected {len(new_docs)} documents.")
    return {"documents": new_docs}


def rewrite_question_node(state: CustomState):
    if state.get("question_appropriate"):
        return {"messages": state.get("messages")}
    reason = state.get("question_reason", "이유 없음")
    prompt = (
        f"{HITL_PROMPT.format(language=state.get('language', 'ko'))}\n"
        f"질문이 부적절한 이유: {reason}\n"
        "질문을 적절하게 재작성하거나 안내 메시지를 제공해주세요."
    )
    response = model.invoke([SystemMessage(content=prompt)])
    state.get("messages").append(response)
    logger.info("Rewritten question/feedback added.")
    return {"messages": state.get("messages")}


def generation_node(state: CustomState):
    language = state.get("language", "ko")
    documents = state.get("documents", [])
    summarization = state.get("summarization", "")
    last_msg = state.get("messages")[-1]

    # 시스템 메시지 생성
    system_message = SYSTEM_PROMPT.format(
        input=last_msg.content,
        documents=documents
    )

    # 이전 대화 요약이 존재하면 앞에 포함
    if summarization:
        system_message = f"이전 대화 요약:\n{summarization}\n\n" + system_message

    # LLM 호출
    response = model.invoke([SystemMessage(content=system_message)])
    state.get("messages").append(response)
    return {"messages": state.get("messages")}



def summarization_node(state: CustomState):
    messages = state.get("messages")
    summary_prompt = "대화를 요약하세요:\n" + "\n".join([msg.content for msg in messages])
    response = model.invoke([SystemMessage(content=summary_prompt)])

    delete_msgs = [RemoveMessage(id=msg.id) for msg in messages[:-8]]
    state["summarization"] = str(response.content).strip()
    logger.info("Conversation summarized.")
    return {"summarization": state["summarization"], "messages": delete_msgs}