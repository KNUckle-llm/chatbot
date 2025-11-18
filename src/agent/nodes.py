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
        "이 질문이 챗봇 사용자와 챗봇의 벡터 DB에 저장된 문서 정보를 기준으로 적절하면 'yes', 부적절하면 'no'로 판단하세요."
        "반드시 yes/no 중에 하나가 답변으로 있어야 합니다."
        "'no'라면 부적절한 이유를 1~2문장으로 구체적으로 작성하세요.\n"
        f"{HITL_PROMPT.format(language=state.get('language', 'ko'))}\n\n"
    )
    response = model.invoke([SystemMessage(content=prompt)])
    raw_text = response.content.strip()
    
    # yes/no 판단
    eval_text = "yes" if raw_text.lower().startswith("yes") else "no"
    state["last_question_evaluation"] = raw_text
    state["question_appropriate"] = eval_text == "yes"

    # 부적절한 경우 이유만 저장
    if not state["question_appropriate"]:
        reason = raw_text[len("no"):].strip()  # "no" 이후 내용을 부적절 이유로 사용
        state["question_reason"] = reason

    # 디버그 로그
    logger.info("===== Question Evaluation Debug =====")
    logger.info(f"User Question: {last_msg.content}")
    logger.info(f"Raw LLM Response: {raw_text}")
    logger.info(f"question_appropriate: {state['question_appropriate']}")
    if not state["question_appropriate"]:
        logger.info(f"question_reason: {state['question_reason']}")
    logger.info("===================================")

    return {
        "messages": state.get("messages"),
        "question_appropriate": state.get("question_appropriate"),
        "question_reason": state.get("question_reason", None)
    }



def route_before_retrieval_node(state: CustomState) -> Literal["retrieve", "rewrite_question"]:
    # 안전하게 기본값 False 지정
    is_appropriate = state.get("question_appropriate")
    if is_appropriate is None:
        logger.warning("question_appropriate가 None입니다. 기본값 False로 처리합니다.")
        is_appropriate = True
        state["question_appropriate"] = is_appropriate

    logger.info(f"Routing decision - question_appropriate: {is_appropriate}")
    return "retrieve" if is_appropriate else "rewrite_question"

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
        # 적절한 질문이면 그냥 메시지 반환
        return {"messages": state.get("messages")}

    last_msg = state.get("messages")[-1]
    reason = state.get("question_reason", "부적절한 이유 없음")

    # 사용자에게 질문을 어떻게 재작성하면 좋을지 안내
    prompt = (
        f"사용자가 한 질문: {last_msg.content}\n"
        f"부적절한 이유: {reason}\n\n"
        "사용자에게 보여줄 안내 메시지를 작성하세요. 형식은 다음과 같아야 합니다:\n"
        "1. 첫 문단: '질문은 다음과 같은 이유로 부적절합니다. 질문을 다시 입력해주세요.'\n"
        "   이어서 실제 부적절한 이유를 서술.\n"
        "2. 두 번째 문단: '그 밑에 이렇게 질문하는건 어떨까요?' 형식으로, "
        "   사용자가 입력한 질문을 기반으로 조금 더 구체적이고 적절하게 만든 1~2개의 질문 예시 제공."
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