# nodes_rag_safe.py
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from src.agent.state import CustomState
from src.agent.utils import initialize_components, detect_language
from src.agent.prompts import HITL_PROMPT, SYSTEM_PROMPT
from ..core.logger import get_logger

logger = get_logger(__name__)
model, store, retriever_tool = initialize_components()


def language_detection_node(state: CustomState):
    logger.info(">>> [NODE] language_detection_node START")
    last_msg = state.get("messages")[-1]
    text = str(last_msg.content)
    state["language"] = detect_language(text)
    logger.info(f"Detected language: {state['language']}")
    return {"language": state["language"]}


def generate_query_or_response_node(state: CustomState):
    """
    질문 적절성 판단
    """
    logger.info(">>> [NODE] generate_query_or_response_node START")
    last_msg = state.get("messages")[-1]

    prompt = (
        f"질문: {last_msg.content}\n"
        "이 질문이 공주대 챗봇 사용자의 문서 검색 목적에 적절한 질문인지 판단하세요.\n"
        "- 문서 검색 목적에 맞으면 'yes', 일상적이거나 적절하지 않으면 'no'를 가장 처음에 출력\n"
        "- 불명확하면 이유를 1~2문장으로 작성\n"
        "- 일상 대화, 인사 등은 'no'로 판단\n"
    )
    response = model.invoke([SystemMessage(content=prompt)])
    raw_text = response.content.strip()

    if raw_text.lower().startswith("no"):
        state["question_appropriate"] = False
        state["question_reason"] = raw_text[2:].strip()
    else:
        state["question_appropriate"] = True
        state["question_reason"] = None

    logger.info(f"User Question: {last_msg.content}")
    logger.info(f"Raw LLM Response: {raw_text}")
    logger.info(f"question_appropriate: {state['question_appropriate']}")
    if not state["question_appropriate"]:
        logger.info(f"question_reason: {state['question_reason']}")
    return {
        "question_appropriate": state["question_appropriate"],
        "question_reason": state.get("question_reason", None)
    }


def route_before_retrieval_node(state: CustomState) -> Literal["retrieve", "rewrite_question"]:
    logger.info(">>> [NODE] route_before_retrieval_node START")
    is_appropriate = state.get("question_appropriate")
    if is_appropriate is None:
        logger.warning("question_appropriate가 None입니다. 기본값 False로 처리합니다.")
        is_appropriate = False
        state["question_appropriate"] = is_appropriate
    logger.info(f"Routing decision - question_appropriate: {is_appropriate}")
    return "retrieve" if is_appropriate else "rewrite_question"


def retrieve_documents_node(state: CustomState, max_docs: int = 3):
    logger.info(">>> [NODE] retrieve_documents_node START")
    last_msg = state.get("messages")[-1]
    query = str(last_msg.content).strip()

    departments = [
        "소프트웨어학과",
        "컴퓨터공학과",
        "공주대학교",
        "공주대학교 SW중심대학사업단",
        "SW중심대학사업단",
        "스마트정보기술공학과",
        "인공지능학부",
        "공주대학교 현장실습지원센터"
    ]

    alias_map = {
        "공주대학교 SW중심대학사업단": ["공주대학교 SW중심대학사업단", "SW중심대학사업단"],
        "SW중심대학사업단": ["공주대학교 SW중심대학사업단", "SW중심대학사업단"],
    }

    dept_prompt = (
        f"사용자 질문: {query}\n"
        f"목록 중에서 관련 학과/부서를 하나 선택하세요:\n"
        f"{', '.join(departments)}\n"
        "출력은 반드시 목록 중 하나"
    )
    dept_response = model.invoke([SystemMessage(content=dept_prompt)])
    predicted_department = dept_response.content.strip()
    logger.info(f"Predicted department: {predicted_department}")

    if predicted_department in departments:
        aliases = alias_map.get(predicted_department, [predicted_department])
        filter_expr = {"department": {"$in": aliases}}
        logger.info(f"Using filter: {filter_expr}")
        docs = store.similarity_search(query, k=max_docs, filter=filter_expr)
    else:
        logger.info("Predicted department not recognized. Running search without filter.")
        docs = store.similarity_search(query, k=max_docs)

    # 문서 구조 그대로 유지
    state["documents"] = [
        {
            "content": d.page_content,
            "metadata": {
                "file_name": d.metadata.get("file_name"),
                "department": d.metadata.get("department"),
                "url": d.metadata.get("url"),
                "date": d.metadata.get("date")
            }
        }
        for d in docs
    ]
    logger.info(f"Retrieved {len(docs)} documents for query: {query}")
    return {"documents": state["documents"]}


def rewrite_question_node(state: CustomState):
    logger.info(">>> [NODE] rewrite_question_node START")
    if state.get("question_appropriate"):
        return {"messages": state.get("messages")}

    last_msg = state.get("messages")[-1]
    reason = state.get("question_reason", "불명확한 이유 없음")
    previous_summary = state.get("summarization", "")

    qtype_prompt = (
        f"질문: {last_msg.content}\n"
        "다음 중 하나로 분류하세요: "
        "1. 이전 대화 관련, "
        "2. 일반 일상 대화, "
        "3. 문서/정보 검색용 질문\n"
        "출력은 숫자(1,2,3)만 사용하세요."
    )
    qtype_resp = model.invoke([SystemMessage(content=qtype_prompt)])
    qtype = qtype_resp.content.strip()

    if qtype == "1":
        prompt = (
            f"사용자가 한 질문: {last_msg.content}\n"
            f"이전 대화 요약: {previous_summary}\n\n"
            "이 질문은 이전 대화와 관련 있습니다. "
            "사용자 질문에 대한 답변으로 이전 대화 요약을 참고하세요."
        )
    elif qtype == "2":
        # 일상 질문 안내문만 출력
        prompt = (
            "공주대 챗봇이므로 일상적인 대화에는 부적절합니다. "
            "다시 질문해주세요."
        )
    else:
        prompt = (
            f"사용자가 한 질문: {last_msg.content}\n"
            f"불명확한 이유: {reason}\n\n"
            "사용자에게 보여줄 안내 메시지를 작성하세요. 형식은 다음과 같아야 합니다:\n"
            "첫 문단: '질문은 다음과 같은 이유로 불명확합니다. 질문을 다시 입력해주세요.'\n"
            "두 번째 문단: 불명확한 이유를 서술하세요.\n"
            "세 번째 문단: '이렇게 질문하는건 어떨까요?' 형식으로 1~2개의 예시 질문 제공."
        )

    response = model.invoke([SystemMessage(content=prompt)])
    state.get("messages").append(response)
    logger.info("Rewritten question/feedback added.")
    return {"messages": state.get("messages")}


def generation_node(state: CustomState):
    logger.info(">>> [NODE] generation_node START")
    language = state.get("language", "ko")
    documents = state.get("documents", [])
    summarization = state.get("summarization", "")
    last_msg = state.get("messages")[-1]

    # 문서 내용 그대로 전달 + 개행 유지 + 문서 사이 빈 줄 추가
    docs_text = "\n\n---\n\n".join([
        f"문서 {i+1}\n"
        f"본문 내용:\n{d['content']}\n\n"
        f"제목:\n{d.get('metadata', {}).get('file_name', '')}\n\n"
        f"부서:\n{d.get('metadata', {}).get('department', '')}\n\n"
        f"작성일:\n{d.get('metadata', {}).get('date', '')}\n\n"
        f"출처:\n{d.get('metadata', {}).get('url', '')}\n"
        for i, d in enumerate(documents)
    ])

    system_message = SYSTEM_PROMPT.format(
        input=last_msg.content,
        documents=docs_text
    )

    if summarization:
        system_message = f"이전 대화 요약:\n{summarization}\n\n" + system_message

    response = model.invoke([SystemMessage(content=system_message)])
    state.get("messages").append(response)
    return {"messages": state.get("messages")}


def summarization_node(state: CustomState):
    logger.info(">>> [NODE] summarization_node START")
    messages = state.get("messages")
    summary_prompt = "대화를 요약하세요:\n" + "\n".join([msg.content for msg in messages])
    response = model.invoke([SystemMessage(content=summary_prompt)])

    delete_msgs = [RemoveMessage(id=msg.id) for msg in messages[:-8]]
    state["summarization"] = str(response.content).strip()
    logger.info("Conversation summarized.")
    return {"summarization": state["summarization"], "messages": delete_msgs}
