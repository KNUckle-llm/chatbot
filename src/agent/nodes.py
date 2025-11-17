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
    logger.info("Detecting language...")
    message = str(state.get("messages")[-1].content)
    return {
        "language": detect_language(message)
    }


def generate_query_or_response_node(state: CustomState):
    logger.info("Generating query or response...")
    response = (
        model.bind_tools([retriever_tool]).invoke(state["messages"])
    )
    
    logger.info(f"Response from model.bind_tools(): {response}")
    return {
        "messages": [response],
    }


def route_before_retrieval_node(
    state: CustomState
) -> Literal["retrieve", "rewrite_question"]:
    message = state.get("messages")[-1]
    tool_calls = (
        getattr(message, "tool_calls", [])
        or getattr(message, "additional_kwargs", {}).get("tool_calls")
    )

    documents = state.get("documents", [])

    return "retrieve" if tool_calls or documents else "rewrite_question"


def collect_documents_node(state: CustomState):
    logger.info("Collecting documents...")
    tool_texts = []
    for msg in state.get("messages"):
        role = getattr(msg, "role", None) or getattr(msg, "type", None)
        if role == "tool":
            tool_texts.append(msg.content)

    combined = "\n\n---\n\n".join([t for t in tool_texts if t])

    # 기존 documents가 있으면 이어붙이기
    prev = state.get("documents") or ""
    documents = (
        prev + ("\n\n---\n\n" if prev and combined else "") + combined
    ).strip()

    # 너무 길면 뒤에서 자르기 (최근 검색 결과를 우선 보존)
    max_chars = 12000  # 필요시 조정
    if len(documents) > max_chars:
        documents = documents[-max_chars:]
        
    # 로그 확인
    logger.info("=== Log Collected Documents ===")
    logger.info(f"Collected {len(tool_texts)} documents:")
    for i, doc in enumerate(tool_texts):
        logger.debug("Document %d full content:\n%s\n%s", i+1, doc, "-"*200)

    return {"documents": documents}

# class GradeDocuments(BaseModel):
#     """Grade documents using a binary score for relevance check."""

#     binary_score: str = Field(
#         description=(
#             "Relevance score: 'yes' if relevant, or 'no' if not relevant"
#         )
#     )


# def grade_documents_node(
#     state: CustomState
# ) -> Literal["generate", "rewrite_question"]:
#     question = state["messages"][-1].content
#     context = state.get("documents")

#     prompt = GRADE_PROMPT.format(question=question, context=context)
#     response = (
#         model.with_structured_output(GradeDocuments).invoke(
#             [{"role": "system", "content": prompt}]
#         )
#     )

#     score = response.binary_score.strip().lower()
#     return "generate" if score == "yes" else "rewrite_question"


def rewrite_question_node(state: CustomState):
    logger.info("Rewriting question for HITL...")
    language = state.get("language", "ko")
    prompt = HITL_PROMPT.format(language=language)
    response = model.invoke([{"role": "system", "content": prompt}])
    return {"messages": [response]}


def generation_node(state: CustomState):
    logger.info("Generating answer...")
    # state에서 language 가져오기
    language = state.get("language")

    # 검색된 문서 state에서 documents 가져오기
    documents = state.get("documents")

    language_message = (
        f"Answer the question in {language}."
        "If en, use English; if ko, use Korean.\n"
    )

    if documents:
        system_message = SYSTEM_PROMPT.format(
            documents=documents, input=state["messages"][-1].content
        )

    else:
        summarization = state.get("summarization")
        messages = state.get("messages")

        system_message = (
            "Answer based on conversation history and summary."
            f"Conversation Summary: {summarization}\n"
            f"messages: {messages}\n"
        )

    message = language_message + system_message
    response = model.invoke([SystemMessage(content=message)])
    return {"messages": state.get("messages") + [response]}


def summarization_node(state: CustomState):
    logger.info("Summarizing conversation...")
    summarization = state.get("summarization")

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
