from typing import Literal
from pydantic import BaseModel, Field

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
    message = str(state.get("messages")[-1].content)
    return {
        "language": detect_language(message)
    }


def generate_query_or_response_node(state: CustomState):
    retriever_response = retriever_tool.invoke({
        "query": state["messages"][-1].content
    })
    retrieved_docs = retriever_response.get("documents", [])
    
    response = (
        model.bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {
        "messages": state["messages"] + [response],
        "documents": retrieved_docs
    }


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description=(
            "Relevance score: 'yes' if relevant, or 'no' if not relevant"
        )
    )


def grade_documents_node(
    state: CustomState
) -> Literal["generate", "rewrite_question"]:
    question = state["messages"][-1].content
    context = state.get("documents")

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        model.with_structured_output(GradeDocuments).invoke(
            [{"role": "system", "content": prompt}]
        )
    )

    score = response.binary_score.strip().lower()
    return "generate" if score == "yes" else "rewrite_question"


def rewrite_question_node(state: CustomState):
    language = state.get("language", "ko")
    prompt = HITL_PROMPT.format(language=language)
    response = model.invoke([{"role": "system", "content": prompt}])
    return {"messages": state["messages"] + [response]}


def generation_node(state: CustomState):
    # state에서 language 가져오기
    language = state.get("language")

    # 검색된 문서 state에서 documents 가져오기
    documents = state.get("documents")
    if not isinstance(documents, list):
        documents = [documents] if documents else []

    language_message = (
        f"Answer the question in {language}."
        "If en, use English; if ko, use Korean.\n"
    )

    if documents:
        # documents가 dict list이면 문자열로 변환
        documents_text = "\n\n".join(
            f"Content: {doc.get('content', '')}\nSource: {doc.get('metadata', {}).get('source', '')}"
            for doc in documents
        )

        system_message = SYSTEM_PROMPT.format(
            documents=documents_text,
            input=state["messages"][-1].content
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