from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain.schema import Document

from src.agent.state import CustomState
from src.agent.utils import initialize_components, detect_language
from src.agent.prompts import GRADE_PROMPT, HITL_PROMPT, SYSTEM_PROMPT
from ..core.logger import get_logger

logger = get_logger(__name__)
model, store, retriever_tool = initialize_components()


# -----------------------------
# 언어 감지
# -----------------------------
def language_detection_node(state: CustomState):
    message = str(state.get("messages")[-1].content)
    return {
        "language": detect_language(message)
    }


# -----------------------------
# 쿼리 검색 및 모델 응답
# -----------------------------
def generate_query_or_response_node(state: CustomState):
    last_message = state["messages"][-1].content

    # 벡터 DB 검색
    retriever_response = retriever_tool.invoke({"query": last_message})
    retrieved_docs: list[Document] = retriever_response.get("documents", [])

    # 모델 응답 생성
    response = model.bind_tools([retriever_tool]).invoke(state["messages"])

    logger.info(f"[GENERATE_NODE] Retrieved Docs: {len(retrieved_docs)}")
    logger.info(f"[GENERATE_NODE] Model Response Type: {type(response)}, Content: {str(response)[:200]}")
    
    return {
        "messages": state["messages"] + [response],
        "documents": retrieved_docs
    }


# -----------------------------
# 문서 평가용 Pydantic 모델
# -----------------------------
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


# -----------------------------
# 문서 평가 노드
# -----------------------------
def grade_documents_node(state: CustomState) -> Literal["generate", "rewrite_question"]:
    question = state["messages"][-1].content
    context = state.get("documents", [])

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = model.with_structured_output(GradeDocuments).invoke(
        [{"role": "system", "content": prompt}]
    )

    score = response.binary_score.strip().lower()
    next_node = "generate" if score == "yes" else "rewrite_question"

    logger.info(f"[GRADE_NODE] Score: {score}, Next Node: {next_node}")
    return next_node


# -----------------------------
# 질문 재작성 노드
# -----------------------------
def rewrite_question_node(state: CustomState):
    language = state.get("language", "ko")
    prompt = HITL_PROMPT.format(language=language)
    response = model.invoke([{"role": "system", "content": prompt}])
    return {"messages": state["messages"] + [response]}


# -----------------------------
# 최종 생성 노드
# -----------------------------
def generation_node(state: CustomState):
    language = state.get("language", "ko")
    documents: list[Document] = state.get("documents", [])
    if not isinstance(documents, list):
        documents = [documents] if documents else []

    language_message = (
        f"Answer the question in {language}. "
        "If en, use English; if ko, use Korean.\n"
    )

    if documents:
        # Document 객체 기준 안전하게 텍스트 + 메타데이터 추출
        documents_text = "\n\n".join(
            f"Content: {doc.page_content}\n"
            f"Source: {doc.metadata.get('url', '') or doc.metadata.get('file_name', '')}"
            for doc in documents
        )

        system_message = SYSTEM_PROMPT.format(
            documents=documents_text,
            input=state["messages"][-1].content
        )
    else:
        summarization = state.get("summarization", "")
        messages = state.get("messages", [])
        system_message = (
            "Answer based on conversation history and summary.\n"
            f"Conversation Summary: {summarization}\n"
            f"messages: {messages}\n"
        )

    message = language_message + system_message
    response = model.invoke([SystemMessage(content=message)])
    return {"messages": state.get("messages") + [response]}


# -----------------------------
# 요약 노드
# -----------------------------
def summarization_node(state: CustomState):
    summarization = state.get("summarization", "")
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

    # 마지막 8개 메시지만 남기고 나머지 삭제
    delete_messages = [
        RemoveMessage(id=msg.id) for msg in state.get("messages")[:-8]
    ]
    
    logger.info(f"[SUMMARIZATION_NODE] Delete Messages: {[msg.id for msg in delete_messages]}")
    logger.info(f"[SUMMARIZATION_NODE] New Summary: {response.content}")

    return {
        "summarization": str(response.content).strip(),
        "messages": delete_messages
    }
