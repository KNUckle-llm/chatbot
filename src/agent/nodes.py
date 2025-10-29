from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from src.agent.state import CustomState
from src.agent.utils import (
    initialize_components,
    detect_language,
)

model, retriever = initialize_components()


def language_detection_node(state: CustomState):
    message = str(state.get("messages")[-1].content)
    return {
        "language": detect_language(message)
    }


def retrieval_node(state: CustomState):
    message = str(state.get("messages")[-1].content)
    results = retriever.invoke(message)
    return {
        "documents": [document.page_content for document in results]
    }


def generation_node(state: CustomState):
    # state에서 language 가져오기
    language = state.get("language")

    # 검색된 문서 state에서 documents 가져오기
    documents = state.get("documents")

    language_message = (
        f"Answer the question in {language}."
        "If en, use English; if ko, use Korean.\n"
    )

    if documents:
        system_message = (
            "You are an AI assistant that provides accurate answers.\n"
            f"Answer the {state.get('messages')[-1].content}\n"
            "Use the following documents to provide the answer:\n"
            f"{documents}\n"
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

    # Delete all but the 4 most recent messages
    delete_messages = [
        RemoveMessage(id=msg.id) for msg in state.get("messages")[:-4]
    ]
    return {
        "summarization": str(response.content).strip(),
        "messages": delete_messages
    }
