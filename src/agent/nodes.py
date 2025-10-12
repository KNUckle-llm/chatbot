from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, RemoveMessage

from .state import CustomState
from ..core.config import settings
from .utils import (
    detect_language,
)


def _configure_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings["llm"]["model"],
        api_key=settings["openai_api_key"],
        temperature=settings["llm"]["temperature"],
        max_retries=settings["llm"]["retry"]
    )


def detect_language_node(state: CustomState):
    message = str(state.get("messages")[-1].content)
    return {
        "language": detect_language(message)
    }


def answer_node(state: CustomState):
    llm = _configure_llm()
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            "다음은 지금까지 대화의 요약입니다:\n"
            f"{summary}\n"
            "이 요약을 참고하여 답변해 주세요."
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # 대화 요약 추가
    messages = state["messages"] + [SystemMessage(content=summary_message)]
    response = llm.invoke(messages)

    # 마지막 4개 메세지만 유지
    keep = (state["messages"] + [response])[-4:]
    delete_ids = {m.id for m in state["messages"] if m not in keep}
    deletes = [RemoveMessage(id=i) for i in delete_ids]
    return {"summary": response.content, "messages": deletes + [response]}
