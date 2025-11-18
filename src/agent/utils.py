from typing import Literal
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)

def initialize_components():
    # LLM
    model = ChatOpenAI(
        model=settings["llm"]["model"],
        api_key=settings["openai_api_key"],
        temperature=settings["llm"]["temperature"],
        max_retries=settings["llm"]["retry"]
    )

    # Embeddings & Chroma
    hf_embeddings = HuggingFaceEmbeddings(model_name=settings["embedding"]["model"])
    store = Chroma(persist_directory=".src/agent/chatbot_db", embedding_function=hf_embeddings)

    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    base_tool = create_retriever_tool(
        name="retrieve_kongju_national_university_info",
        description="Search vector DB and return content + metadata",
        retriever=retriever
    )

    def retriever_tool_fn(query: str):
        logger.info(f"Retriever 호출: {query}")
        docs = base_tool.run(query)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

    retriever_tool_structured = StructuredTool.from_function(
        func=retriever_tool_fn,
        name="retrieve_kongju_national_university_info",
        description="Search vector DB and return content + metadata"
    )

    class RetrieverToolNode(ToolNode):
        def __init__(self, tool):
            super().__init__([tool])
            self.tool = tool

        def run(self, state, *args, **kwargs):
            query = None
            for msg in reversed(state.get("messages", [])):
                role = getattr(msg, "role", getattr(msg, "role", None))
                content = getattr(msg, "content", getattr(msg, "content", None))
                if role == "user" and content:
                    query = str(content)
                    break
            if not query:
                return state
            # 이전 tool 메시지 삭제
            old_tool_msgs = [msg for msg in state.get("messages", []) if getattr(msg, "role", None) == "tool"]
            for msg in old_tool_msgs:
                try:
                    state.remove_message(msg.id)
                except AttributeError:
                    state.set("messages", [m for m in state.get("messages", []) if m != msg])
            results = self.tool.run(query)
            if results:
                state.add_tool_message(content=results, tool_name=self.tool.name)
            return state

    return model, store, retriever_tool_structured, RetrieverToolNode



def detect_language(text: str, threshold: float = 0.6) -> Literal["ko", "en"]:
    """
    주어진 텍스트의 주요 언어를 토큰 분석을 통해 감지합니다.
    이 함수는 입력 텍스트를 분석하여 토큰의 threshold 이상을 차지하는 언어를 반환합니다.
    한국어와 영어 텍스트를 구분합니다.

    Args:
        text (str): 언어 감지를 위해 분석할 입력 텍스트

    Returns:
        Literal["ko", "en"]: 토큰의 threshold 이상이 한국어인 경우 "ko" 반환,
                            그 외에는 "en" 반환

    """
    if not text or not text.strip():
        return "ko"

    encoding = tiktoken.encoding_for_model(settings["llm"]["model"])
    tokens = encoding.encode(text)
    total_len = len(tokens)

    korean_count = 0
    for token in tokens:
        decoded = encoding.decode_single_token_bytes(token)
        piece = decoded.decode("utf-8", errors="ignore")

        if any("가" <= char <= "힣" for char in piece):
            korean_count += 1

    if total_len > 0 and korean_count / total_len >= threshold:
        return "ko"
    else:
        return "en"
