from typing import Literal
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.tools.retriever import create_retriever_tool

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)


def initialize_components():
    model = ChatOpenAI(
        model=settings["llm"]["model"],
        api_key=settings["openai_api_key"],
        temperature=settings["llm"]["temperature"],
        max_retries=settings["llm"]["retry"]
    )

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=settings["embedding"]["model"]
    )

    store = Chroma(
        persist_directory="/app/src/agent/chatbot_db",
        embedding_function=hf_embeddings,
    )

    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k":10, "lambda_mult":0.5}
    )

    retriever_tool = create_retriever_tool(
        name="retrieve_kongju_national_university_info",
        description="Search and return information about 국립공주대학교",
        retriever=retriever,
    )

    return model, store, retriever_tool


def detect_language(text: str, threshold: float = 0.6) -> Literal["ko", "en"]:
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

    return "ko" if total_len > 0 and korean_count / total_len >= threshold else "en"
