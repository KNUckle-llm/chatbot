from typing import Literal
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)

# 1) metadata 포함 retriever tool (BaseTool 미사용)
class RetrieverWithMetadataTool:
    name = "retrieve_kongju_national_university_info"
    description = "Search vector DB and return content + metadata"

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query: str):
        docs = self.retriever.invoke(query)
        results = []
        for d in docs:
            results.append({
                "content": d.page_content,
                "metadata": d.metadata  # metadata 유지
            })
        return results

    async def arun(self, query: str):
        return self.run(query)


def initialize_components():
    # LLM 로드
    model = ChatOpenAI(
        model=settings["llm"]["model"],
        api_key=settings["openai_api_key"],
        temperature=settings["llm"]["temperature"],
        max_retries=settings["llm"]["retry"]
    )

    # 임베딩 로드
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=settings["embedding"]["model"]
    )

    # 벡터 DB
    store = Chroma(
        persist_directory="./chatbot_db",
        embedding_function=hf_embeddings,
    )

    # Retriever 설정
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    # 커스텀 툴 사용
    retriever_tool = RetrieverWithMetadataTool(retriever)

    return model, store, retriever_tool


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
