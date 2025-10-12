from typing import Literal
import tiktoken

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)


def detect_language(text: str, threshold: float = 0.6) -> Literal["ko", "en"]:
    """
    주어진 텍스트의 주요 언어를 토큰 분석을 통해 감지합니다.
    이 함수는 입력 텍스트를 분석하여 토큰의 60% 이상을 차지하는 언어를 반환합니다.
    한국어와 영어 텍스트를 구분합니다.

    Args:
        text (str): 언어 감지를 위해 분석할 입력 텍스트

    Returns:
        Literal["ko", "en"]: 토큰의 80% 이상이 한국어인 경우 "ko" 반환,
                            그 외에는 "en" 반환

    """
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
