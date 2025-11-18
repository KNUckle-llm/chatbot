from typing import Any, Dict, List, Optional, Literal
from langgraph.graph import MessagesState


class CustomState(MessagesState):
    profile: Optional[Dict[str, Any]]
    summarization: Optional[str]          # 이전 대화 요약
    documents: List[Dict]                 # 현재 질문에 대한 관련 문서만 저장
    language: Literal["ko", "en"]
    question_appropriate: Optional[bool]  # 질문 적절성 판단
    question_reason: Optional[str]        # 질문 부적절 시 이유
