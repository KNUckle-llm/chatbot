from typing import Any, Dict, List, Optional, Literal
from langgraph.graph import MessagesState


class CustomState(MessagesState):
    profile: Optional[Dict[str, Any]]
    summarization: Optional[str]          # 이전 대화 요약
    documents: List[Dict]                 # 마지막 질문으로 벡터DB에서 검색된 문서만 저장
    language: Literal["ko", "en"]
    question_appropriate: Optional[bool]  # 질문 적절성 판단
    question_reason: Optional[str]        # 질문 판단 이유
    predicted_department: Optional[str]   # 마지막 질문에 대해 추출된 학과/부서