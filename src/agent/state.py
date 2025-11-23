from typing import Any, Dict, List, Optional, Literal
from langgraph.graph import MessagesState


class CustomState(MessagesState):
    profile: Optional[Dict[str, Any]]
    summarization: Optional[str]          # 이전 대화 요약
    documents: List[Dict]                 # 마지막 질문으로 벡터DB에서 검색된 문서만 저장
    language: Literal["ko", "en"]
    question_appropriate: Optional[bool]  # 질문 적절성 판단
    question_reason: Optional[str]        # 질문 판단 이유
    current_department: Optional[str] = None   # 현재 대화 주제와 관련된 학과
    current_topic: Optional[str] = None        # 현재 대화 주제 (짧은 한 문장)
    follow_up: Optional[bool] = None           # follow-up 여부