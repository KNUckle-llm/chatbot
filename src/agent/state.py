from typing import Any, Dict, List, Optional, Literal
from langgraph.graph import MessagesState


class CustomState(MessagesState):
    profile: Optional[Dict[str, Any]]
    summarization: str | None
    documents: List[Dict]
    language: Literal["ko", "en"]