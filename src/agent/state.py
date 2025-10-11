from typing import Any, Dict, Optional, Literal
from langgraph.graph import MessagesState


class CustomState(MessagesState):
    profile: Optional[Dict[str, Any]]
    summary: str
    language: Literal["ko", "en"]
