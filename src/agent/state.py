from typing import Any, Dict, Optional
from langgraph.graph import MessagesState


class CustomState(MessagesState):
    profile: Optional[Dict[str, Any]]
    summary: str
