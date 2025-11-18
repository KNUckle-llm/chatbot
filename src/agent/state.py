from typing import Any, Dict, List, Optional, Literal, TypedDict
from langgraph.graph import MessagesState

class Document(TypedDict):
    content: str
    metadata: dict

class CustomState(MessagesState):
    profile: Optional[Dict[str, Any]] = None
    summarization: Optional[str] = None
    documents: List[Document] = []
    language: Literal["ko", "en"] = "ko"
