from typing import Any, Dict, List, Optional, Literal, TypedDict
from langgraph.graph import MessagesState

class Document(TypedDict):
    content: str
    metadata: dict

class CustomState(MessagesState):
    profile: Optional[Dict[str, Any]]
    summarization: Optional[str]
    documents: List[Document]
    language: Literal["ko", "en"]
    next_node: Optional[str] = None  # 다음 노드 지정용