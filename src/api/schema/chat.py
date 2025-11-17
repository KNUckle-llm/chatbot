from typing import List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None
    stream: bool = True
    prompt_variant: str = "user_focused"


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


class Message(BaseModel):
    role: str  # "user" or "ai"
    content: str
    timestamp: Optional[str] = None


class ThreadSummary(BaseModel):
    thread_id: str
    message_count: int
    last_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ThreadListResponse(BaseModel):
    threads: List[ThreadSummary]


class ThreadDetailResponse(BaseModel):
    thread_id: str
    messages: List[Message]
    summarization: Optional[str] = None
    language: Optional[str] = None
