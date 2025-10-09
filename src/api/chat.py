from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schema.chat import ChatRequest, ChatResponse
from src.rag.chat import generate_answer, generate_streaming_answer

router = APIRouter(prefix="/chat", tags=["chat"])


# Server-Sent Events (SSE)
@router.post("/stream")
async def chat_stream(request: ChatRequest):
    if not getattr(request, "question", None):
        raise HTTPException(422, detail="question 필드가 필요합니다.")
    return StreamingResponse(
        generate_streaming_answer(request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


# non-streaming
@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response = await generate_answer(request.question)
    return ChatResponse(answer=response.content)
