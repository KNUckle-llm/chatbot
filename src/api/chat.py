import uuid
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig

from src.api.schema.chat import (
    ChatRequest,
    ChatResponse,
    Message,
    ThreadSummary,
    ThreadListResponse,
    ThreadDetailResponse
)

router = APIRouter(prefix="/chat", tags=["chat"])


# ============================================================================
# Helper Functions
# ============================================================================

def generate_thread_id() -> str:
    """Generate a new UUID-based thread_id"""
    return str(uuid.uuid4())


def get_config(thread_id: str) -> RunnableConfig:
    """Create a RunnableConfig with the given thread_id"""
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def format_messages(messages: List) -> List[Message]:
    """Convert LangGraph messages to API Message format"""
    formatted = []
    for msg in messages:
        # Determine role
        msg_type = getattr(msg, "type", None) or getattr(msg, "role", None)

        if msg_type in ("human", "user"):
            role = "user"
        elif msg_type in ("ai", "assistant"):
            role = "ai"
        else:
            # Skip system messages or other types
            continue

        # Extract content
        content = getattr(msg, "content", "")

        # Extract timestamp if available
        timestamp = None
        if hasattr(msg, "created_at"):
            created_at = getattr(msg, "created_at", None)
            if created_at:
                timestamp = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)

        formatted.append(Message(
            role=role,
            content=content,
            timestamp=timestamp
        ))

    return formatted


def get_thread_metadata(graph, thread_id: str) -> Optional[ThreadSummary]:
    """Extract metadata for a specific thread"""
    try:
        config = get_config(thread_id)
        state = graph.get_state(config)

        if not state or not state.values:
            return None

        messages = state.values.get("messages", [])
        message_count = len(messages)

        # Get last message content
        last_message = None
        if messages:
            last_msg = messages[-1]
            last_message = getattr(last_msg, "content", None)
            if last_message and len(last_message) > 100:
                last_message = last_message[:100] + "..."

        # Get timestamps
        created_at = None
        updated_at = None
        if hasattr(state, "created_at") and state.created_at:
            created_at = state.created_at.isoformat() if hasattr(state.created_at, "isoformat") else str(state.created_at)
        if hasattr(state, "updated_at") and state.updated_at:
            updated_at = state.updated_at.isoformat() if hasattr(state.updated_at, "isoformat") else str(state.updated_at)

        return ThreadSummary(
            thread_id=thread_id,
            message_count=message_count,
            last_message=last_message,
            created_at=created_at,
            updated_at=updated_at
        )
    except Exception:
        return None


def _graph_input(question: str) -> dict:
    return {
        "messages": [{"role": "user", "content": question}]
    }


async def _generate_streaming_answer(graph, question: str, config, thread_id: str):
    """
    stream_mode='values' 를 사용해서 state["messages"][-1].content를 스트리밍.
    프론트엔드가 기대하는 형식으로 응답.
    """
    import json
    from datetime import datetime

    # 1. 시작 이벤트 전송
    start_event = {
        "type": "start",
        "thread_id": thread_id,
        "timestamp": datetime.now().astimezone().isoformat()
    }
    yield f"data: {json.dumps(start_event)}\n\n".encode("utf-8")

    last_text = ""

    try:
        astream = graph.astream(
            _graph_input(question),
            config=config,
            stream_mode="values",
        )

        async for state in astream:
            # 1) state는 dict 형태의 전체 상태여야 함
            if not isinstance(state, dict):
                continue

            messages = state.get("messages") or []
            if not messages:
                continue

            last = messages[-1]

            # 2) 마지막 메시지가 assistant/ai인지 확인
            role = getattr(last, "role", None) or getattr(last, "type", None)
            if role not in ("assistant", "ai"):
                continue

            # 3) content 꺼내기
            content = getattr(last, "content", None)
            if not content:
                continue

            # 4) 직전까지 보낸 텍스트와 비교해서 새로 추가된 부분만 전송
            new_part = content[len(last_text):]
            if not new_part:
                continue

            last_text = content

            # 청크 이벤트 전송
            chunk_event = {
                "type": "chunk",
                "content": new_part,
                "timestamp": datetime.now().astimezone().isoformat()
            }
            yield f"data: {json.dumps(chunk_event)}\n\n".encode("utf-8")

    except Exception as e:
        # 에러 이벤트 전송
        error_event = {
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().astimezone().isoformat()
        }
        yield f"data: {json.dumps(error_event)}\n\n".encode("utf-8")
        return

    # 종료 이벤트 전송
    end_event = {
        "type": "end",
        "timestamp": datetime.now().astimezone().isoformat()
    }
    yield f"data: {json.dumps(end_event)}\n\n".encode("utf-8")


# Server-Sent Events (SSE)
@router.post("/stream")
async def chat_stream(request: Request, payload: ChatRequest):
    if not payload.question:
        raise HTTPException(422, detail="question 필드가 필요합니다.")

    graph = getattr(request.app.state, "graph", None)

    if graph is None:
        raise HTTPException(500, detail="그래프가 초기화되지 않았습니다.")

    # Generate or use existing thread_id
    thread_id = payload.thread_id or generate_thread_id()
    config = get_config(thread_id)

    return StreamingResponse(
        _generate_streaming_answer(graph, payload.question, config, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


async def _generate_sse_response(graph, question: str, thread_id: str, config: RunnableConfig):
    """스트리밍 채팅 API (SSE)"""
    import json
    from src.agent.prompts import USER_FOCUSED_PROMPT

    last_text = ""
    try:
        # 1. 시작 이벤트
        yield f"data: {json.dumps({'type': 'start', 'thread_id': thread_id})}\n\n"

        # 2. 스트리밍
        async for chunk in graph.astream(
            {"messages": ("user", question)},
            config,
            stream_mode="values",
        ):
            messages = chunk.get("messages")
            if not messages:
                continue

            last = messages[-1]
            if last.type != "ai":
                continue

            new_part = last.content[len(last_text):]
            if new_part:
                yield f"data: {json.dumps({'type': 'chunk', 'content': new_part})}\n\n"
                last_text = last.content
        
        # 3. 종료 이벤트
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


@router.post("/stream_sse")
async def chat_stream_sse(request: Request, payload: ChatRequest):
    """스트리밍 채팅 API (SSE 방식)"""
    if not payload.question:
        raise HTTPException(422, detail="question 필드가 필요합니다.")

    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(500, detail="그래프가 초기화되지 않았습니다.")

    thread_id = payload.thread_id or generate_thread_id()
    config = get_config(thread_id)

    return StreamingResponse(
        _generate_sse_response(graph, payload.question, thread_id, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


# ============================================================================
# Chat Endpoints
# ============================================================================

# non-streaming
@router.post("", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest):
    if not payload.question:
        raise HTTPException(422, detail="question 필드가 필요합니다.")

    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(500, detail="그래프가 초기화되지 않았습니다.")

    # Generate or use existing thread_id
    thread_id = payload.thread_id or generate_thread_id()
    config = get_config(thread_id)

    try:
        result = await graph.ainvoke(_graph_input(payload.question), config=config)
    except Exception as e:
        raise HTTPException(500, detail=f"그래프 처리 중 오류가 발생했습니다: {str(e)}")

    messages = result.get("messages", [])[-1].content

    return ChatResponse(answer=messages, thread_id=thread_id)


# ============================================================================
# Thread Management Endpoints
# ============================================================================

@router.get("/threads", response_model=ThreadListResponse)
async def get_all_threads(request: Request):
    """Get all thread IDs and their metadata"""
    checkpointer = getattr(request.app.state, "checkpointer", None)
    graph = getattr(request.app.state, "graph", None)

    if checkpointer is None or graph is None:
        raise HTTPException(500, detail="Checkpointer 또는 그래프가 초기화되지 않았습니다.")

    threads = []

    # Try to access InMemorySaver's internal storage
    if hasattr(checkpointer, "storage"):
        thread_ids = list(checkpointer.storage.keys())

        for thread_id in thread_ids:
            metadata = get_thread_metadata(graph, thread_id)
            if metadata:
                threads.append(metadata)
    else:
        # Fallback: storage attribute not available
        raise HTTPException(500, detail="Checkpointer가 thread 목록 조회를 지원하지 않습니다.")

    return ThreadListResponse(threads=threads)


@router.get("/thread/{thread_id}", response_model=ThreadDetailResponse)
async def get_thread_messages(thread_id: str, request: Request):
    """Get all messages from a specific thread"""
    graph = getattr(request.app.state, "graph", None)

    if graph is None:
        raise HTTPException(500, detail="그래프가 초기화되지 않았습니다.")

    try:
        config = get_config(thread_id)
        state = graph.get_state(config)

        if not state or not state.values:
            raise HTTPException(404, detail=f"Thread {thread_id}를 찾을 수 없습니다.")

        raw_messages = state.values.get("messages", [])
        formatted_messages = format_messages(raw_messages)

        return ThreadDetailResponse(
            thread_id=thread_id,
            messages=formatted_messages,
            summarization=state.values.get("summarization"),
            language=state.values.get("language")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Thread 조회 중 오류가 발생했습니다: {str(e)}")