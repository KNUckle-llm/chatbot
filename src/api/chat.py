from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig

from src.api.schema.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])
config: RunnableConfig = {
    "configurable": {
        "thread_id": 123
    }
}


def _graph_input(question: str) -> dict:
    return {
        "messages": [{"role": "user", "content": question}]
    }


async def _generate_streaming_answer(graph, question: str, config):
    """
    stream_mode='values' 를 사용해서 state["messages"][-1].content를 스트리밍.
    """
    last_text = ""

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
        yield f"data: {new_part}\n\n".encode("utf-8")

    # 스트림 종료 신호 (선택)
    yield b"event: end\ndata: [DONE]\n\n"


# Server-Sent Events (SSE)
@router.post("/stream")
async def chat_stream(request: Request, payload: ChatRequest):
    if not payload.question:
        raise HTTPException(422, detail="question 필드가 필요합니다.")

    graph = getattr(request.app.state, "graph", None)

    if graph is None:
        raise HTTPException(500, detail="그래프가 초기화되지 않았습니다.")

    return StreamingResponse(
        _generate_streaming_answer(graph, payload.question, config),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


# non-streaming
@router.post("", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest):
    if not payload.question:
        raise HTTPException(422, detail="question 필드가 필요합니다.")

    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(500, detail="그래프가 초기화되지 않았습니다.")

    try:
        result = await graph.ainvoke(_graph_input(payload.question), config=config)
    except Exception as e:
        raise HTTPException(500, detail=f"그래프 처리 중 오류가 발생했습니다: {str(e)}")

    messages = result.get("messages", [])[-1].content

    return ChatResponse(answer=messages)
