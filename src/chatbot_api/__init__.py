from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import asyncio
from typing import AsyncGenerator, Dict, Any
import logging
from datetime import datetime
import chatbot_api.routers.chat_history as chat

app = FastAPI(title="KNU Streaming Chatbot API")

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(chat.router, prefix="/chat", tags=["chat"])

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 요청/응답 모델
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    stream: bool = True
    prompt_variant: str = "user_focused"


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    response_time: float
    sources: list = []


# RAG 파이프라인 import
try:
    from .rag_pipeline import rebuild_chain, answer_question
    # from .rag_pipeline import streaming_chain, retriever  # 스트리밍용 체인도 import
except ImportError:
    logger.warning("RAG 파이프라인을 찾을 수 없습니다. 더미 함수를 사용합니다.")


    # 더미 함수들 (실제 구현 전 테스트용)
    def rebuild_chain():
        logger.info("더미 RAG 체인 초기화")
        return True


    def answer_question(question: str, session_id: str) -> str:
        return f"더미 응답: {question}에 대한 답변입니다."


    async def streaming_answer_question(question: str, session_id: str) -> AsyncGenerator[str, None]:
        """더미 스트리밍 응답 생성기"""
        response_parts = [
            "안녕하세요! ",
            "공주대학교 ",
            "정보 안내 ",
            "AI입니다. ",
            f"'{question}'에 ",
            "대한 답변을 ",
            "찾아드리겠습니다."
        ]

        for part in response_parts:
            yield part
            await asyncio.sleep(0.1)  # 0.1초 지연으로 타이핑 효과

# FastAPI 서버 시작 시 체인 구성
try:
    rag_chain = rebuild_chain()
    logger.info("✅ RAG 체인 초기화 완료")
except Exception as e:
    logger.error(f"❌ RAG 체인 초기화 실패: {e}")
    rag_chain = None


# 스트리밍 응답 생성기
async def generate_streaming_response(
        question: str,
        session_id: str,
        prompt_variant: str = "user_focused"
) -> AsyncGenerator[str, None]:
    """스트리밍 응답 생성기"""
    try:
        # 응답 시작 신호
        yield f"data: {json.dumps({'type': 'start', 'timestamp': datetime.now().isoformat()})}\n\n"

        # RAG 체인이 스트리밍을 지원하는 경우
        if hasattr(streaming_chain, 'astream'):
            full_response = ""
            async for chunk in streaming_chain.astream({"input": question}):
                if chunk:
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    await asyncio.sleep(0.01)  # 자연스러운 타이핑 효과

            # 완료 신호
            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"

        else:
            # 기존 방식을 스트리밍으로 시뮬레이션
            logger.info("기존 방식을 스트리밍으로 변환 중...")

            # 전체 응답 생성
            full_response = answer_question(question, session_id)

            # 응답을 청크 단위로 분할하여 스트리밍
            words = full_response.split()
            current_chunk = ""

            for i, word in enumerate(words):
                current_chunk += word + " "

                # 5-10 단어마다 청크 전송
                if (i + 1) % 7 == 0 or i == len(words) - 1:
                    yield f"data: {json.dumps({'type': 'token', 'content': current_chunk})}\n\n"
                    current_chunk = ""
                    await asyncio.sleep(0.15)  # 읽기 좋은 속도로 조절

            # 완료 신호
            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"

    except Exception as e:
        logger.error(f"스트리밍 응답 생성 오류: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# 일반 채팅 API (비스트리밍)
@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """일반 채팅 API (비스트리밍)"""
    start_time = datetime.now()

    try:
        if not rag_chain:
            raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다")

        logger.info(f"질문 수신: {chat_request.question[:50]}...")


        answer = answer_question(chat_request.question, chat_request.session_id)

        response_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"응답 생성 완료 ({response_time:.2f}초)")

        return ChatResponse(
            answer=answer,
            session_id=chat_request.session_id,
            response_time=response_time,
            sources=[]  # TODO: 출처 정보 추가
        )

    except Exception as e:
        logger.error(f"채팅 응답 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 스트리밍 채팅 API
@app.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest):
    """스트리밍 채팅 API (Server-Sent Events)"""
    try:
        if not rag_chain:
            raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다")

        logger.info(f"스트리밍 질문 수신: {chat_request.question[:50]}...")

        return StreamingResponse(
            generate_streaming_response(
                chat_request.question,
                chat_request.session_id,
                chat_request.prompt_variant
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "Access-Control-Allow-Origin": "*",
            }
        )

    except Exception as e:
        logger.error(f"스트리밍 채팅 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 기존 호환성을 위한 엔드포인트
@app.post("/chat/legacy")
async def chat_legacy(request: Request):
    """기존 코드와의 호환성을 위한 레거시 엔드포인트"""
    try:
        body = await request.json()
        question = body.get("question")
        session_id = body.get("session_id", "default")

        if not question:
            raise HTTPException(status_code=400, detail="질문이 필요합니다")

        answer = answer_question(question, session_id)
        return {"answer": answer}

    except Exception as e:
        logger.error(f"레거시 채팅 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "rag_chain_status": "initialized" if rag_chain else "not_initialized",
        "timestamp": datetime.now().isoformat()
    }


# WebSocket 지원 (선택사항)
try:
    from fastapi import WebSocket, WebSocketDisconnect


    @app.websocket("/chat/ws")
    async def websocket_chat(websocket: WebSocket):
        """WebSocket을 통한 실시간 채팅"""
        await websocket.accept()
        logger.info("WebSocket 연결 설정됨")

        try:
            while True:
                # 클라이언트로부터 메시지 수신
                data = await websocket.receive_json()
                question = data.get("question", "")
                session_id = data.get("session_id", "default")
                prompt_variant = data.get("prompt_variant", "user_focused")

                if not question:
                    await websocket.send_json({"type": "error", "message": "질문이 필요합니다"})
                    continue

                # 응답 시작 신호
                await websocket.send_json({
                    "type": "start",
                    "timestamp": datetime.now().isoformat()
                })

                # 스트리밍 응답 생성 및 전송
                async for chunk in generate_streaming_response(question, session_id, prompt_variant):
                    # SSE 형식을 JSON으로 변환
                    if chunk.startswith("data: "):
                        json_data = chunk[6:].strip()
                        if json_data:
                            try:
                                parsed_data = json.loads(json_data)
                                await websocket.send_json(parsed_data)
                            except json.JSONDecodeError:
                                continue

        except WebSocketDisconnect:
            logger.info("WebSocket 연결 종료됨")
        except Exception as e:
            logger.error(f"WebSocket 오류: {e}")
            await websocket.close()

except ImportError:
    logger.warning("WebSocket 지원이 불가능합니다")


def main() -> None:
    """서버 시작 함수"""
    print("🚀 KNU Streaming Chatbot API 시작 중...")
    print("📡 스트리밍 엔드포인트: POST /chat/stream")
    print("💬 일반 채팅 엔드포인트: POST /chat")
    print("🔗 WebSocket 엔드포인트: /chat/ws")
    print("📊 헬스 체크: GET /health")

    uvicorn.run(
        "chatbot_api.__init__:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()