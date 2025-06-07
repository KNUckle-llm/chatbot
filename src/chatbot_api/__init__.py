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
    from .rag_pipeline import rebuild_chain, answer_question, streaming_answer_question, generate_streaming_response
except ImportError:
    logger.warning("RAG 파이프라인을 찾을 수 없습니다. 더미 함수를 사용합니다.")


    # 더미 함수들 (실제 구현 전 테스트용)
    def rebuild_chain():
        logger.info("더미 RAG 체인 초기화")
        return True

# FastAPI 서버 시작 시 체인 구성
try:
    rag_chain = rebuild_chain()
    logger.info("✅ RAG 체인 초기화 완료")
except Exception as e:
    logger.error(f"❌ RAG 체인 초기화 실패: {e}")
    rag_chain = None


# 🚀 새로운 스트리밍 채팅 API (SSE 방식)
@app.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest):
    """스트리밍 채팅 API (Server-Sent Events)"""
    try:
        if not rag_chain:
            raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다")

        logger.info(f"스트리밍 질문 수신: {chat_request.question[:50]}...")

        # SSE 스트리밍 응답 생성
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
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

    except Exception as e:
        logger.error(f"스트리밍 채팅 응답 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))



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


# 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "rag_chain_status": "initialized" if rag_chain else "not_initialized",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "streaming": ["/chat/stream", "/chat/stream-simple"],
            "regular": ["/chat"],
            "websocket": ["/chat/ws"]
        }
    }

def main() -> None:
    """서버 시작 함수"""
    print("🚀 KNU Streaming Chatbot API 시작 중...")
    print("📡 SSE 스트리밍 엔드포인트: POST /chat/stream")
    print("💬 일반 채팅 엔드포인트: POST /chat")
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