from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from .core.config import settings
from .api import chat


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

# 헬스체크 엔드포인트
@app.get("/health")
def health_check():
    return {"status": "ok"}

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
    ],  # Next.js 개발 서버 / 배포 도메인
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(chat.router)


@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "server_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now().astimezone().isoformat()
    }
