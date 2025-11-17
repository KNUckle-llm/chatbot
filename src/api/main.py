"""
import logging  # 최상단에 추가

logging.basicConfig(
    level=logging.DEBUG,  # INFO → DEBUG
    format='%(asctime)s | %(levelname)s | %(filename)s: line %(lineno)d | %(message)s'
)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from langgraph.checkpoint.memory import InMemorySaver

from src.core.config import settings
from src.api import chat
from src.agent.graph import build_graph

NAME = settings["app"]["name"]
VERSION = settings["app"]["version"]


# 서버 시작 전 이벤트
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting server initialization...")
    checkpointer = InMemorySaver()

    graph = build_graph(checkpointer)

    app.state.checkpointer = checkpointer
    app.state.graph = graph
    yield
    print("👋 Shutting down server...")


app = FastAPI(
    title=NAME,
    version=VERSION,
    lifespan=lifespan
)


# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://knuckle-client.vercel.app",
    ],
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
        "server_name": NAME,
        "version": VERSION,
        "timestamp": datetime.now().astimezone().isoformat()
    }
