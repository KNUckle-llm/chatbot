FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

# 로그를 바로 출력하도록 설정
ENV PYTHONUNBUFFERED=1

# ─── 빌드 도구 설치 (g++, make 등) ───────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 설정
WORKDIR /app

# uv.lock 파일 복사
COPY pyproject.toml uv.lock README.md ./
RUN mkdir -p ./src/chatbot_api
RUN touch ./src/chatbot_api/__init__.py

# lockfile 기반으로 의존성 동기화
RUN uv sync --locked

# 프로젝트 소스 전체 복사 (uv.lock 포함)
ADD . .

# FastAPI/uvicorn이 외부에서 접근 가능하도록 포트 노출
EXPOSE 8000

# 컨테이너 시작 시 애플리케이션 실행
CMD ["uv", "run", "chatbot-api"]