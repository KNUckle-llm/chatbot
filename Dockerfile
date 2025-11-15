FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV WORKSPACE_ROOT=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

# 작업 디렉터리 설정
WORKDIR ${WORKSPACE_ROOT}

# pyproject.toml, uv.lock 복사
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# 소스 코드 및 설정 파일 복사
COPY src ./src
COPY configs ./configs
COPY .env ./.env

# 컨테이너 시작 시 애플리케이션 실행
CMD ["/app/.venv/bin/fastapi", "run", "src/api/main.py", "--port", "80", "--host", "0.0.0.0"]