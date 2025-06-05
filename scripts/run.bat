@echo off
setlocal

echo [1/5] Starting project setup...

:: 1. Check and install uv if not found
where uv >nul 2>nul
if errorlevel 1 (
    echo [INFO] uv not found. Installing uv using PowerShell...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

:: Re-check uv availability
where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to install uv or not found in PATH.
    exit /b 1
)

:: 2. Run MySQL Docker container
IF EXIST Dockerfile (
    echo [2/5] Building and running MySQL Docker container...
    docker build -t chatbot-mysql .
    docker run -d --name chatbot-mysql -p 3306:3306 chatbot-mysql
) ELSE (
    echo [WARNING] Dockerfile not found. MySQL container will not be started.
)

:: 3. Create virtual environment if not exist
IF NOT EXIST ".venv" (
    echo [3/5] Creating virtual environment...
    uv venv --python 3.13
)

:: 4. Install dependencies
echo [4/5] Installing project dependencies from pyproject.toml...
call .venv\Scripts\activate
uv lock
uv sync


:: 5. Run FastAPI server
echo [5/5] Starting FastAPI server at http://localhost:8000 ...
uv run chatbot-api
