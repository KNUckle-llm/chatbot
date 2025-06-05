### ✅ run.sh (macOS / Linux)
#!/bin/bash

set -e

echo "[1/5] Starting project setup..."

# 1. Check and install uv if not found
if ! command -v uv &> /dev/null; then
    echo "[INFO] uv not found. Installing uv using curl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Re-check uv
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv installation failed or not found in PATH."
    exit 1
fi

# 2. Run MySQL Docker container
if [ -f "Dockerfile" ]; then
  echo "[2/5] Building and running MySQL Docker container..."
  docker build -t chatbot-mysql .
  docker run -d --name chatbot-mysql -p 3306:3306 chatbot-mysql
else
  echo "[WARNING] Dockerfile not found. MySQL container will not be started."
fi

# 3. Create virtual environment if not exist
if [ ! -d ".venv" ]; then
  echo "[3/5] Creating virtual environment..."
  uv venv --python 3.13
fi

# 4. Install dependencies
echo "[4/5] Installing project dependencies from requirements.txt..."
source .venv/bin/activate
uv add -r requirements.txt

# 5. Run FastAPI server
echo "[5/5] Starting FastAPI server at http://localhost:8000 ..."
uv run chatbot-api