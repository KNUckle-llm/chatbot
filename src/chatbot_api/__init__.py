from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

from .rag_pipeline import rebuild_chain, answer_question

# FastAPI 서버 시작 시 체인 구성
rag_chain = rebuild_chain()

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    question = body.get("question")
    session_id = body.get("session_id")
    answer = answer_question(question, session_id)
    return {"answer": answer}

def main() -> None:
    print("Starting chatbot API...")
    uvicorn.run("chatbot_api.__init__:app", host="0.0.0.0", port=8000, reload=True)
