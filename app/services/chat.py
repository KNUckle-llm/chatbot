from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.services.prompt import PROMPT
from app.services.rag import rag

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        max_retries=3
    )

def get_rag_chain():
    return PROMPT | get_llm()

async def generate_answer(question: str):
    message = rag(question)
    rag_chain = get_rag_chain()
    return await rag_chain.ainvoke(message)

async def generate_streaming_answer(question: str):
    message = rag(question)
    rag_chain = get_rag_chain()
    try:
        async for chunk in rag_chain.astream(message):
            yield f"data: {chunk}\n\n"
    except Exception as e:
        err = str(e).replace("\n", " ")
        yield f"data: [ERROR] {err}\n\n"
