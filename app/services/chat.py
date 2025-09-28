from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.prompt import PROMPT
from app.services.rag import rag


llm = ChatOpenAI(model="gpt-4o-mini",
                 api_key=settings.OPENAI_API_KEY,
                 temperature=0,
                 max_retries=3)

rag_chain = PROMPT | llm


async def generate_answer(question: str):
    message = rag(question)
    print(message)
    return await rag_chain.ainvoke(message)


async def generate_streaming_answer(question: str):
    message = rag(question)
    try:
        async for chunk in rag_chain.astream(message):
            yield f"data: {chunk}\n\n"
    except Exception as e:
        err = str(e).replace("\n", " ")
        yield f"data: [ERROR] {err}\n\n"
