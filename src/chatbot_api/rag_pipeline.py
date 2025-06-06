
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, trim_messages
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Load environment
load_dotenv()

# Global resources (pre-loaded for performance)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Prompt and parser
system_message = """당신은 공주대학교와 관련된 정보를 안내하는 AI입니다.
공식 문서나 공주대학교 사이트에서 제공되는 정보만 바탕으로 대답하세요.
문맥에서 명확한 정보가 없으면 "찾을 수 없습니다"라고 말해주세요.
정확한 출처를 아래와 같이 반드시 포함하세요.
파일명 :
부서/학과 :
URL :"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("placeholder", "{memory}"),
    ("user", "🔍 검색된 문서:\n{context}"),
    ("human", "{input}"),
])

parser = StrOutputParser()
trimmer = trim_messages(max_tokens=500, token_counter=llm, strategy="last", include_system=True, start_on="human")

# Session store
history_store = {}

def init_session(session_id: str):
    if session_id not in history_store:
        mysql_url = os.getenv("DATABASE_URL")
        history_store[session_id] = SQLChatMessageHistory(
            connection=mysql_url,
            table_name="chat_history",
            session_id=session_id,
            session_id_field_name="session_id",
        )
        history_store[session_id].add_message(SystemMessage(content=system_message.strip()))

def get_session_history(session_id: str):
    return history_store[session_id]


retriever = None
rag_chain = None

def rebuild_chain(selected_dept: str = None):
    global retriever, rag_chain
    db_path = "./ChromaDB/knu_chroma_db_all"
    db = Chroma(persist_directory=db_path, embedding_function=hf_embeddings)

    results = db.get(include=["documents", "metadatas"])
    docs = [Document(page_content=d, metadata=m) for d, m in zip(results["documents"], results["metadatas"])]

    chroma_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3
    base_retriever = EnsembleRetriever(retrievers=[chroma_retriever, bm25_retriever])
    retriever = MultiQueryRetriever.from_llm(base_retriever, llm=llm)

    core_chain = {
        "memory": trimmer,
        "context": retriever,
        "input": RunnablePassthrough()
    } | prompt_template | llm | parser

    rag_chain = RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="memory"
    )

def answer_question(question: str, session_id: str) -> str:
    if not session_id:
        return "⚠️ 세션을 먼저 생성하거나 선택해주세요."
    init_session(session_id)
    return rag_chain.invoke({
        "memory": None,
        "context": None,
        "input": question
    }, config={"configurable": {"session_id": session_id}})
