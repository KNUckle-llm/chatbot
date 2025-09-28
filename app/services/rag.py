from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.core.config import settings

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma(
    persist_directory=settings.VECTORSTORE_DIR,
    embedding_function=hf_embeddings,
)
retriever = store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})


def rag(question: str):
    context = retriever.invoke(question)
    return {"question": question, "context": context}
