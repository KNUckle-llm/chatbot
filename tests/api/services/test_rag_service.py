from unittest.mock import patch
from app.services.rag import search_documents

@patch("app.services.rag.chroma_client.query")
def test_search_documents(mock_query):
    mock_query.return_value = {"documents": ["test_doc"]}
    results = search_documents("test")
    assert "test_doc" in results
