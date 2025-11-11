from unittest.mock import patch
from app.services.rag import rag

@patch("app.services.rag.retriever.invoke")
def test_rag(mock_invoke):
    mock_invoke.return_value = ["test_doc"]
    results = rag("test")
    # rag 함수는 dict 반환
    assert results["context"] == ["test_doc"]
    mock_invoke.assert_called_once_with("test")
