from unittest.mock import patch, MagicMock
from app.services import rag as rag_module

@patch("app.services.rag.retriever")
def test_rag(mock_retriever):
    # invoke 메서드 모킹
    mock_retriever.invoke = MagicMock(return_value=["test_doc"])

    results = rag_module.rag("test")

    # 반환값 검증
    assert results["context"] == ["test_doc"]
    mock_retriever.invoke.assert_called_once_with("test")
