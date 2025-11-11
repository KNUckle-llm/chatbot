from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@patch("app.services.chat.get_rag_chain", new_callable=AsyncMock)
def test_chat_endpoint_valid(mock_get_rag_chain):
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Hello"
    mock_get_rag_chain.return_value = mock_chain

    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    assert response.json() == {"reply": "Hello"}


def test_chat_endpoint_invalid():
    response = client.post("/chat", json={})
    assert response.status_code == 422