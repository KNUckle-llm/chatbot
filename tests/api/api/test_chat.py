from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@patch("app.services.chat.rag_chain", new_callable=AsyncMock)
def test_chat_endpoint_valid(mock_create):
    mock_create.return_value = {"choices": [{"message": {"content": "Hello"}}]}
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    assert response.json() == {"reply": "Hello"}

def test_chat_endpoint_invalid():
    response = client.post("/chat", json={})
    assert response.status_code == 422