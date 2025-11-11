from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_endpoint_valid():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200

def test_chat_endpoint_invalid():
    response = client.post("/chat", json={})
    assert response.status_code == 422
