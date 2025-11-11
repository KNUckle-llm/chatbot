from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app

client = TestClient(app)

@patch("app.services.chat.get_rag_chain", new_callable=AsyncMock)
def test_chat_endpoint_valid(mock_get_rag_chain):
    # AsyncMock으로 반환값 만들기
    mock_chain = AsyncMock()
    
    # generate_answer에서 await rag_chain.ainvoke(...) → content 속성 필요
    mock_response = AsyncMock()
    mock_response.content = "Hello"
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)
    
    mock_get_rag_chain.return_value = mock_chain

    # request body 키 수정: 'message' → 'question'
    response = client.post("/chat", json={"question": "Hello"})
    
    assert response.status_code == 200
    assert response.json() == {"answer": "Hello"}

def test_chat_endpoint_invalid():
    # question 없이 요청 → 422 확인
    response = client.post("/chat", json={})
    assert response.status_code == 422
