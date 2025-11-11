from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from app.main import app

client = TestClient(app)

@patch("app.services.chat.get_rag_chain")  # 동기 함수 patch
def test_chat_endpoint_valid(mock_get_rag_chain):
    # AsyncMock chain 생성
    mock_chain = AsyncMock()
    
    # ainvoke 호출 시 반환값에 .content 속성 추가
    mock_response = AsyncMock()
    mock_response.content = "Hello"
    mock_chain.ainvoke.return_value = mock_response
    
    # patch된 get_rag_chain()이 위 chain 반환
    mock_get_rag_chain.return_value = mock_chain

    # request body 키 'question' 사용
    response = client.post("/chat", json={"question": "Hello"})
    
    assert response.status_code == 200
    assert response.json() == {"answer": "Hello"}

def test_chat_endpoint_invalid():
    response = client.post("/chat", json={})
    assert response.status_code == 422
