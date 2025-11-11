import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.services.chat import generate_answer

@pytest.mark.asyncio
@patch("app.services.chat.get_rag_chain")  # 동기 함수 patch
async def test_generate_answer(mock_get_rag_chain):
    # AsyncMock chain 생성
    mock_chain = AsyncMock()
    
    # ainvoke 반환값에 .content 속성 추가
    mock_response = AsyncMock()
    mock_response.content = "Hi!"
    mock_chain.ainvoke.return_value = mock_response

    # get_rag_chain patch
    mock_get_rag_chain.return_value = mock_chain

    # generate_answer 호출
    result = await generate_answer("Hello")

    assert result.content == "Hi!"
    mock_chain.ainvoke.assert_called_once_with({"question": "Hello", "context": None})
