import pytest
from unittest.mock import patch, AsyncMock
from app.services.chat import generate_answer

@pytest.mark.asyncio
@patch("app.services.chat.get_rag_chain", new_callable=AsyncMock)
async def test_generate_answer(mock_get_rag_chain):
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "Hi!"
    mock_get_rag_chain.return_value = mock_chain
    
    result = await generate_answer("Hello")
    
    assert result == "Hi!"
    mock_chain.ainvoke.assert_called_once_with({"question": "Hello", "context": None})
