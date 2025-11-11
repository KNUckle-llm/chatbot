import pytest
from unittest.mock import patch, AsyncMock
from app.services.chat import generate_answer

@pytest.mark.asyncio
@patch("app.services.chat.rag_chain", new_callable=AsyncMock)
async def test_generate_answer(mock_chain):
    mock_chain.ainvoke.return_value = "Hi!"
    
    result = await generate_answer("Hello")
    
    assert result == "Hi!"
    mock_chain.ainvoke.assert_called_once_with("Hello")