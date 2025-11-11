import asyncio
from unittest.mock import patch, AsyncMock
from app.services.chat import generate_answer

@patch("app.services.chat.get_rag_chain")
def test_generate_answer(mock_get_rag_chain):
    mock_chain = AsyncMock()
    mock_response = AsyncMock()
    mock_response.content = "Hi!"
    mock_chain.ainvoke.return_value = mock_response
    mock_get_rag_chain.return_value = mock_chain

    result = asyncio.run(generate_answer("Hello"))
    assert result.content == "Hi!"
