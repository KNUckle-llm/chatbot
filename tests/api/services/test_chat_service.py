from unittest.mock import patch, AsyncMock
from app.services.chat import generate_response

@patch("app.services.chat.llm", new_callable=AsyncMock)  # <- 여기 변경
def test_generate_response(mock_llm):
    mock_llm.ainvoke.return_value = "Hi!"  # generate_response에서 await llm.ainvoke(...)
    
    result = generate_response("Hello")  # 비동기면 await 필요: await generate_response("Hello")
    
    assert result == "Hi!"
    mock_llm.ainvoke.assert_called_once_with("Hello")
