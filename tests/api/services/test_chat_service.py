from unittest.mock import patch
from app.services.chat import generate_response

@patch("app.services.chat.openai.ChatCompletion.create")
def test_generate_response(mock_create):
    mock_create.return_value = {"choices": [{"message": {"content": "Hi!"}}]}
    result = generate_response("Hello")
    assert "Hi" in result
