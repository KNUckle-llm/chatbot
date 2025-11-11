from unittest.mock import patch, MagicMock
from app.services.chat import generate_response

@patch("app.services.chat.ChatOpenAI")
def test_generate_response(mock_chatopenai):
    # ChatOpenAI 인스턴스 mock 생성
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = "Hi!"
    mock_chatopenai.return_value = mock_instance

    result = generate_response("Hello")

    assert result == "Hi!"
    mock_instance.invoke.assert_called_once_with("Hello")