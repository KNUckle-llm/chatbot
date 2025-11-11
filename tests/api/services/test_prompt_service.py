from app.services.prompt import PROMPT

def test_prompt_structure():
    # PROMPT가 ChatPromptTemplate 객체인지 확인
    from langchain_core.prompts import ChatPromptTemplate
    assert isinstance(PROMPT, ChatPromptTemplate)
    
    # system 메시지가 포함되어 있는지 간단히 체크
    system_msg = PROMPT.messages[0].content
    assert "You are the official information assistant" in system_msg
