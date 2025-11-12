from app.services.prompt import PROMPT

def test_prompt_structure():
    from langchain_core.prompts import ChatPromptTemplate
    from app.services.prompt import PROMPT

    # ChatPromptTemplate인지 확인
    assert isinstance(PROMPT, ChatPromptTemplate)

    # system 메시지 템플릿의 내용이 존재하는지 확인
    system_msg_template = PROMPT.messages[0]
    assert hasattr(system_msg_template, "prompt")
    assert "template" in system_msg_template.prompt.__dict__

    template_content = system_msg_template.prompt.template
    assert "역할" in template_content or "Role" in template_content  # 예시 키워드 체크
