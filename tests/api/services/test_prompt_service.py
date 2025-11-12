import pytest
from langchain_core.prompts import ChatPromptTemplate
from app.services.prompt import PROMPT


def test_prompt_structure():
    # ✅ PROMPT가 ChatPromptTemplate 인스턴스인지 확인
    assert isinstance(PROMPT, ChatPromptTemplate)

    # ✅ system 메시지가 올바르게 정의되어 있는지 확인
    system_msg_template = PROMPT.messages[0]
    assert hasattr(system_msg_template, "prompt"), "system 메시지에 prompt 속성이 없습니다."
    assert hasattr(system_msg_template.prompt, "template"), "system 메시지 prompt에 template 속성이 없습니다."

    # ✅ 템플릿 내용 확인
    template_content = system_msg_template.prompt.template
    assert isinstance(template_content, str), "system 메시지 template 내용이 문자열이 아닙니다."
    assert len(template_content) > 0, "system 메시지 template이 비어 있습니다."

    # ✅ human 메시지도 존재하는지 간단히 체크
    assert any("HumanMessagePromptTemplate" in str(type(m)) for m in PROMPT.messages), \
        "human 메시지 템플릿이 포함되어 있지 않습니다."
