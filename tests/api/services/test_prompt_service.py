from app.services.prompt import build_prompt

def test_build_prompt():
    prompt = build_prompt("공주대학교")
    assert "공주대학교" in prompt
