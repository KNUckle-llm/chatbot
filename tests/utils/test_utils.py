from src.agent.utils import detect_language


def test_korean_dominant_returns_ko():
    text = (
        "이것은 한국어 문장입니다. 비자 연장은 어떻게 하나요? "
        "제출 서류와 마감 기한을 알려주세요."
    )
    assert detect_language(text, threshold=0.6) == "ko"


def test_english_dominant_returns_en():
    text = (
        "This is a clear English paragraph. It contains multiple full "
        "sentences so that English tokens overwhelmingly dominate."
    )
    assert detect_language(text, threshold=0.6) == "en"


def test_mixed_english_dominant_returns_en():
    text = (
        "안녕하세요. This paragraph is mostly in English and includes many "
        "long English words and phrases to dominate the token count overall."
    )
    assert detect_language(text, threshold=0.6) == "en"


def test_mixed_korean_dominant_returns_ko():
    text = (
        "비자 연장 안내 부탁드립니다. 필요한 서류, 수수료, 마감일이 궁금합니다. "
        "Thanks."
    )
    assert detect_language(text, threshold=0.6) == "ko"


def test_numbers_or_symbols_defaults_to_en():
    text = "12345 !!! ??? --- ___"
    # 토큰에 한글이 없으므로 en
    assert detect_language(text, threshold=0.6) == "en"


def test_empty_string_defaults_to_en():
    assert detect_language("", threshold=0.6) == "en"
