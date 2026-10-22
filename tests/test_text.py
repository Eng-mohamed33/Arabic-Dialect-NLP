from src.data.text import normalize_arabic


def test_normalize_arabic_removes_noise_and_unifies_alif():
    assert normalize_arabic("  أإآ ــ حلووو https://example.com @user ") == "اا حلوو"


def test_normalize_arabic_rejects_non_string():
    try:
        normalize_arabic(None)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")

