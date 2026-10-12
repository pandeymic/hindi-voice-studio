import pytest

from hindi_voice_studio.text import (
    chunk_text,
    language_code,
    normalize_text,
    pronunciation_preview,
)


def test_language_codes_keep_hinglish_on_hindi_model():
    assert language_code("Hindi") == "hi"
    assert language_code("Hinglish") == "hi"
    assert language_code("English") == "en"


def test_normalize_text_preserves_devanagari_and_collapses_spaces():
    assert normalize_text("  नमस्ते…   duniya  ") == "नमस्ते... duniya"


def test_chunk_text_prefers_sentence_boundaries():
    text = "पहला वाक्य बहुत महत्वपूर्ण है और इसमें पर्याप्त शब्द हैं। दूसरा वाक्य भी उतना ही महत्वपूर्ण और स्पष्ट है।"
    chunks = chunk_text(text, limit=60)
    assert len(chunks) == 2
    assert chunks[0].endswith("हैं।")
    assert chunks[1].endswith("है।")


def test_chunk_text_breaks_long_input_without_dropping_words():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, limit=80)
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert "word0" in chunks[0]
    assert "word99" in chunks[-1]


def test_hinglish_preview_is_editable_and_supports_overrides():
    assert pronunciation_preview("Aaj kaafi achha hai", "Hinglish") == "आज काफ़ी अच्छा है"
    assert pronunciation_preview("Aaj main ready hoon", "Hinglish", "main=main") == "आज main ready हूँ"


def test_preview_rejects_invalid_replacement_rule():
    with pytest.raises(ValueError, match="word=pronunciation"):
        pronunciation_preview("hello", "Hinglish", "not a rule")
