from pathlib import Path
from types import SimpleNamespace

import pytest

from hindi_voice_studio import app


class StubEngine:
    device = "cpu"

    def __init__(self):
        self.generated_text = None

    def generate(self, text, language_id, reference_path, options, progress):
        self.generated_text = text
        self.language_id = language_id
        return SimpleNamespace(chunks=1, sample_rate=24000, seed=123)

    def model_id(self, language_id):
        return "test-model"

    def revision_for(self, language_id):
        return "test-revision"


@pytest.fixture
def stub_engine(monkeypatch):
    engine = StubEngine()
    monkeypatch.setattr(app, "get_engine", lambda: engine)
    monkeypatch.setattr(
        app,
        "save_generation",
        lambda **kwargs: (Path("output.wav"), Path("output.json")),
    )
    return engine


def generate(engine, text, language="Hinglish", prepared_text="", **kwargs):
    app.generate_audio(
        text,
        language,
        "reference.wav",
        True,
        prepared_text,
        0.5,
        0.5,
        0.8,
        0,
        **kwargs,
    )
    return engine.generated_text


def test_ui_adapter_forwards_manual_override(stub_engine):
    manual_text = "यह exact manual text है"
    app._generate_audio_from_ui(
        "Aaj ek din hai",
        "Hinglish",
        "reference.wav",
        True,
        manual_text,
        0.5,
        0.5,
        0.8,
        0,
        "",
        True,
    )
    assert stub_engine.generated_text == manual_text


def test_generate_prepares_current_text_when_preview_is_missing(stub_engine):
    assert generate(stub_engine, "Aaj ek achha din hai", prepared_text="") == "आज एक अच्छा din है"


def test_generate_does_not_use_stale_preview_by_default(stub_engine):
    assert (
        generate(
            stub_engine,
            "Aaj ek nayi duniya hai",
            prepared_text="पुराना preview text",
        )
        == "आज एक नई दुनिया है"
    )


def test_manual_override_preserves_exact_speech_text(stub_engine):
    manual_text = "Aaj kaafi अलग pronunciation hai"
    assert (
        generate(
            stub_engine,
            "Aaj ek naya din hai",
            prepared_text=manual_text,
            manual_override=True,
        )
        == manual_text
    )


def test_manual_override_rejects_blank_text(stub_engine):
    with pytest.raises(ValueError, match="(?i)manual pronunciation override.*blank"):
        generate(stub_engine, "Aaj ek din hai", prepared_text=" \n\t ", manual_override=True)


def test_english_generation_uses_current_text_unchanged(stub_engine):
    text = "This English sentence stays exactly as written."
    assert (
        generate(
            stub_engine,
            text,
            language="English",
            prepared_text="stale preview",
            replacements="This=यह",
        )
        == text
    )


def test_current_hinglish_rules_are_applied_on_generate(stub_engine):
    assert (
        generate(
            stub_engine,
            "Aaj main ready hoon",
            replacements="main=main\nready=रेडी",
        )
        == "आज main रेडी हूँ"
    )
