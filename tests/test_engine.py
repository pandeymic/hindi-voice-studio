from hindi_voice_studio.engine import VoiceEngine


def test_hindi_uses_the_multilingual_model_by_default():
    engine = VoiceEngine(device="cpu", model_version="v3")
    assert engine.hindi_model == "multilingual"
    assert engine.model_id("hi") == "ResembleAI/chatterbox:v3"
    assert engine.revision_for("hi") == "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"


def test_dedicated_hindi_model_is_explicit():
    engine = VoiceEngine(device="cpu", model_version="v3", hindi_model="dedicated")
    assert engine.model_id("hi") == "ResembleAI/Chatterbox-Multilingual-hi"
    assert engine.revision_for("hi") == "82ca71273cc2a9ab19efdf8315f865c1a5af0ee7"
    assert engine.model_id("en").endswith(":v3")


def test_multilingual_fallback_is_explicit():
    engine = VoiceEngine(device="cpu", model_version="v3", hindi_model="multilingual")
    assert engine.model_id("hi") == "ResembleAI/chatterbox:v3"
