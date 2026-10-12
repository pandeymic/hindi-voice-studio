from pathlib import Path

import pytest

from hindi_voice_studio.audio import validate_reference_audio


def test_missing_reference_has_actionable_error():
    with pytest.raises(ValueError, match="Upload or record"):
        validate_reference_audio(None)


def test_unsupported_reference_extension_is_rejected(tmp_path: Path):
    sample = tmp_path / "voice.txt"
    sample.write_text("not audio")
    with pytest.raises(ValueError, match="audio formats"):
        validate_reference_audio(str(sample))
