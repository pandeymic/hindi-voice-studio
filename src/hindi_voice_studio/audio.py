"""Bounded decoding and light reference-audio preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".aiff", ".aif"}
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 30.0
MAX_REFERENCE_BYTES = 30 * 1024 * 1024
REFERENCE_RATE = 24000


@dataclass
class ReferenceAudio:
    samples: np.ndarray
    sample_rate: int
    duration: float
    messages: list[str]

    @property
    def description(self) -> str:
        return f"{self.duration:.1f}s · mono · {self.sample_rate} Hz. " + " ".join(self.messages)


def validate_reference_audio(path: str | None) -> Path:
    if not path:
        raise ValueError("Upload or record a reference voice sample first.")
    reference = Path(path).expanduser().resolve()
    if not reference.is_file():
        raise ValueError("The reference audio file could not be found.")
    if reference.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Use one of the supported audio formats: WAV, FLAC, MP3, M4A, OGG, WebM, or AIFF.")
    if reference.stat().st_size > MAX_REFERENCE_BYTES:
        raise ValueError("Keep the reference file below 30 MB.")
    return reference


def prepare_reference(path: str | None) -> ReferenceAudio:
    """Decode at most 31s, trim edge silence, and preserve dynamics.

    No speech enhancement or denoising: those can change the speaker's identity.
    This detects signal problems, not speech, speaker count, or permission.
    """
    import librosa

    reference = validate_reference_audio(path)
    try:
        samples, rate = librosa.load(
            reference, sr=REFERENCE_RATE, mono=True, duration=MAX_REFERENCE_SECONDS + 1,
        )
    except Exception as exc:
        raise ValueError("Audio decoding failed. Try uploading a mono WAV file.") from exc
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("The recording is empty or contains invalid audio samples.")
    duration = len(samples) / rate
    if duration < MIN_REFERENCE_SECONDS:
        raise ValueError("Use at least 3 seconds of speech; 6–12 seconds is ideal.")
    if duration > MAX_REFERENCE_SECONDS:
        raise ValueError("Trim the reference to 30 seconds or less before uploading.")
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if peak < 0.001 or rms < 0.0001:
        raise ValueError("The recording is silent or too quiet. Record a clearer sample.")
    messages = []
    if np.mean(np.abs(samples) >= 0.999) > 0.005:
        messages.append("Clipping detected; a quieter recording may sound cleaner.")
    trimmed, _ = librosa.effects.trim(samples, top_db=35)
    duration = len(trimmed) / rate
    if duration < MIN_REFERENCE_SECONDS:
        raise ValueError("After trimming silence, fewer than 3 seconds remain. Record more speech.")
    if rms < 0.01:
        messages.append("Low recording level detected.")
    if duration > 12:
        messages.append("A clean 6–12 second excerpt often works better than a longer recording.")
    # Attenuate overflow, but do not boost a quiet/noisy clip to full scale.
    trimmed = trimmed / max(1.0, peak / 0.98)
    return ReferenceAudio(trimmed.astype(np.float32), rate, duration, messages)
