"""Save generated audio and enough metadata to repeat a run."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import soundfile as sf

from . import __version__
from .engine import GenerationOptions, GenerationResult


def save_generation(
    result: GenerationResult, directory: Path, text: str, language: str,
    options: GenerationOptions, model_version: str, device: str, revision: str,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC)
    stem = timestamp.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    wav_path = directory / f"{stem}.wav"
    metadata_path = directory / f"{stem}.json"
    # FLOAT WAV avoids rescaling/clipping the model's watermarked waveform.
    sf.write(wav_path, result.audio, result.sample_rate, subtype="FLOAT")
    metadata = {
        "app_version": __version__, "created_at": timestamp.isoformat(),
        "engine": "chatterbox-multilingual", "model_version": model_version,
        "model_revision": revision, "device": device, "language": language,
        "spoken_text": text, "options": {**asdict(options), "seed": result.seed},
        "sample_rate": result.sample_rate,
        "duration_seconds": len(result.audio) / result.sample_rate,
        "chunks": result.chunks,
        "watermark": "Perth, applied by upstream Chatterbox; not independently verified",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return wav_path.resolve(), metadata_path.resolve()
