"""Serialized, lazy multilingual inference. Importing this module doesn't load torch."""

from __future__ import annotations

import os
import random
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from .audio import prepare_reference
from .text import chunk_text

MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
MODEL_FILES = {"v2": "t3_mtl23ls_v2.safetensors", "v3": "t3_mtl23ls_v3.safetensors"}


@dataclass(frozen=True)
class GenerationOptions:
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    seed: int = 0

    def validate(self):
        for name, value, lower, upper in (
            ("Expressiveness", self.exaggeration, 0.25, 1.5),
            ("CFG", self.cfg_weight, 0.0, 1.0),
            ("Temperature", self.temperature, 0.05, 1.5),
        ):
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}.")
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer between 0 and 4294967295.")


@dataclass
class GenerationResult:
    sample_rate: int
    audio: np.ndarray
    chunks: int
    seed: int
    reference_description: str


def select_device(preference: str = "auto") -> str:
    import torch

    preference = preference.lower().strip()
    if preference in {"mps", "cuda", "cpu"}:
        if preference == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable. Set VOICE_STUDIO_DEVICE=cpu in .env.")
        if preference == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Set VOICE_STUDIO_DEVICE=auto in .env.")
        return preference
    if preference != "auto":
        raise ValueError("VOICE_STUDIO_DEVICE must be auto, mps, cuda, or cpu.")
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class VoiceEngine:
    def __init__(self, device: str = "auto", model_version: str = "v3"):
        if model_version not in MODEL_FILES:
            raise ValueError("CHATTERBOX_MULTILINGUAL_T3_MODEL must be v2 or v3.")
        self.device = device
        self.model_version = model_version
        self.revision = os.getenv("VOICE_STUDIO_MODEL_REVISION", MODEL_REVISION)
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is None:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            try:
                import torch
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError(
                    "The local model dependencies are missing or incompatible. "
                    "Run `uv sync --extra local` in the project directory."
                ) from exc
            self.device = select_device(self.device)
            torch.set_num_threads(max(1, int(os.getenv("VOICE_STUDIO_THREADS", "4"))))
            # Pin weights independently of the package version for repeatable runs.
            checkpoint = snapshot_download(
                repo_id="ResembleAI/chatterbox", revision=self.revision,
                allow_patterns=[
                    "ve.pt", MODEL_FILES[self.model_version], "s3gen.pt",
                    "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json",
                ],
                token=os.getenv("HF_TOKEN"),
            )
            self._model = ChatterboxMultilingualTTS.from_local(
                Path(checkpoint), self.device, t3_model=self.model_version,
            )
        return self._model

    def generate(
        self, text: str, language_id: str, reference_path: str,
        options: GenerationOptions,
        progress: Callable[[float, str], None] = lambda value, description: None,
    ) -> GenerationResult:
        options.validate()
        if language_id not in {"hi", "en"}:
            raise ValueError("Choose Hindi, Hinglish, or English.")
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Enter some text to generate.")
        reference = prepare_reference(reference_path)
        actual_seed = options.seed or secrets.randbelow(2**32 - 1) + 1
        # Speaker conditionals and global RNGs are mutable. Lock the whole run,
        # not just model loading, so concurrent requests cannot mix speakers.
        with self._lock, TemporaryDirectory(prefix="hindi-voice-") as temporary:
            progress(0.03, "Loading model; the first run downloads weights…")
            model = self._load_model()
            import torch

            random.seed(actual_seed)
            np.random.seed(actual_seed)
            torch.manual_seed(actual_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(actual_seed)
            prompt = Path(temporary) / "reference.wav"
            sf.write(prompt, reference.samples, reference.sample_rate, subtype="PCM_16")
            waveforms = []
            with torch.inference_mode():
                model.prepare_conditionals(str(prompt), exaggeration=options.exaggeration)
                for index, chunk in enumerate(chunks):
                    progress(0.1 + 0.85 * index / len(chunks), f"Speaking {index + 1}/{len(chunks)}…")
                    wav = model.generate(
                        chunk, language_id=language_id, exaggeration=options.exaggeration,
                        cfg_weight=options.cfg_weight, temperature=options.temperature,
                    )
                    audio = np.asarray(wav.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
                    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                        raise RuntimeError("The model returned invalid audio. Try a shorter sentence.")
                    if index:
                        waveforms.append(np.zeros(int(model.sr * 0.12), dtype=np.float32))
                    waveforms.append(audio)
            progress(1.0, "Audio ready")
            return GenerationResult(
                int(model.sr), np.concatenate(waveforms), len(chunks),
                actual_seed, reference.description,
            )


_ENGINE: VoiceEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> VoiceEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = VoiceEngine(
                os.getenv("VOICE_STUDIO_DEVICE", "auto"),
                os.getenv("CHATTERBOX_MULTILINGUAL_T3_MODEL", "v3"),
            )
        return _ENGINE
