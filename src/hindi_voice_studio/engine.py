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
HINDI_MODEL_REVISION = "82ca71273cc2a9ab19efdf8315f865c1a5af0ee7"
MODEL_FILES = {"v2": "t3_mtl23ls_v2.safetensors", "v3": "t3_mtl23ls_v3.safetensors"}
MULTILINGUAL_REPO = "ResembleAI/chatterbox"
HINDI_REPO = "ResembleAI/Chatterbox-Multilingual-hi"


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
    def __init__(
        self, device: str = "auto", model_version: str = "v3", hindi_model: str = "multilingual"
    ):
        if model_version not in MODEL_FILES:
            raise ValueError("CHATTERBOX_MULTILINGUAL_T3_MODEL must be v2 or v3.")
        if hindi_model not in {"dedicated", "multilingual"}:
            raise ValueError("VOICE_STUDIO_HINDI_MODEL must be dedicated or multilingual.")
        self.device = device
        self.model_version = model_version
        self.hindi_model = hindi_model
        self.revision = os.getenv("VOICE_STUDIO_MODEL_REVISION", MODEL_REVISION)
        self.hindi_revision = os.getenv("VOICE_STUDIO_HINDI_MODEL_REVISION", HINDI_MODEL_REVISION)
        self._models = {}
        self._lock = threading.Lock()

    def model_id(self, language_id: str) -> str:
        """Describe the model selected for a language in generated metadata."""

        if language_id == "hi" and self.hindi_model == "dedicated":
            return "ResembleAI/Chatterbox-Multilingual-hi"
        return f"{MULTILINGUAL_REPO}:{self.model_version}"

    def revision_for(self, language_id: str) -> str:
        if language_id == "hi" and self.hindi_model == "dedicated":
            return self.hindi_revision
        return self.revision

    def _load_multilingual_model(self):
        if "multilingual" not in self._models:
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
                repo_id=MULTILINGUAL_REPO, revision=self.revision,
                allow_patterns=[
                    "ve.pt", MODEL_FILES[self.model_version], "s3gen.pt",
                    "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json",
                ],
                token=os.getenv("HF_TOKEN"),
            )
            self._models["multilingual"] = ChatterboxMultilingualTTS.from_local(
                Path(checkpoint), self.device, t3_model=self.model_version,
            )
        return self._models["multilingual"]

    def _load_hindi_model(self):
        """Load the upstream Hindi single-language V3 finetune.

        It supplies Hindi-specific T3 and S3Gen weights. The speaker encoder and
        tokenizer are shared assets from the base Chatterbox repository.
        """
        if "hindi" not in self._models:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            try:
                import torch
                from chatterbox.models.s3gen import S3Gen
                from chatterbox.models.t3 import T3
                from chatterbox.models.t3.modules.t3_config import T3Config
                from chatterbox.models.tokenizers import MTLTokenizer
                from chatterbox.models.voice_encoder import VoiceEncoder
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                from huggingface_hub import snapshot_download
                from safetensors.torch import load_file as load_safetensors
            except ImportError as exc:
                raise RuntimeError(
                    "The local model dependencies are missing or incompatible. "
                    "Run `uv sync --extra local` in the project directory."
                ) from exc

            self.device = select_device(self.device)
            shared = snapshot_download(
                repo_id=MULTILINGUAL_REPO,
                revision=self.revision,
                allow_patterns=["ve.pt", "grapheme_mtl_merged_expanded_v1.json"],
                token=os.getenv("HF_TOKEN"),
            )
            hindi = snapshot_download(
                repo_id=HINDI_REPO,
                revision=self.hindi_revision,
                allow_patterns=["t3_hi.safetensors", "s3gen_v3.pt", "grapheme_mtl_merged_expanded_v1.json"],
                token=os.getenv("HF_TOKEN"),
            )
            shared_path = Path(shared)
            hindi_path = Path(hindi)
            map_location = torch.device("cpu") if self.device in {"cpu", "mps"} else None

            ve = VoiceEncoder()
            ve.load_state_dict(torch.load(shared_path / "ve.pt", map_location=map_location, weights_only=True))
            ve.to(self.device).eval()

            t3 = T3(T3Config.multilingual())
            t3.load_state_dict(load_safetensors(hindi_path / "t3_hi.safetensors"))
            t3.to(self.device).eval()

            s3gen = S3Gen()
            s3gen_state = torch.load(
                hindi_path / "s3gen_v3.pt", map_location=map_location, weights_only=True
            )
            load_result = s3gen.load_state_dict(s3gen_state, strict=False)
            ignored_missing = set(getattr(s3gen, "ignore_state_dict_missing", ()))
            unexpected = set(load_result.unexpected_keys)
            missing = set(load_result.missing_keys) - ignored_missing
            if missing or unexpected:
                raise RuntimeError(
                    "The Hindi V3 S3Gen checkpoint is incompatible with the installed "
                    f"Chatterbox loader (missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)})."
                )
            s3gen.to(self.device).eval()
            tokenizer_path = hindi_path / "grapheme_mtl_merged_expanded_v1.json"
            if not tokenizer_path.exists():
                tokenizer_path = shared_path / "grapheme_mtl_merged_expanded_v1.json"
            tokenizer = MTLTokenizer(str(tokenizer_path))
            self._models["hindi"] = ChatterboxMultilingualTTS(
                t3, s3gen, ve, tokenizer, self.device,
            )
        return self._models["hindi"]

    def _load_model(self, language_id: str):
        if language_id == "hi" and self.hindi_model == "dedicated":
            return self._load_hindi_model()
        return self._load_multilingual_model()

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
            model = self._load_model(language_id)
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
                os.getenv("VOICE_STUDIO_HINDI_MODEL", "multilingual"),
            )
        return _ENGINE
