"""Text preparation for multilingual speech generation."""

from __future__ import annotations

import re
import unicodedata

LANGUAGE_CODES = {
    "Hindi": "hi",
    "Hinglish": "hi",
    "English": "en",
}

# Chatterbox's multilingual examples use short prompts. Keeping chunks below
# 300 characters leaves room for punctuation normalization inside the model.
MAX_CHUNK_CHARS = 260
MAX_TEXT_CHARS = 3000

# Deliberately small and inspectable. Ambiguous tokens such as "is", "me", and
# "main" are left alone; this is a pronunciation aid, not an Indic language model.
HINGLISH_WORDS = {
    "aaj": "आज", "aap": "आप", "aapka": "आपका", "aapki": "आपकी",
    "aapko": "आपको", "abhi": "अभी", "achha": "अच्छा", "accha": "अच्छा",
    "acchi": "अच्छी", "achhi": "अच्छी", "apna": "अपना", "apni": "अपनी",
    "aur": "और", "awaaz": "आवाज़", "awaz": "आवाज़", "bahut": "बहुत",
    "bhi": "भी", "chahiye": "चाहिए", "chhota": "छोटा", "chota": "छोटा",
    "dhanyavaad": "धन्यवाद", "dosto": "दोस्तों", "doston": "दोस्तों",
    "duniya": "दुनिया", "hai": "है", "hain": "हैं", "hamara": "हमारा",
    "hamari": "हमारी", "ho": "हो", "hoga": "होगा", "hona": "होना",
    "hoti": "होती", "hota": "होता", "hoon": "हूँ", "hum": "हम",
    "humne": "हमने", "ka": "का", "kaafi": "काफ़ी", "kaise": "कैसे",
    "kar": "कर", "karenge": "करेंगे", "karna": "करना", "karte": "करते",
    "ke": "के", "ki": "की", "ko": "को", "kuch": "कुछ", "kya": "क्या",
    "kyunki": "क्योंकि", "lekin": "लेकिन", "liye": "लिए", "mein": "में",
    "mera": "मेरा", "meri": "मेरी", "mujhe": "मुझे", "nahi": "नहीं",
    "nahin": "नहीं", "namaste": "नमस्ते", "namaskar": "नमस्कार",
    "naya": "नया", "nayi": "नई", "raha": "रहा", "rahe": "रहे",
    "rahi": "रही", "rakhenge": "रखेंगे", "rakhna": "रखना", "sa": "सा",
    "sabhi": "सभी", "saath": "साथ", "se": "से", "sirf": "सिर्फ़",
    "suniye": "सुनिए", "swagat": "स्वागत", "thoda": "थोड़ा", "toh": "तो",
    "tum": "तुम", "ye": "ये", "yeh": "यह", "yahan": "यहाँ",
}


def language_code(label: str) -> str:
    """Return the Chatterbox language code for a user-facing language label."""

    try:
        return LANGUAGE_CODES[label]
    except KeyError as exc:
        raise ValueError(f"Unsupported language: {label}") from exc


def normalize_text(text: str) -> str:
    """Normalize whitespace while retaining Hindi and Hinglish characters."""

    text = unicodedata.normalize("NFC", text or "")
    text = " ".join(text.split())
    text = text.replace("…", "...").replace("–", "-").replace("—", "-")
    return text.strip()


def pronunciation_preview(text: str, language: str, replacements: str = "") -> str:
    """Suggest mixed-script Hinglish, with user overrides of word=pronunciation.

    The returned preview stays editable. Nothing is sent to an external service.
    A custom identity replacement (e.g. `main=main`) can preserve a word.
    """
    language_code(language)
    result = normalize_text(text)
    if len(result) > MAX_TEXT_CHARS:
        raise ValueError(f"Use at most {MAX_TEXT_CHARS} characters per generation.")
    mapping = dict(HINGLISH_WORDS) if language == "Hinglish" else {}
    for line in replacements.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise ValueError("Pronunciation rules need one word=pronunciation per line.")
        mapping[key.strip().casefold()] = value.strip()
    if mapping:
        # Longest phrase first, with Unicode word boundaries; do one pass so a
        # replacement is not accidentally rewritten by another rule.
        pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
            + r")(?!\w)", re.IGNORECASE,
        )
        result = pattern.sub(lambda match: mapping[match.group().casefold()], result)
    return result


def _split_long_piece(piece: str, limit: int) -> list[str]:
    words = piece.split()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        if len(word) > limit:
            raise ValueError(f"A single word exceeds {limit} characters. Add spaces or punctuation.")
        extra = len(word) if not current else len(word) + 1
        if current and current_length + extra > limit:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
        current.append(word)
        current_length += len(word) if len(current) == 1 else len(word) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text at sentence boundaries, then at words when necessary."""

    if limit < 40:
        raise ValueError("Text chunk limit must be at least 40 characters")

    normalized = normalize_text(text)
    if not normalized:
        return []

    if len(normalized) > MAX_TEXT_CHARS:
        raise ValueError(f"Use at most {MAX_TEXT_CHARS} characters per generation.")
    sentences = re.split(r"(?<=[!?।॥])\s+|(?<=\.)\s+", normalized)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > limit:
            chunks.extend(_split_long_piece(current, limit))
            current = sentence
        else:
            current = candidate
    if current:
        chunks.extend(_split_long_piece(current, limit))
    return chunks
