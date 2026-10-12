# Hindi Voice Studio

Local, consent-first voice cloning for **Hindi**, **Hinglish**, and **English**.
Upload a short reference recording, write text, and generate a new speech sample
that preserves the reference speaker's voice characteristics. The app runs on
your machine using [Chatterbox Multilingual V3](https://github.com/resemble-ai/chatterbox).

This is a starting point for a production-quality voice product, not a claim that
any open-source model will universally outperform ElevenLabs. Quality depends on
the reference recording, language, text normalization, hardware, and model
settings. The UI exposes the important controls so samples can be compared.

## What is included

- Hindi text-to-speech with the V3 multilingual checkpoint.
- Hinglish input: select `Hinglish` for Roman Hindi mixed with English; use
  Devanagari directly when you want Hindi pronunciation to be unambiguous.
- English text-to-speech.
- Zero-shot cloning from an uploaded file or microphone recording.
- Local inference with MPS support on Apple Silicon, CUDA support when available,
  and CPU fallback.
- A required consent/authorization confirmation before generation.
- Automatic Chatterbox Perth watermarking in generated audio.
- Long-text chunking so the demo is not limited to a single short sentence.
- No audio files, API keys, or model weights are committed to Git.

## Responsible use

Only clone your own voice or a voice for which you have explicit permission. Do
not impersonate people, create deceptive calls, bypass identity checks, or publish
synthetic speech without disclosure where disclosure is required. The consent
checkbox is a product safeguard, not a substitute for permission.

Generated Chatterbox audio contains an imperceptible Perth watermark. Keep that
watermark intact when distributing generated files.

## Quick start on an M1/M2/M3 Mac

Python 3.11 is recommended for the current Chatterbox dependency set.

```bash
git clone https://github.com/pandeymic/hindi-voice-studio.git
cd hindi-voice-studio

uv venv --python 3.11
source .venv/bin/activate
uv sync --extra local

cp .env.example .env
python -m hindi_voice_studio.app
```

Open <http://127.0.0.1:7860>. The first generation downloads the model from
Hugging Face and can require several GB of disk space and memory. Later runs use
the local cache.

If `uv` is not installed, create the environment with `python3.11 -m venv .venv`
and install with `pip install -e '.[local]'`.

## Reference recording tips

For the best similarity, record **6–12 seconds** of clean speech:

- one speaker only;
- quiet room, no music or reverb;
- the same microphone and distance you want to sound like;
- natural conversational delivery;
- avoid clipping and long silence.

The app accepts common audio formats supported by `librosa`/`soundfile`. If a
format cannot be decoded, convert it to WAV with `ffmpeg`:

```bash
ffmpeg -i reference.m4a -ar 24000 -ac 1 reference.wav
```

## Text examples

**Hindi** (`hi`):

```text
आज हम एक छोटा सा प्रयोग करेंगे। आवाज़ साफ़ और natural रखनी है।
```

**Hinglish** (`hi` internally):

```text
Aaj hum ek chhota sa experiment karenge, aur result kaafi natural hona chahiye.
```

**English** (`en`):

```text
This is a natural English sample in the same reference voice.
```

For Hinglish, keep Hindi words in Roman script and English words in English. The
app shows a conservative pronunciation preview. Edit that preview directly, or
add one `word=pronunciation` rule per line. If a Roman Hindi word is pronounced
incorrectly, write that word in Devanagari.

## Settings

| Control | Starting value | What it does |
| --- | ---: | --- |
| Exaggeration | `0.50` | Expressiveness. Higher values can sound dramatic or unstable. |
| CFG / pace | `0.50` | Guidance and pacing. Lower values can reduce reference-language transfer. |
| Temperature | `0.80` | Variation. Lower values are more repeatable. |
| Seed | `0` | `0` is random; use a fixed integer for comparison. |

## Architecture

```text
Gradio UI
   -> consent + audio validation
   -> text normalization and sentence chunking
   -> ChatterboxMultilingualTTS (V3, hi/en)
   -> Perth-watermarked WAV
```

The model is loaded lazily and cached for the process. A provider boundary can be
added later for a hosted provider such as ElevenLabs without changing the UI
contract, but the default is local inference so the uploaded voice sample stays
on the user's machine.

## Development

```bash
source .venv/bin/activate
pytest
ruff check .
```

The tests cover text normalization, chunking, and input validation without
downloading model weights. An end-to-end generation test requires the model
checkpoint and suitable RAM/GPU, so it is intentionally opt-in through the app.

GitHub Actions runs the lightweight test and lint suite on every push and pull
request. Install the `local` extra only on a machine where you want to download
PyTorch and run voice generation.

## License

The application code is MIT licensed. Chatterbox, PyTorch, and model checkpoints
retain their own licenses and terms; review them before commercial distribution.
