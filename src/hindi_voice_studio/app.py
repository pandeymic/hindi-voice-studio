"""Gradio application for local Hindi/Hinglish/English voice cloning."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .engine import GenerationOptions, get_engine
from .storage import save_generation
from .text import LANGUAGE_CODES, language_code, normalize_text, pronunciation_preview

load_dotenv()


LANGUAGE_INFO = {
    "Hindi": "hi",
    "Hinglish": "hi",
    "English": "en",
}


def generate_audio(
    text: str,
    language: str,
    reference_audio: str | None,
    consent: bool,
    prepared_text: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    seed: float,
    progress=None,
    replacements: str = "",
    manual_override: bool = False,
):
    """Validate UI inputs and return a Gradio-compatible audio tuple."""

    if not consent:
        raise ValueError(
            "Confirm that you own this voice or have explicit permission to clone it."
        )
    if language not in LANGUAGE_CODES:
        raise ValueError("Choose Hindi, Hinglish, or English.")
    if manual_override:
        cleaned_text = normalize_text(prepared_text)
        if not cleaned_text:
            raise ValueError(
                "Manual pronunciation override is enabled, but the speech text is blank. "
                "Enter speech text or turn off the override."
            )
    else:
        # The preview is a convenience for the UI, not the source of truth. Rebuild it
        # from the current inputs so a stale preview cannot silently change the request.
        cleaned_text = pronunciation_preview(text, language, replacements)
    if not cleaned_text:
        raise ValueError("Enter Hindi, Hinglish, or English text first.")

    options = GenerationOptions(
        exaggeration=float(exaggeration),
        cfg_weight=float(cfg_weight),
        temperature=float(temperature),
        seed=int(seed or 0),
    )
    engine = get_engine()
    result = engine.generate(
        cleaned_text,
        language_code(language),
        reference_audio or "",
        options,
        progress=progress or (lambda value, description: None),
    )
    wav_path, metadata_path = save_generation(
        result=result,
        directory=Path(os.getenv("VOICE_STUDIO_OUTPUT_DIR", "outputs")),
        text=cleaned_text,
        language=language,
        options=options,
        model_version=engine.model_id(language_code(language)),
        device=engine.device,
        revision=engine.revision_for(language_code(language)),
    )
    status = (
        f"Generated {result.chunks} text chunk{'s' if result.chunks != 1 else ''} "
        f"at {result.sample_rate} Hz. Seed: `{result.seed}`.\n\n"
        f"Saved locally: `{wav_path}`\nMetadata: `{metadata_path}`\n\n"
        "Perth watermark included by Chatterbox."
    )
    return str(wav_path), status


def _generate_audio_from_ui(
    text: str,
    language: str,
    reference_audio: str | None,
    consent: bool,
    prepared_text: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    seed: float,
    replacements: str,
    manual_override: bool,
):
    """Adapt Gradio's inputs without treating them as the progress callback."""

    return generate_audio(
        text,
        language,
        reference_audio,
        consent,
        prepared_text,
        exaggeration,
        cfg_weight,
        temperature,
        seed,
        replacements=replacements,
        manual_override=manual_override,
    )


def build_demo():
    import gradio as gr

    with gr.Blocks(title="Hindi Voice Studio") as demo:
        gr.Markdown(
            """
            # Hindi Voice Studio
            ### Your voice. Hindi, Hinglish, or English. Generated locally.

            Upload **your own voice** or a recording you are explicitly authorized
            to use. The default target is Chatterbox Multilingual V3. A
            Hindi-specific loader (`VOICE_STUDIO_HINDI_MODEL=dedicated`) is an opt-in
            experimental setting and has not been audio-verified; generated samples
            stay local while this server is running.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                reference = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Reference voice (2–30 seconds)",
                )
                consent = gr.Checkbox(
                    label="I own this voice or have explicit permission to clone it.",
                    info="Required before generating audio.",
                )
                language = gr.Dropdown(
                    choices=list(LANGUAGE_INFO),
                    value="Hinglish",
                    label="Language",
                    info="Hinglish uses the Hindi model code and accepts Roman Hindi + English.",
                )
                replacements = gr.Textbox(
                    label="Hinglish pronunciation rules (optional)",
                    lines=2,
                    placeholder="kaafi=काफ़ी\nmain=main",
                    info="One rule per line: word=pronunciation. The preview stays editable.",
                )
                text = gr.Textbox(
                    label="Text",
                    lines=6,
                    max_length=3000,
                    value="Aaj hum ek chhota sa experiment karenge, aur result natural hona chahiye.",
                )
                prepared_text = gr.Textbox(
                    label="Speech text preview (edit before generating)",
                    lines=6,
                    info=(
                        "Hinglish help is conservative: a small dictionary is converted, "
                        "while English and unknown Roman-Hindi words stay as entered."
                    ),
                )
                manual_override = gr.Checkbox(
                    label="Use edited speech text exactly (manual override)",
                    value=False,
                    info=(
                        "Off: Generate prepares the current Text and rules again. On: "
                        "the edited preview is sent exactly as written."
                    ),
                )
                with gr.Accordion("Voice controls", open=True):
                    exaggeration = gr.Slider(
                        0.25, 1.5, value=0.5, step=0.05, label="Expressiveness / exaggeration"
                    )
                    cfg_weight = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05, label="CFG / pace"
                    )
                    temperature = gr.Slider(
                        0.05, 1.5, value=0.8, step=0.05, label="Temperature / variation"
                    )
                    seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
                generate = gr.Button("Generate voice", variant="primary", size="lg")

            with gr.Column(scale=1):
                output = gr.Audio(label="Generated audio", format="wav", autoplay=False)
                status = gr.Markdown("Ready. The first generation downloads the model.")
                gr.Markdown(
                    """
                    **Quality tips:** use 6–12 seconds of clean speech, avoid music and
                    reverb. Automatic Hinglish preparation is intentionally limited, so
                    use Devanagari for a Roman Hindi word that is pronounced incorrectly
                    or enable the manual override. Generated files include Chatterbox's
                    imperceptible Perth watermark.
                    """
                )

        generate.click(
            fn=_generate_audio_from_ui,
            inputs=[
                text,
                language,
                reference,
                consent,
                prepared_text,
                exaggeration,
                cfg_weight,
                temperature,
                seed,
                replacements,
                manual_override,
            ],
            outputs=[output, status],
        )
        preview_inputs = [text, language, replacements]
        for component in preview_inputs:
            component.change(
                fn=pronunciation_preview,
                inputs=preview_inputs,
                outputs=prepared_text,
                show_progress="hidden",
            )
        demo.load(
            fn=pronunciation_preview,
            inputs=preview_inputs,
            outputs=prepared_text,
            show_progress="hidden",
        )
    return demo


def main() -> None:
    import gradio as gr

    host = os.getenv("VOICE_STUDIO_HOST", "127.0.0.1")
    port = int(os.getenv("VOICE_STUDIO_PORT", "7860"))
    share = os.getenv("VOICE_STUDIO_SHARE", "false").lower() == "true"
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )
    build_demo().queue(max_size=2).launch(
        server_name=host, server_port=port, share=share, theme=theme,
    )


if __name__ == "__main__":
    main()
