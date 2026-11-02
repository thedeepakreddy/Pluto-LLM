"""Gradio demo for a speech run. gradio is imported only inside ``build_app``."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def build_app(run_dir: Path):
    import gradio as gr

    from llm.speech.generate import SpeechSampling, continue_speech, speak
    from llm.speech.interfaces import make_asr, make_codec
    from llm.train.checkpoint import load_speech_model_for_inference

    model, tok, cfg = load_speech_model_for_inference(Path(run_dir))
    codec, asr = make_codec(cfg.speech), make_asr(cfg.speech)
    sr = codec.sample_rate

    def _speak(text, temperature, top_k, seed):
        sampling = SpeechSampling(temperature=float(temperature), top_k=int(top_k), seed=int(seed))
        try:
            codes = speak(model, text, tok, cfg.audio, sampling)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        return sr, codec.decode(codes)

    def _continue(mic, temperature, top_k, seed, max_new_frames):
        if mic is None:
            raise gr.Error("record or upload some audio first")
        in_sr, x = mic
        x = np.asarray(x)
        if x.dtype.kind == "i":
            x = x.astype(np.float32) / np.iinfo(x.dtype).max
        if x.ndim > 1:
            x = x.mean(axis=1)
        sampling = SpeechSampling(temperature=float(temperature), top_k=int(top_k), seed=int(seed))
        codes = continue_speech(model, x.astype(np.float32), int(in_sr), tok, codec, asr, cfg.audio,
                                sampling, max_new_frames=int(max_new_frames))
        return sr, codec.decode(codes)

    knobs = lambda: [gr.Slider(0.0, 1.5, 0.8, label="temperature"),
                     gr.Slider(1, 200, 50, step=1, label="top-k"),
                     gr.Number(1234, label="seed", precision=0)]
    with gr.Blocks(title="Pluto") as demo:
        with gr.Tab("Text to speech"):
            text = gr.Textbox(label="text")
            k = knobs()
            out = gr.Audio(label="speech", type="numpy")
            gr.Button("Speak").click(_speak, [text, *k], out)
        with gr.Tab("Continue speech"):
            mic = gr.Audio(sources=["microphone", "upload"], type="numpy", label="prompt")
            k2 = knobs()
            n = gr.Slider(5, 400, 250, step=1, label="max new frames")
            out2 = gr.Audio(label="continuation", type="numpy")
            gr.Button("Continue").click(_continue, [mic, *k2, n], out2)
    return demo
