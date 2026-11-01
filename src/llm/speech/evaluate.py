"""Whisper-scored speech evaluation (M2): speak -> codec.decode -> ASR -> WER/CER.

``ceiling_wer`` scores the stored reference codes through the same codec + ASR (the best any
model could do with this TTS -> codec -> ASR chain); the floor is the same evaluation with a
freshly initialised model (``llm eval-speech --floor``).
"""

from __future__ import annotations

import time

import numpy as np

from llm.config import AudioConfig
from llm.speech.generate import SpeechSampling, real_time_factor, speak
from llm.speech.interfaces import ASR, Codec
from llm.speech.wer import char_error_rate, word_error_rate

DEMO_SET_VERSION = "speech-demo-v1"
# Frozen: changing any sentence changes DEMO_SET_VERSION (checksum test in tests).
DEMO_SENTENCES = [
    "Tom found a little boat under a tree.",
    "The sun was warm and the sky was blue.",
    "Mia gave her dog a big red ball.",
    "They walked to the park and played all day.",
    "The cat sat on the soft green mat.",
    "Ben was happy to help his mom bake a cake.",
    "A small bird sang a song in the morning.",
    "Lily saw a bright star and made a wish.",
    "The boy and his dad went to the lake.",
    "It was a good day, and everyone smiled.",
]


def _score(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    """Corpus-level WER/CER: concatenate the items (errors are weighted by length)."""
    return word_error_rate(" ".join(refs), " ".join(hyps)), char_error_rate(" ".join(refs), " ".join(hyps))


def evaluate_speech(model, tok, codec: Codec, asr: ASR, examples: list[tuple[str, np.ndarray]],
                    audio: AudioConfig, sampling: SpeechSampling) -> dict:
    items, refs, hyps = [], [], []
    gen_s, gen_frames, n_frames, ref_frames = 0.0, 0, 0, 0
    for i, (text, ref_codes) in enumerate(examples):
        t0 = time.perf_counter()
        codes = speak(model, text, tok, audio, sampling)
        dt = time.perf_counter() - t0
        hyp = ""
        if len(codes):
            hyp = asr.transcribe(codec.decode(codes), codec.sample_rate).text
        n, r = int(len(codes)), int(len(ref_codes))
        items.append({"i": i, "text": text, "hypothesis": hyp, "frames": n, "ref_frames": r,
                      "wer": word_error_rate(text, hyp), "cer": char_error_rate(text, hyp),
                      "duration_ratio": n / r if r else None})
        refs.append(text)
        hyps.append(hyp)
        gen_s, gen_frames, ref_frames = gen_s + dt, gen_frames + n, ref_frames + r
    wer, cer = _score(refs, hyps) if examples else (float("nan"), float("nan"))
    return {"wer": wer, "cer": cer,
            "duration_ratio": gen_frames / ref_frames if ref_frames else float("nan"),
            "rtf": real_time_factor(gen_s, gen_frames, codec.frame_rate),
            "n": len(examples), "per_item": items}


def ceiling_wer(codec: Codec, asr: ASR, examples: list[tuple[str, np.ndarray]]) -> dict:
    """Decode the stored reference codes -> ASR -> WER/CER (the TTS -> codec -> ASR ceiling)."""
    refs, hyps = [], []
    for text, codes in examples:
        refs.append(text)
        hyps.append(asr.transcribe(codec.decode(np.asarray(codes)), codec.sample_rate).text
                    if len(codes) else "")
    wer, cer = _score(refs, hyps) if examples else (float("nan"), float("nan"))
    return {"wer": wer, "cer": cer, "n": len(examples)}


def run_eval_speech(run_dir, n: int = 200, floor: bool = False):
    """Score a speech run on its first ``n`` test chunks; writes ``<run>/speech_eval.json``."""
    import json
    from pathlib import Path

    import torch

    from llm.manifest import atomic_write_json
    from llm.model.voice import SpeechGPT
    from llm.speech.build import speech_dir
    from llm.speech.dataset import SpeechShards
    from llm.speech.interfaces import make_asr, make_codec
    from llm.train.checkpoint import find_latest_valid, load_speech_model_for_inference

    run_dir = Path(run_dir)
    model, tok, cfg = load_speech_model_for_inference(run_dir)
    if cfg.speech is None:
        raise ValueError("config has no speech section")
    ds = SpeechShards(speech_dir(cfg), "test", cfg.audio.codebooks)
    examples = [(ds.text(i), ds[i][2]) for i in range(min(n, len(ds)))]
    codec, asr = make_codec(cfg.speech), make_asr(cfg.speech)
    sampling = SpeechSampling()
    ckpt = find_latest_valid(run_dir)
    out = {"set": "held-out test, first chunks", "n_requested": n, "checkpoint": ckpt.name,
           "ceiling": ceiling_wer(codec, asr, examples),
           "model": evaluate_speech(model, tok, codec, asr, examples, cfg.audio, sampling)}
    if floor:
        torch.manual_seed(0)
        fresh = SpeechGPT(cfg.model, cfg.audio, tok.vocab_size + 4).eval()
        out["floor"] = evaluate_speech(fresh, tok, codec, asr, examples, cfg.audio, sampling)
    path = run_dir / "speech_eval.json"
    atomic_write_json(path, json.loads(json.dumps(out)))
    return path
