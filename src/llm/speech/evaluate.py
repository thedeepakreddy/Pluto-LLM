"""Whisper-scored speech evaluation (M2): speak -> codec.decode -> ASR -> WER/CER.

``ceiling_wer`` scores the stored reference codes through the same codec + ASR (the best any
model could do with this TTS -> codec -> ASR chain); the floor is the same evaluation with a
freshly initialised model (``llm eval-speech --floor``).
"""

from __future__ import annotations

import sys
import time
from typing import Literal

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


def _progress(label: str, i: int, n: int) -> None:
    """One progress line every 10 items (and at the end) on stderr, so long evals are visible."""
    if label and ((i + 1) % 10 == 0 or i + 1 == n):
        print(f"{label}: {i + 1}/{n} ({100 * (i + 1) // n}%)", file=sys.stderr, flush=True)


def evaluate_speech(model, tok, codec: Codec, asr: ASR, examples: list[tuple[str, np.ndarray]],
                    audio: AudioConfig, sampling: SpeechSampling, label: str = "") -> dict:
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
        _progress(label, i, len(examples))
    wer, cer = _score(refs, hyps) if examples else (float("nan"), float("nan"))
    return {"wer": wer, "cer": cer,
            "duration_ratio": gen_frames / ref_frames if ref_frames else float("nan"),
            "rtf": real_time_factor(gen_s, gen_frames, codec.frame_rate),
            "n": len(examples), "per_item": items}


def ceiling_wer(codec: Codec, asr: ASR, examples: list[tuple[str, np.ndarray]], label: str = "") -> dict:
    """Decode the stored reference codes -> ASR -> WER/CER (the TTS -> codec -> ASR ceiling)."""
    refs, hyps = [], []
    for i, (text, codes) in enumerate(examples):
        refs.append(text)
        hyps.append(asr.transcribe(codec.decode(np.asarray(codes)), codec.sample_rate).text
                    if len(codes) else "")
        _progress(label, i, len(examples))
    wer, cer = _score(refs, hyps) if examples else (float("nan"), float("nan"))
    return {"wer": wer, "cer": cer, "n": len(examples)}


def greedy_text_continuation(model, tok, prompt: str, max_new_tokens: int = 24) -> str:
    """BOS + prompt tokens as text with EMPTY codes -> greedy next text tokens (specials excluded,
    stops at the context limit). Returns the decoded continuation."""
    import torch

    from llm.speech.layout import EMPTY, special_ids

    sp = special_ids(tok.vocab_size)
    device = next(model.parameters()).device
    ids = [sp.bos, *tok.encode(prompt)]
    new: list[int] = []
    was = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(ids) >= model.cfg.context_length:
                break
            text = torch.tensor([ids], dtype=torch.long, device=device)
            codes = torch.full((1, len(ids), model.K), EMPTY, dtype=torch.long, device=device)
            logits = model.text_head(model.hidden(text, codes)[0, -1])
            nxt = int(logits[: tok.vocab_size].argmax())
            ids.append(nxt)
            new.append(nxt)
    model.train(was)
    return tok.decode(new)


def identity_recall(model, tok, probes, mode: Literal["text", "speech"], codec: Codec | None = None,
                    asr: ASR | None = None, audio: AudioConfig | None = None) -> dict:
    """Fraction of probes whose continuation contains any expected string (case-insensitive).

    text: greedy text continuation of 24 tokens. speech: ``speak(prompt)`` -> decode ->
    ``continue_speech`` 60 frames -> ASR transcript of the continuation (aligned layout only).
    """
    from llm.speech.generate import continue_speech

    items: list[dict] = []
    per: dict[str, list[bool]] = {}
    for p in probes:
        if mode == "text":
            out = greedy_text_continuation(model, tok, p.prompt, 24)
        elif mode == "speech":
            if codec is None or asr is None or audio is None:
                raise ValueError("speech recall needs codec, asr and audio")
            sampling = SpeechSampling()
            prompt_codes = speak(model, p.prompt, tok, audio, sampling)
            out = ""
            if len(prompt_codes):
                cont = continue_speech(model, codec.decode(prompt_codes), codec.sample_rate, tok, codec,
                                       asr, audio, sampling, max_new_frames=60)
                if len(cont):
                    out = asr.transcribe(codec.decode(cont), codec.sample_rate).text
        else:
            raise ValueError(f"unknown mode {mode!r}")
        hit = any(e.lower() in out.lower() for e in p.expected)
        per.setdefault(p.fact, []).append(hit)
        items.append({"prompt": p.prompt, "expected": list(p.expected), "fact": p.fact,
                      "output": out, "hit": hit})
    n = len(items)
    return {"recall": sum(i["hit"] for i in items) / n if n else float("nan"),
            "per_fact": {f: sum(v) / len(v) for f, v in per.items()}, "items": items}


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
    path = run_dir / "speech_eval.json"

    def save():  # after every phase, so an interrupted eval keeps the finished phases
        atomic_write_json(path, json.loads(json.dumps(out)))

    out = {"set": "held-out test, first chunks", "n_requested": n, "checkpoint": ckpt.name}
    out["ceiling"] = ceiling_wer(codec, asr, examples, label="ceiling")
    save()
    out["model"] = evaluate_speech(model, tok, codec, asr, examples, cfg.audio, sampling, label="model")
    save()
    if floor:
        torch.manual_seed(0)
        device = next(model.parameters()).device  # the floor must run where the model runs
        fresh = SpeechGPT(cfg.model, cfg.audio, tok.vocab_size + 4).to(device).eval()
        out["floor"] = evaluate_speech(fresh, tok, codec, asr, examples, cfg.audio, sampling, label="floor")
        save()
    if cfg.identity is not None:
        from llm.identity.pcid import identity_dir, load_probes

        probes = load_probes(identity_dir(cfg))
        ident = {"n_probes": len(probes), "text": identity_recall(model, tok, probes, "text")}
        if cfg.audio.layout == "aligned":
            ident["speech"] = identity_recall(model, tok, probes, "speech", codec, asr, cfg.audio)
        out["identity"] = ident
    save()
    return path
