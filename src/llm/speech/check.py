"""Stage 0 sanity check of the real TTS / codec / ASR on a handful of val chunks."""

from __future__ import annotations

import json
import time

import numpy as np

from llm.config import Config
from llm.data.prepare import data_dir
from llm.speech import interfaces
from llm.speech.build import WORDS_PER_SECOND
from llm.speech.chunk import TimingMismatch, chunk_story, token_start_frames
from llm.speech.evaluate import ceiling_wer
from llm.tokenizer.bpe import BPETokenizer, tokenizer_dir

MISSING_LIMIT = 0.05


def run_speech_check(cfg: Config, n: int = 100) -> str:
    sp = cfg.speech
    tts, codec, asr = interfaces.make_tts(sp), interfaces.make_codec(sp), interfaces.make_asr(sp)
    tok = BPETokenizer.load(tokenizer_dir(cfg))
    max_words = max(1, int(sp.max_seconds * WORDS_PER_SECOND))
    texts: list[str] = []
    with open(data_dir(cfg) / "val.jsonl", encoding="utf-8") as f:
        for line in f:
            texts.extend(chunk_story(json.loads(line)["text"], max_words))
            if len(texts) >= n:
                break
    texts = texts[:n]
    if not texts:
        return "speech-check: no val chunks available"

    t_tts = t_codec = t_asr = audio_s = 0.0
    usable, errs, examples = 0, [], []
    for text in texts:
        t0 = time.perf_counter()
        syn = tts.synthesize(text)
        t_tts += time.perf_counter() - t0
        dur = len(syn.audio) / syn.sample_rate
        audio_s += dur
        if syn.words:
            try:
                token_start_frames(text, tok.encode(text), tok, syn.words, codec.frame_rate)
                usable += 1
                errs.append(abs(syn.words[-1].end_s - dur))
            except TimingMismatch:
                pass
        t0 = time.perf_counter()
        codes = codec.encode(syn.audio, syn.sample_rate)
        t_codec += time.perf_counter() - t0
        examples.append((text, codes, syn))

    ceil_in = []
    for text, codes, syn in examples:
        t0 = time.perf_counter()
        wav = codec.decode(codes) if len(codes) else np.zeros(0, np.float32)
        t_codec += time.perf_counter() - t0
        t0 = time.perf_counter()
        asr.transcribe(wav, codec.sample_rate)
        t_asr += time.perf_counter() - t0
        ceil_in.append((text, codes))
    wer = ceiling_wer(codec, asr, ceil_in)["wer"]

    per = lambda s: s / audio_s if audio_s else float("nan")  # noqa: E731
    share = usable / len(texts)
    lines = [
        f"speech-check on {len(texts)} val chunks ({audio_s:.1f} s of audio)",
        f"usable timings:   {share:.1%} ({usable}/{len(texts)})",
        f"duration error:   {float(np.mean(errs)):.3f} s mean |timing end - audio duration|"
        if errs else "duration error:   n/a (no usable timings)",
        f"round-trip WER:   {wer:.3f} (TTS -> Mimi -> ASR ceiling)",
        f"s/s TTS:          {per(t_tts):.3f}",
        f"s/s codec:        {per(t_codec):.3f} (encode + decode)",
        f"s/s ASR:          {per(t_asr):.3f}",
    ]
    if 1 - share > MISSING_LIMIT:
        lines.append(f"WARNING: TTS timings unusable for {1 - share:.1%} of chunks (> 5%); "
                     f"rebuild with `llm build-speech <config> --timings whisper`")
    return "\n".join(lines)
