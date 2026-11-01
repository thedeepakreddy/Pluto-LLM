"""Resumable builder of the synthetic-speech dataset (TinyStories-Speech).

For every cleaned M1 story: split into sentence chunks, synthesize each with the TTS, align
text tokens to audio frames, optionally spot-check with ASR, encode with the codec, and append
to the per-split store. Order is test, val, train so the evaluation splits are always complete
first. The build can be killed at any moment and re-run; see ``llm.speech.store``.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from llm.config import Config
from llm.data.prepare import data_dir
from llm.manifest import (
    MANIFEST,
    ProvenanceError,
    atomic_write_bytes,
    atomic_write_json,
    manifest_checksum,
    read_manifest,
    verify_files,
    write_manifest,
)
from llm.speech.chunk import TimingMismatch, chunk_story, token_start_frames
from llm.speech.interfaces import ASR, TTS, Codec
from llm.speech.store import SplitStore
from llm.speech.wer import word_error_rate
from llm.tokenizer.bpe import BPETokenizer, tokenizer_dir

SPLITS = ("test", "val", "train")  # build order
WORDS_PER_SECOND = 2.5  # chunk size heuristic: max_seconds * this words (longer audio is dropped)
STATS = "stats.json"


def speech_dir(cfg: Config) -> Path:
    return Path(cfg.speech.root) / f"{cfg.speech.name}-v{cfg.speech.version}"


def _clock() -> float:  # indirection so tests can fake wall-clock time
    return time.monotonic()


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _read_docs(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def _in_qa_sample(chunk_id: str, fraction: float) -> bool:
    return int(hashlib.sha256(chunk_id.encode()).hexdigest(), 16) % 10000 < fraction * 10000


class _Stats:
    """Drop bookkeeping persisted to stats.json after every chunk (atomic replace).

    Dropped chunk ids are remembered, so a resumed build neither retries them nor counts them
    twice. Kept chunks are counted from the split indexes, which are the source of truth.
    """

    def __init__(self, path: Path, fingerprint: dict, stores: dict[str, SplitStore]):
        self.path, self.fingerprint, self.stores = path, fingerprint, stores
        self.dropped: dict[str, str] = {}
        if path.exists():
            old = json.loads(path.read_text())
            if old["fingerprint"] != fingerprint:
                raise ProvenanceError(
                    f"{path.parent}: this partial build was started with a different recipe or "
                    f"different inputs; delete the directory or bump speech.version"
                )
            self.dropped = old["dropped"]

    def drop(self, chunk_id: str, reason: str) -> None:
        self.dropped[chunk_id] = reason
        self.save()

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for reason in self.dropped.values():
            out[reason] = out.get(reason, 0) + 1
        return dict(sorted(out.items()))

    def save(self) -> None:
        kept = sum(len(s.records) for s in self.stores.values())
        atomic_write_json(self.path, {
            "fingerprint": self.fingerprint, "attempted": kept + len(self.dropped), "kept": kept,
            "drops": self.counts(), "dropped": dict(sorted(self.dropped.items())),
        })


def build_speech_data(cfg: Config, tts: TTS, codec: Codec, asr: ASR,
                      max_seconds_this_session: float | None = None,
                      progress: Callable[[str], None] | None = None) -> tuple[Path, bool]:
    """Build (or resume) the speech dataset. Returns ``(directory, complete)``.

    ``max_seconds_this_session`` is a wall-clock budget: the build stops cleanly between
    chunks once it is used up (``complete`` is then False and no manifest is written).
    A split with fewer documents than requested is complete once its source is exhausted.
    """
    sp = cfg.speech
    if sp is None:
        raise ValueError("config has no speech section")
    say = progress or (lambda _msg: None)
    out = speech_dir(cfg)
    ddir, tdir = data_dir(cfg), tokenizer_dir(cfg)
    inputs = {"data": manifest_checksum(ddir), "tokenizer": manifest_checksum(tdir)}
    fingerprint = {"inputs": inputs, "voice": sp.voice, "max_seconds": sp.max_seconds,
                   "tts": sp.tts, "codec": sp.codec, "asr": sp.asr, "asr_model": sp.asr_model,
                   "qa_fraction": sp.qa_fraction, "qa_max_wer": sp.qa_max_wer,
                   "frame_rate": codec.frame_rate, "codebooks": codec.codebooks}
    if (out / MANIFEST).exists():
        if read_manifest(out)["inputs"] != inputs:
            raise ProvenanceError(f"{out}: built from different data/tokenizer; bump speech.version")
        verify_files(out)
        return out, True

    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    tok = BPETokenizer.load(tdir)
    targets = {"test": sp.test_chunks, "val": sp.val_chunks}
    stores = {s: SplitStore(out, s, codec.codebooks) for s in SPLITS}
    stats = _Stats(out / STATS, fingerprint, stores)
    start = _clock()
    max_words = max(1, int(sp.max_seconds * WORDS_PER_SECOND))

    def out_of_time() -> bool:
        return max_seconds_this_session is not None and _clock() - start >= max_seconds_this_session

    def done(split: str) -> bool:
        st = stores[split]
        return st.seconds >= sp.hours * 3600 if split == "train" else len(st.records) >= targets[split]

    def process(split: str, cid: str, text: str) -> None:
        syn = tts.synthesize(text)
        if len(syn.audio) / syn.sample_rate > sp.max_seconds:
            return stats.drop(cid, "too_long")
        ids = tok.encode(text)
        try:
            if syn.words is None:
                raise TimingMismatch("TTS returned no word timings")
            token_frames = token_start_frames(text, ids, tok, syn.words, codec.frame_rate)
        except TimingMismatch:
            return stats.drop(cid, "timing_mismatch")
        if _in_qa_sample(cid, sp.qa_fraction):
            hyp = asr.transcribe(syn.audio, syn.sample_rate).text
            if word_error_rate(text, hyp) > sp.qa_max_wer:
                return stats.drop(cid, "qa_wer")
        codes = codec.encode(syn.audio, syn.sample_rate)
        if len(codes) == 0:
            return stats.drop(cid, "empty_audio")
        store = stores[split]
        if split == "train" and len(store.records) < sp.keep_wavs:
            atomic_write_bytes(out / "samples" / f"{cid}.wav", _wav_bytes(syn.audio, syn.sample_rate))
        store.append({"id": cid, "text": text, "text_ids": ids, "token_frames": token_frames,
                      "seconds": len(codes) / codec.frame_rate}, codes)
        stats.save()

    def run_split(split: str) -> bool:
        if done(split):
            return True
        store = stores[split]
        for doc_i, story in enumerate(_read_docs(ddir / f"{split}.jsonl")):
            for chunk_i, text in enumerate(chunk_story(story, max_words)):
                cid = f"{split}-{doc_i}-{chunk_i}"
                if cid in store.ids or cid in stats.dropped:
                    continue
                if out_of_time():
                    return False
                process(split, cid, text)
                if (len(store.records) + 1) % 50 == 0:
                    say(f"{split}: {len(store.records)} chunks, {store.seconds / 3600:.3f} h")
                if done(split):
                    return True
        return True  # source exhausted: take what exists

    try:
        stats.save()
        complete = True
        for split in SPLITS:
            if not run_split(split):
                complete = False
                break
            say(f"{split}: complete ({len(stores[split].records)} chunks, "
                f"{stores[split].seconds / 3600:.3f} h)")
    finally:
        for st in stores.values():
            st.close()
    if not complete:
        return out, False

    files = sorted([f"{s}_{k}" for s in SPLITS for k in ("codes.bin", "index.jsonl")] + [STATS]
                   + [f"samples/{p.name}" for p in (out / "samples").glob("*.wav")])
    extra = {
        "voice": sp.voice, "tts": sp.tts, "codec": sp.codec, "asr": sp.asr,
        "frame_rate": codec.frame_rate, "codebooks": codec.codebooks,
        "splits": {s: {"chunks": len(stores[s].records), "frames": stores[s].frames,
                       "hours": stores[s].frames / codec.frame_rate / 3600} for s in SPLITS},
        "drops": stats.counts(),
    }
    write_manifest(out, "speech", sp.name, sp.version, files, inputs, extra=extra,
                   info={"build_seconds_last_session": round(_clock() - start, 1)})
    return out, True
