"""Deterministic, CPU-only fakes of TTS / codec / ASR for tests and CLI end-to-end runs.

Not real speech: ``FakeTTS`` emits one fixed-length tone per word and remembers the audio it
produced so ``FakeASR`` can "recognise" it exactly; any other audio (including
``FakeCodec`` round-trips) transcribes to the empty string.
"""

from __future__ import annotations

import hashlib

import numpy as np

from llm.speech.interfaces import Synthesis, Transcript, WordTiming

WORD_SECONDS = 0.4
_SAMPLE_RATE = 24000
_FRAME = 1920  # samples per codec frame (24 kHz / 12.5 fps)
_CODEBOOK_SIZE = 2048

# sha256(audio bytes) -> text, filled by FakeTTS, read by FakeASR.
_REGISTRY: dict[str, str] = {}


def _digest(audio: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def _word_freq(word: str) -> float:
    h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
    return 200.0 + (h % 2000)


class FakeTTS:
    def synthesize(self, text: str) -> Synthesis:
        n = int(round(WORD_SECONDS * _SAMPLE_RATE))
        t = np.arange(n) / _SAMPLE_RATE
        words = text.split()
        chunks, timings = [], []
        for i, w in enumerate(words):
            chunks.append((0.5 * np.sin(2 * np.pi * _word_freq(w) * t)).astype(np.float32))
            timings.append(WordTiming(w, i * WORD_SECONDS, (i + 1) * WORD_SECONDS))
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        _REGISTRY[_digest(audio)] = text
        return Synthesis(audio=audio, sample_rate=_SAMPLE_RATE, words=timings)


class FakeCodec:
    sample_rate = _SAMPLE_RATE
    frame_rate = 12.5
    codebooks = 8

    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != self.sample_rate:
            raise ValueError(f"FakeCodec expects {self.sample_rate} Hz audio, got {sample_rate}")
        audio = np.asarray(audio, dtype=np.float32)
        frames = len(audio) // _FRAME
        means = np.abs(audio[: frames * _FRAME].reshape(frames, _FRAME).astype(np.float64).mean(axis=1))
        base = (means * 1e6).astype(np.int64)
        ks = np.arange(self.codebooks, dtype=np.int64)
        return (base[:, None] + ks[None, :]) % _CODEBOOK_SIZE

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes)
        const = (codes[:, 0].astype(np.float64) / 1e6).astype(np.float32)
        return np.repeat(const, _FRAME)


class FakeASR:
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        text = _REGISTRY.get(_digest(audio), "")
        words = [
            WordTiming(w, i * WORD_SECONDS, (i + 1) * WORD_SECONDS)
            for i, w in enumerate(text.split())
        ]
        return Transcript(text=text, words=words)
