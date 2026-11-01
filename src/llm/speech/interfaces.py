"""Protocols for the speech components and config-driven factories.

Real implementations (Kokoro TTS, Mimi codec, Whisper ASR) live in ``llm.speech.adapters``
and are imported lazily so that CPU/CI runs never pull in heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from llm.config import SpeechConfig


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_s: float
    end_s: float


@dataclass
class Synthesis:
    """Mono float32 audio at ``sample_rate``; ``words`` is None if the TTS gives no timings."""

    audio: np.ndarray
    sample_rate: int
    words: list[WordTiming] | None


@dataclass
class Transcript:
    text: str
    words: list[WordTiming]


class TTS(Protocol):
    def synthesize(self, text: str) -> Synthesis: ...


class Codec(Protocol):
    sample_rate: int
    frame_rate: float
    codebooks: int

    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return int64 codes of shape (frames, codebooks)."""
        ...

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Return mono float32 audio at ``sample_rate`` for (frames, codebooks) codes."""
        ...


class ASR(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript: ...


def _unknown(kind: str, name: str, options: str) -> ValueError:
    return ValueError(f"unknown {kind} {name!r}; expected one of: {options}")


def make_tts(cfg: SpeechConfig) -> TTS:
    if cfg.tts == "fake":
        from llm.speech.fakes import FakeTTS

        return FakeTTS()
    if cfg.tts == "kokoro":
        from llm.speech.adapters import KokoroTTS

        return KokoroTTS(voice=cfg.voice)
    raise _unknown("tts", cfg.tts, "fake, kokoro")


def make_codec(cfg: SpeechConfig) -> Codec:
    if cfg.codec == "fake":
        from llm.speech.fakes import FakeCodec

        return FakeCodec()
    if cfg.codec == "mimi":
        from llm.speech.adapters import MimiCodec

        return MimiCodec()
    raise _unknown("codec", cfg.codec, "fake, mimi")


def make_asr(cfg: SpeechConfig) -> ASR:
    if cfg.asr == "fake":
        from llm.speech.fakes import FakeASR

        return FakeASR()
    if cfg.asr == "whisper":
        from llm.speech.adapters import WhisperASR

        return WhisperASR(model=cfg.asr_model)
    raise _unknown("asr", cfg.asr, "fake, whisper")
