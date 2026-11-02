"""Thin real-model adapters: Kokoro TTS, Mimi codec, Whisper ASR (all off-the-shelf).

Heavy libraries (kokoro, torch, transformers) are imported inside constructors/methods only, so
importing this module is free and CI never downloads models. ``describe()`` returns the pinned
model ids and library versions that the builder records in the speech manifest.
"""

from __future__ import annotations

import re

import numpy as np

from llm.speech.interfaces import Synthesis, Transcript, WordTiming
from llm.speech.wavio import resample

_WORDISH = re.compile(r"\w")


def _lib_version(module) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


class KokoroTTS:
    """Kokoro-82M via ``kokoro.KPipeline`` (British English, 24 kHz)."""

    sample_rate = 24000

    def __init__(self, voice: str):
        import kokoro

        self._kokoro = kokoro
        self.voice = voice
        self._pipeline = kokoro.KPipeline(lang_code="b")

    def synthesize(self, text: str) -> Synthesis:
        segments: list[np.ndarray] = []
        words: list[WordTiming] = []
        have_times = True
        offset = 0.0
        for res in self._pipeline(text, voice=self.voice):
            if res.audio is None:
                continue
            audio = _to_numpy(res.audio)
            tokens = getattr(res, "tokens", None)
            if not tokens:
                have_times = False
            else:
                for t in tokens:
                    if not _WORDISH.search(t.text or ""):
                        continue  # punctuation / whitespace tokens carry no spoken word
                    if t.start_ts is None or t.end_ts is None:
                        have_times = False
                        break
                    words.append(WordTiming(t.text.strip(), offset + float(t.start_ts),
                                            offset + float(t.end_ts)))
            segments.append(audio)
            offset += len(audio) / self.sample_rate
        audio = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
        return Synthesis(audio=audio, sample_rate=self.sample_rate,
                         words=words if have_times and words else None)

    def describe(self) -> dict:
        return {"model": "hexgrad/Kokoro-82M", "voice": self.voice, "lang_code": "b",
                "kokoro": _lib_version(self._kokoro)}


class MimiCodec:
    """Kyutai Mimi (24 kHz, 12.5 frames/s, first 8 of 32 codebooks, 2048 entries each)."""

    sample_rate = 24000
    frame_rate = 12.5
    codebooks = 8
    model_id = "kyutai/mimi"

    def __init__(self):
        import torch
        import transformers

        self._torch, self._tf = torch, transformers
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = transformers.MimiModel.from_pretrained(self.model_id).to(self._device).eval()

    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != self.sample_rate:
            audio = resample(audio, sample_rate, self.sample_rate)
        x = self._torch.as_tensor(np.asarray(audio, dtype=np.float32)).reshape(1, 1, -1).to(self._device)
        with self._torch.no_grad():
            codes = self._model.encode(x, num_quantizers=self.codebooks).audio_codes  # [1, K, frames]
        return codes[0].transpose(0, 1).cpu().numpy().astype(np.int64)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        c = self._torch.as_tensor(np.asarray(codes, dtype=np.int64)).T.unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            audio = self._model.decode(c).audio_values  # [1, 1, T]
        return audio.reshape(-1).cpu().numpy().astype(np.float32)

    def describe(self) -> dict:
        return {"model": self.model_id, "codebooks": self.codebooks,
                "transformers": _lib_version(self._tf)}


class WhisperASR:
    """Whisper via the transformers ASR pipeline with word-level timestamps."""

    def __init__(self, model: str):
        import torch
        import transformers

        self.model = model
        self._tf = transformers
        kwargs = {"device": 0} if torch.cuda.is_available() else {}
        self._pipe = transformers.pipeline("automatic-speech-recognition", model=model,
                                           return_timestamps="word", **kwargs)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        raw = resample(audio, sample_rate, 16000)
        out = self._pipe({"raw": raw, "sampling_rate": 16000})
        words: list[WordTiming] = []
        for ch in out.get("chunks") or []:
            w = ch["text"].strip()
            start, end = ch["timestamp"]
            if not w or start is None:
                continue
            words.append(WordTiming(w, float(start), float(end if end is not None else start)))
        return Transcript(text=out["text"].strip(), words=words)

    def describe(self) -> dict:
        return {"model": self.model, "transformers": _lib_version(self._tf)}
