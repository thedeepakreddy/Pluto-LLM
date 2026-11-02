"""Minimal 16-bit PCM mono WAV reading/writing with the stdlib (no soundfile dependency)."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (mono float32 audio, sample_rate). Only 16-bit PCM is supported."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM wav is supported")
        ch, sr = w.getnchannels(), w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    x = raw.astype(np.float32) / 32768.0
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), sr


def to_mono_float32(x: np.ndarray) -> np.ndarray:
    """Normalise int PCM to [-1, 1] (divide by 32768) and average channels (samples, channels)."""
    x = np.asarray(x)
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """FFT-based resampling (like scipy.signal.resample); mono float32; identity if rates match."""
    x = to_mono_float32(audio)
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = max(1, round(len(x) * sr_out / sr_in))
    spec = np.fft.rfft(x)
    out = np.zeros(n_out // 2 + 1, dtype=spec.dtype)
    m = min(len(spec), len(out))
    out[:m] = spec[:m]
    if n_out < len(x) and n_out % 2 == 0:
        out[m - 1] = out[m - 1].real * 2  # fold the Nyquist bin when downsampling
    y = np.fft.irfft(out, n=n_out) * (n_out / len(x))
    return y.astype(np.float32)
