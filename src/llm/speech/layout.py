"""Sequence layouts that interleave text tokens and audio codec frames (M2).

Every layout yields an :class:`Example` with two parallel streams per position ``t``:
``text[t]`` (one text-vocab id, including the four specials) and ``codes[t, :]`` (one id per
codebook, ``EMPTY`` where the position carries no audio for that codebook). The masks mark
prediction *targets*: ``text_mask[t]`` means ``text[t]`` is predicted (from position ``t-1``),
``code_mask[t, k]`` means ``codes[t, k]`` is predicted. Position 0 (BOS) is never a target.

Layouts
-------
* ``build_aligned`` - the main time-aligned "inner monologue" layout: text tokens sit on the
  frame positions, a little ahead of the audio of the word they belong to.
* ``build_text_then_audio`` - the baseline: all text, an AUDIO marker, then all frames.
* ``build_text_only`` - text-only examples (no audio, no code targets).

Acoustic delay: codebook 0 of frame ``f`` and codebooks ``1..K-1`` of frame ``f`` are emitted
``delay`` positions apart, so the later codebooks can condition on codebook 0 of the same frame
(and, with delay 1, on one more step of context). Hence a stream of ``N`` frames occupies
``N + delay`` frame positions.

Everything here is pure numpy and exactly invertible by ``extract_codes`` / ``extract_text``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMPTY = 2048  # "no code here" id; real codec codes are in [0, 2048)
CODEBOOK_SIZE = 2048


@dataclass(frozen=True)
class SpecialIds:
    bos: int
    audio: int
    end_audio: int
    pad_text: int

    def as_set(self) -> frozenset[int]:
        return frozenset((self.bos, self.audio, self.end_audio, self.pad_text))


def special_ids(vocab_size: int) -> SpecialIds:
    """Specials are appended after the text vocabulary: BOS=V, AUDIO=V+1, END_AUDIO=V+2, PAD_TEXT=V+3."""
    v = int(vocab_size)
    return SpecialIds(bos=v, audio=v + 1, end_audio=v + 2, pad_text=v + 3)


@dataclass
class Example:
    text: np.ndarray  # int64 (T,)
    codes: np.ndarray  # int64 (T, K)
    text_mask: np.ndarray  # bool (T,)  True where text[t] is a prediction target
    code_mask: np.ndarray  # bool (T, K) True where codes[t, k] is a prediction target


# ---------------------------------------------------------------- helpers


def _check_delay(delay: int) -> None:
    if delay < 0:
        raise ValueError(f"delay must be >= 0, got {delay}")


def _check_text(text_ids, sp: SpecialIds) -> np.ndarray:
    arr = np.asarray(list(text_ids), dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= sp.bos):
        raise ValueError(f"text ids must be in [0, {sp.bos}); got range [{arr.min()}, {arr.max()}]")
    return arr


def _check_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"codes must be (N, K), got shape {codes.shape}")
    if codes.size and (codes.min() < 0 or codes.max() >= CODEBOOK_SIZE):
        raise ValueError(f"codes must be in [0, {CODEBOOK_SIZE}); EMPTY is not a valid code")
    return codes.astype(np.int64, copy=False)


def _scatter_frames(out: np.ndarray, start: int, codes: np.ndarray, delay: int) -> None:
    """Write ``codes`` (N, K) into ``out`` so frame f's codebook 0 is at row start+f and codebooks
    1..K-1 at row start+f+delay."""
    n = codes.shape[0]
    if n == 0:
        return
    out[start : start + n, 0] = codes[:, 0]
    if codes.shape[1] > 1:
        out[start + delay : start + delay + n, 1:] = codes[:, 1:]


def _finish(text: np.ndarray, codes: np.ndarray) -> Example:
    text_mask = np.ones(len(text), dtype=bool)
    text_mask[0] = False
    code_mask = codes != EMPTY
    # Flush positions (codebook 0 EMPTY, delayed codebooks present): codebook-0 EMPTY is a
    # target meaning "stop audio", so the model learns when the audio ends.
    if codes.shape[1] > 1:
        code_mask[:, 0] |= (codes[:, 0] == EMPTY) & (codes[:, 1:] != EMPTY).any(axis=1)
    code_mask[0] = False  # BOS is never a target (and carries no codes anyway)
    return Example(text=text, codes=codes, text_mask=text_mask, code_mask=code_mask)


# ---------------------------------------------------------------- builders


def build_aligned(text_ids, token_frames, codes, sp: SpecialIds, delay: int) -> Example:
    """Time-aligned layout: ``[BOS] + frame positions (+ extra) + [END_AUDIO]``.

    Position mapping (BOS occupies position 0, so frame ``f`` is at position ``f + 1``):

    * frame position ``p`` (``p = 1 .. N + delay``) holds codebook 0 of frame ``p - 1`` and
      codebooks 1..K-1 of frame ``p - 1 - delay`` (EMPTY where that frame is out of range);
    * text token ``i`` goes to position ``s_i = max(token_frames[i], s_{i-1} + 1, 1)``. A word that
      starts at frame ``f`` has its audio at position ``f + 1``, so its first token sits at
      position ``f``: one frame of lead. Frame 0 is clamped to position 1. Crowded tokens spill
      forward one position at a time, keeping their order;
    * positions without a token hold PAD_TEXT; if tokens spill past the last frame position,
      extra positions with all codes EMPTY are appended; the final position is END_AUDIO
      (codes EMPTY).
    """
    _check_delay(delay)
    ids = _check_text(text_ids, sp)
    frames = [int(f) for f in token_frames]
    if len(frames) != len(ids):
        raise ValueError(f"text_ids ({len(ids)}) and token_frames ({len(frames)}) differ in length")
    if any(b < a for a, b in zip(frames, frames[1:])) or (frames and frames[0] < 0):
        raise ValueError("token_frames must be non-negative and non-decreasing")
    codes = _check_codes(codes)
    n, k = codes.shape

    slots, last = [], 0  # position 0 is BOS, so the first usable slot is 1
    for f in frames:
        s = max(f, last + 1)
        slots.append(s)
        last = s
    n_frame_pos = n + delay
    n_body = max(n_frame_pos, last)  # frame positions + extra positions
    total = 1 + n_body + 1

    text = np.full(total, sp.pad_text, dtype=np.int64)
    text[0] = sp.bos
    text[-1] = sp.end_audio
    if slots:
        text[np.asarray(slots)] = ids
    code_arr = np.full((total, k), EMPTY, dtype=np.int64)
    _scatter_frames(code_arr, 1, codes, delay)
    return _finish(text, code_arr)


def build_text_then_audio(text_ids, codes, sp: SpecialIds, delay: int) -> Example:
    """Baseline layout: ``[BOS] + text + [AUDIO] + (N + delay) frame positions + [END_AUDIO]``.

    Text and AUDIO positions have codes EMPTY; frame positions have text PAD_TEXT (still text
    targets: the model must learn to emit PAD while speaking, then END_AUDIO).
    """
    _check_delay(delay)
    ids = _check_text(text_ids, sp)
    codes = _check_codes(codes)
    n, k = codes.shape
    t = len(ids)
    audio_pos = 1 + t
    total = audio_pos + 1 + n + delay + 1

    text = np.full(total, sp.pad_text, dtype=np.int64)
    text[0] = sp.bos
    text[1:audio_pos] = ids
    text[audio_pos] = sp.audio
    text[-1] = sp.end_audio
    code_arr = np.full((total, k), EMPTY, dtype=np.int64)
    _scatter_frames(code_arr, audio_pos + 1, codes, delay)
    return _finish(text, code_arr)


def build_text_only(text_ids, sp: SpecialIds, num_codebooks: int = 8) -> Example:
    """``[BOS] + tokens``; codes all EMPTY, no code targets."""
    ids = _check_text(text_ids, sp)
    text = np.concatenate([[sp.bos], ids]).astype(np.int64)
    codes = np.full((len(text), num_codebooks), EMPTY, dtype=np.int64)
    return _finish(text, codes)


# ---------------------------------------------------------------- inverses


def extract_codes(ex: Example, delay: int) -> np.ndarray:
    """Inverse of the audio layouts: the original ``(N, K)`` codes (``(0, K)`` for text-only).

    Codebook 0 is non-EMPTY exactly on the N frame positions (one contiguous run), which gives
    both the start position and N; codebooks 1..K-1 are read ``delay`` positions later.
    """
    _check_delay(delay)
    k = ex.codes.shape[1]
    present = np.flatnonzero(ex.codes[:, 0] != EMPTY)
    n = len(present)
    if n == 0:
        return np.zeros((0, k), dtype=np.int64)
    start = int(present[0])
    if present[-1] - start + 1 != n:
        raise ValueError("codebook-0 frames are not a contiguous run")
    out = np.empty((n, k), dtype=np.int64)
    out[:, 0] = ex.codes[start : start + n, 0]
    if k > 1:
        out[:, 1:] = ex.codes[start + delay : start + delay + n, 1:]
    return out


def extract_text(ex: Example, sp: SpecialIds) -> list[int]:
    """Inverse for text: text ids in order, dropping BOS / AUDIO / END_AUDIO / PAD_TEXT."""
    specials = sp.as_set()
    return [int(x) for x in ex.text if int(x) not in specials]


def max_text_lag(ex: Example, token_frames, delay: int, sp: SpecialIds) -> int:
    """Max over text tokens of ``slot_position - (token_frames[i] + 1)`` in an aligned example.

    ``token_frames[i] + 1`` is the position of codebook 0 of the frame where token i's word
    starts, so <= 0 means the text appears at or before the audio of its word. ``delay`` only
    shifts codebooks 1..K-1 and does not change the result (kept for a uniform signature).
    ``sp`` is needed to tell text slots from PAD_TEXT and the other specials. Returns 0 when
    there are no tokens.
    """
    specials = sp.as_set()
    slots = [t for t, x in enumerate(ex.text) if int(x) not in specials]
    frames = [int(f) for f in token_frames]
    if len(slots) != len(frames):
        raise ValueError(f"{len(slots)} text slots but {len(frames)} token_frames")
    if not slots:
        return 0
    return max(s - (f + 1) for s, f in zip(slots, frames))
