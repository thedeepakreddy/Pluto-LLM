"""Reading the built speech shards and producing deterministic, resumable training batches.

``SpeechShards`` is a read-only view of one split written by ``llm.speech.build`` (memory-mapped
codes plus the chunk index). ``next_speech_batch`` walks a seeded per-epoch permutation of the
chunks, so the data position is just ``(epoch, index)`` (the M1 ``DataPosition``) and resuming
from a checkpointed position reproduces the exact same batches.

Text-only batches: every ``audio.text_every``-th step (``(step + 1) % text_every == 0``) is built
from the M1 text corpus instead, so the model keeps its language ability. Each example is
``[BOS] + chunk[:seq_len]`` where ``chunk`` comes from ``text_chunks`` (a ``TokenChunks`` of
``seq_len + 1`` tokens: the trailing target token is dropped because the layouts already shift
targets via the masks). Callers must therefore build ``TokenChunks(..., seq_len=context_length - 1)``
so BOS + tokens is exactly ``model.context_length`` positions. The speech position is left
untouched by text batches, and the chunk choice depends only on ``(seed, step)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from llm.config import AudioConfig
from llm.shards import DataPosition, TokenChunks, epoch_order
from llm.speech.layout import (
    EMPTY,
    Example,
    SpecialIds,
    build_aligned,
    build_text_only,
    build_text_then_audio,
)
from llm.speech.store import BYTES_PER_CODE


class SpeechShards:
    """One split of a built speech dataset: ``ds[i] -> (text_ids, token_frames, codes[N, K])``."""

    def __init__(self, dir: Path, split: str, codebooks: int = 8):
        dir = Path(dir)
        self.split, self.codebooks = split, codebooks
        index = dir / f"{split}_index.jsonl"
        self._records = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
        frames = sum(r["frames"] for r in self._records)
        codes_path = dir / f"{split}_codes.bin"
        size = codes_path.stat().st_size if codes_path.exists() else 0
        if size != frames * codebooks * BYTES_PER_CODE:
            raise ValueError(f"{codes_path}: {size} bytes, index expects {frames} frames x {codebooks} codebooks")
        self._codes = (np.memmap(codes_path, dtype="<u2", mode="r", shape=(frames, codebooks))
                       if frames else np.zeros((0, codebooks), dtype="<u2"))
        self.seconds_total = float(sum(r["seconds"] for r in self._records))
        self._order_key: tuple[int, int] | None = None
        self._order: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, i: int) -> tuple[list[int], list[int], np.ndarray]:
        if not 0 <= i < len(self):
            raise IndexError(i)
        r = self._records[i]
        codes = np.asarray(self._codes[r["offset"]: r["offset"] + r["frames"]])
        return list(r["text_ids"]), list(r["token_frames"]), codes

    def order(self, seed: int, epoch: int) -> np.ndarray:
        if self._order_key != (seed, epoch):
            self._order, self._order_key = epoch_order(len(self), seed, epoch), (seed, epoch)
        return self._order


@dataclass
class SpeechBatch:
    text: torch.Tensor  # long [B, T]
    codes: torch.Tensor  # long [B, T, K]
    text_mask: torch.Tensor  # bool [B, T]
    code_mask: torch.Tensor  # bool [B, T, K]

    def to(self, device) -> SpeechBatch:
        return SpeechBatch(self.text.to(device), self.codes.to(device),
                           self.text_mask.to(device), self.code_mask.to(device))


def collate(examples: list[Example], sp: SpecialIds) -> SpeechBatch:
    """Right-pad to the longest example: text PAD_TEXT, codes EMPTY, masks False."""
    b, t = len(examples), max(len(e.text) for e in examples)
    k = examples[0].codes.shape[1]
    text = np.full((b, t), sp.pad_text, dtype=np.int64)
    codes = np.full((b, t, k), EMPTY, dtype=np.int64)
    text_mask = np.zeros((b, t), dtype=bool)
    code_mask = np.zeros((b, t, k), dtype=bool)
    for i, e in enumerate(examples):
        n = len(e.text)
        text[i, :n], codes[i, :n] = e.text, e.codes
        text_mask[i, :n], code_mask[i, :n] = e.text_mask, e.code_mask
    return SpeechBatch(torch.from_numpy(text), torch.from_numpy(codes),
                       torch.from_numpy(text_mask), torch.from_numpy(code_mask))


def _speech_example(ds: SpeechShards, i: int, audio: AudioConfig, sp: SpecialIds) -> Example:
    text_ids, token_frames, codes = ds[i]
    if audio.layout == "aligned":
        return build_aligned(text_ids, token_frames, codes, sp, audio.acoustic_delay)
    return build_text_then_audio(text_ids, codes, sp, audio.acoustic_delay)


def _text_batch(text_chunks: TokenChunks, batch_size: int, seed: int, audio: AudioConfig,
                sp: SpecialIds, step: int) -> SpeechBatch:
    order = epoch_order(len(text_chunks), seed, step)
    examples = []
    for j in range(batch_size):
        chunk = text_chunks[int(order[j % len(order)])]
        examples.append(build_text_only(chunk[: text_chunks.seq_len], sp, audio.codebooks))
    return collate(examples, sp)


def next_speech_batch(ds: SpeechShards, pos: DataPosition, batch_size: int, seed: int,
                      audio: AudioConfig, sp: SpecialIds, step: int,
                      text_chunks: TokenChunks | None) -> tuple[SpeechBatch, DataPosition]:
    if text_chunks is not None and (step + 1) % audio.text_every == 0:
        return _text_batch(text_chunks, batch_size, seed, audio, sp, step), pos
    epoch, index = pos.epoch, pos.index
    examples = []
    while len(examples) < batch_size:
        if index >= len(ds):
            epoch, index = epoch + 1, 0
        examples.append(_speech_example(ds, int(ds.order(seed, epoch)[index]), audio, sp))
        index += 1
    if index >= len(ds):
        epoch, index = epoch + 1, 0
    return collate(examples, sp), DataPosition(epoch, index)
