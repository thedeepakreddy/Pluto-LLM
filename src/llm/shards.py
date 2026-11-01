"""Token shards: flat binary arrays of token ids plus a manifest.

Each split's documents are encoded and written back-to-back with <|endoftext|>
after every document. Training views a split as fixed-length chunks; the order
of chunks in each epoch is a permutation fixed by (seed, epoch), so the data
position is just (epoch, index) and resuming is exact.
"""
from __future__ import annotations

import bisect
import json
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from llm.data.prepare import data_dir
from llm.data.split import SPLITS
from llm.manifest import (ProvenanceError, manifest_checksum, read_manifest, verify_files,
                          verify_input, write_manifest)
from llm.tokenizer.bpe import BPETokenizer, tokenizer_dir


def shard_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16) if vocab_size <= 65536 else np.dtype(np.uint32)


def shard_dir(cfg) -> Path:
    return Path(cfg.shards.root) / f"{cfg.shards.name}-v{cfg.shards.version}"


def write_shards(cfg) -> Path:
    ddir, tdir, out = data_dir(cfg), tokenizer_dir(cfg), shard_dir(cfg)
    verify_files(ddir)
    verify_files(tdir)
    verify_input(read_manifest(tdir), "data", ddir)
    tok = BPETokenizer.load(tdir)
    if tok.vocab_size != cfg.tokenizer.vocab_size:
        raise ProvenanceError(f"tokenizer in {tdir} has vocab {tok.vocab_size}, "
                              f"config says {cfg.tokenizer.vocab_size}")
    dt = shard_dtype(tok.vocab_size)
    code = "H" if dt == np.uint16 else "I"
    size = cfg.shards.shard_size_tokens
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.bin"):
        old.unlink()

    splits, files = {}, []
    for split in SPLITS:
        names: list[str] = []
        buf = array(code)
        n_tokens = n_bytes = 0

        def flush(final: bool) -> None:
            nonlocal buf
            while len(buf) >= size or (final and len(buf) > 0):
                name = f"{split}_{len(names):03d}.bin"
                np.frombuffer(buf[:size], dtype=dt).tofile(out / name)
                names.append(name)
                buf = buf[size:]

        with open(ddir / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                ids = tok.encode(text)
                ids.append(tok.eot_id)
                buf.extend(ids)
                n_tokens += len(ids)
                n_bytes += len(text.encode("utf-8"))
                if len(buf) >= size:
                    flush(final=False)
        flush(final=True)
        splits[split] = {"tokens": n_tokens, "bytes": n_bytes, "files": names}
        files += names

    write_manifest(out, kind="shards", name=cfg.shards.name, version=cfg.shards.version, files=files,
                   inputs={"tokenizer": manifest_checksum(tdir), "data": manifest_checksum(ddir)},
                   extra={"dtype": dt.name, "vocab_size": tok.vocab_size, "splits": splits})
    return out


def verify_shard_provenance(cfg) -> None:
    """Shard files intact, and built from the tokenizer and data currently on disk."""
    sdir = shard_dir(cfg)
    verify_files(sdir)
    m = read_manifest(sdir)
    verify_input(m, "tokenizer", tokenizer_dir(cfg))
    verify_input(m, "data", data_dir(cfg))
    if m["extra"]["vocab_size"] != cfg.tokenizer.vocab_size:
        raise ProvenanceError(f"shards were built with vocab {m['extra']['vocab_size']}, "
                              f"config says {cfg.tokenizer.vocab_size} (tokenizer mismatch)")


class TokenChunks:
    """A split viewed as fixed-length chunks of seq_len + 1 tokens (inputs + shifted targets)."""

    def __init__(self, shard_dir: Path, split: str, seq_len: int):
        shard_dir = Path(shard_dir)
        info = read_manifest(shard_dir)["extra"]
        s = info["splits"][split]
        dt = np.dtype(info["dtype"])
        self._arrays = [np.memmap(shard_dir / f, dtype=dt, mode="r") for f in s["files"]]
        self._starts = [0]
        for a in self._arrays:
            self._starts.append(self._starts[-1] + len(a))
        self.n_tokens = self._starts[-1]
        self.seq_len = seq_len
        self.split = split
        self.bytes_per_token = s["bytes"] / s["tokens"] if s["tokens"] else 0.0
        if len(self) == 0:
            raise ValueError(f"split '{split}' has {self.n_tokens} tokens, fewer than one sequence "
                             f"of {seq_len + 1}; use a bigger split or evaluate on another split")
        self._order_key: tuple[int, int] | None = None
        self._order: np.ndarray | None = None

    def __len__(self) -> int:
        return max(0, (self.n_tokens - 1) // self.seq_len)

    def _range(self, start: int, end: int) -> np.ndarray:
        parts = []
        i = bisect.bisect_right(self._starts, start) - 1
        while start < end:
            a, a0 = self._arrays[i], self._starts[i]
            take = min(end, a0 + len(a))
            parts.append(np.asarray(a[start - a0: take - a0]))
            start, i = take, i + 1
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def __getitem__(self, i: int) -> np.ndarray:
        if not 0 <= i < len(self):
            raise IndexError(i)
        s = i * self.seq_len
        return self._range(s, s + self.seq_len + 1)

    def order(self, seed: int, epoch: int) -> np.ndarray:
        if self._order_key != (seed, epoch):
            self._order, self._order_key = epoch_order(len(self), seed, epoch), (seed, epoch)
        return self._order


@dataclass(frozen=True)
class DataPosition:
    epoch: int = 0
    index: int = 0


def epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng([seed, epoch]).permutation(n)


def next_batch(chunks: TokenChunks, pos: DataPosition, batch_size: int,
               seed: int) -> tuple[torch.Tensor, torch.Tensor, DataPosition]:
    rows = []
    epoch, index = pos.epoch, pos.index
    while len(rows) < batch_size:
        if index >= len(chunks):
            epoch, index = epoch + 1, 0
        rows.append(chunks[int(chunks.order(seed, epoch)[index])].astype(np.int64))
        index += 1
    if index >= len(chunks):
        epoch, index = epoch + 1, 0
    batch = torch.from_numpy(np.stack(rows))
    return batch[:, :-1], batch[:, 1:], DataPosition(epoch, index)
