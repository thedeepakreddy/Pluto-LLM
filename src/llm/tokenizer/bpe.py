"""From-scratch byte-level BPE tokenizer.

ID layout: 0..255 are raw bytes, 256..vocab_size-2 are merges in creation
order (a merge's id is also its priority), and vocab_size-1 is <|endoftext|>.
Because every byte has an id, no input can produce an unknown token.
"""
from __future__ import annotations

import heapq
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import regex

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"
_CACHE_LIMIT = 500_000


class TokenizerTrainingError(Exception):
    """The training corpus cannot produce the requested vocabulary size."""


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]], vocab_size: int, pattern: str = GPT2_SPLIT_PATTERN):
        if len(merges) != vocab_size - 257:
            raise ValueError(f"{len(merges)} merges do not match vocab_size {vocab_size}")
        self._merges = [tuple(m) for m in merges]
        self._vocab_size = vocab_size
        self._pattern_str = pattern
        self._pattern = regex.compile(pattern)
        self._ranks = {pair: 256 + i for i, pair in enumerate(self._merges)}
        self._bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self._merges:
            self._bytes.append(self._bytes[a] + self._bytes[b])
        self._bytes.append(b"")  # <|endoftext|> decodes to nothing
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------ properties
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eot_id(self) -> int:
        return self._vocab_size - 1

    # ------------------------------------------------------------ training
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int,
              pattern: str = GPT2_SPLIT_PATTERN) -> BPETokenizer:
        compiled = regex.compile(pattern)
        chunk_counts: Counter[str] = Counter()
        for t in texts:
            chunk_counts.update(compiled.findall(t))
        words = [list(c.encode("utf-8")) for c in chunk_counts]
        freqs = list(chunk_counts.values())

        pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for idx, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(idx)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        for new_id in range(256, vocab_size - 1):
            best = None
            while heap:
                neg, p = heapq.heappop(heap)
                if pair_counts.get(p, 0) == -neg and -neg > 0:
                    best = (p, -neg)
                    break
            if best is None or best[1] < 2:
                raise TokenizerTrainingError(
                    f"corpus too small: only {len(merges)} merges possible, need {vocab_size - 257}")
            pair = best[0]
            merges.append(pair)
            touched: set[tuple[int, int]] = set()
            for idx in where.pop(pair):
                w, f = words[idx], freqs[idx]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    touched.add(p)
                merged = _merge(w, pair, new_id)
                words[idx] = merged
                for p in zip(merged, merged[1:]):
                    pair_counts[p] += f
                    where[p].add(idx)
                    touched.add(p)
            pair_counts.pop(pair, None)
            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
        return cls(merges, vocab_size, pattern)

    # ------------------------------------------------------------ encode/decode
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best = min(zip(ids, ids[1:]), key=lambda p: self._ranks.get(p, float("inf")))
            rank = self._ranks.get(best)
            if rank is None:
                break
            ids = _merge(ids, best, rank)
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self._pattern.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._bytes[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------ persistence
    def save(self, dir: Path) -> None:
        from llm.manifest import atomic_write_json

        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(dir / "tokenizer.json", {
            "vocab_size": self._vocab_size, "pattern": self._pattern_str,
            "special": {EOT: self.eot_id}, "merges": [list(m) for m in self._merges]})

    @classmethod
    def load(cls, dir: Path) -> BPETokenizer:
        d = json.loads((Path(dir) / "tokenizer.json").read_text())
        return cls([tuple(m) for m in d["merges"]], d["vocab_size"], d["pattern"])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i, n = [], 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def tokenizer_dir(cfg) -> Path:
    return Path(cfg.tokenizer.root) / f"{cfg.tokenizer.name}-v{cfg.tokenizer.version}"


def train_tokenizer(cfg) -> Path:
    """Train on up to `train_sample_bytes` of the train split and save with a manifest."""
    from llm.data.prepare import data_dir
    from llm.manifest import manifest_checksum, write_manifest

    ddir = data_dir(cfg)
    sample: list[str] = []
    total = 0
    with open(ddir / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            sample.append(text)
            total += len(text.encode("utf-8"))
            if total >= cfg.tokenizer.train_sample_bytes:
                break
    t0 = time.time()
    tok = BPETokenizer.train(sample, cfg.tokenizer.vocab_size)
    seconds = time.time() - t0
    probe = "\n".join(sample)[:1_000_000]
    out = tokenizer_dir(cfg)
    tok.save(out)
    write_manifest(out, kind="tokenizer", name=cfg.tokenizer.name, version=cfg.tokenizer.version,
                   files=["tokenizer.json"], inputs={"data": manifest_checksum(ddir)},
                   extra={"vocab_size": tok.vocab_size, "sample_bytes": total,
                          "bytes_per_token": len(probe.encode()) / max(1, len(tok.encode(probe)))},
                   info={"train_seconds": round(seconds, 2)})
    return out
