"""Raw document sources. Each source yields plain text documents (or rows, for filtering)."""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice

from llm.config import SourceConfig


def _local_rows(src: SourceConfig) -> Iterator[dict]:
    with open(src.path, encoding="utf-8", errors="surrogatepass") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _hf_rows(src: SourceConfig, streaming: bool = True) -> Iterator[dict]:
    from datasets import load_dataset  # heavy import, only when needed

    yield from load_dataset(src.path, src.config_name, split=src.split, streaming=streaming)


def _rows(src: SourceConfig, streaming: bool = True) -> Iterator[dict]:
    return _local_rows(src) if src.type == "local_jsonl" else _hf_rows(src, streaming)


def iter_source_docs(src: SourceConfig) -> Iterator[str]:
    """Single-source path (M1/M2), unchanged behaviour: always streams HF datasets."""
    it = (row[src.text_field] for row in _rows(src))
    return islice(it, src.max_docs) if src.max_docs is not None else it


@dataclass
class SourceStats:
    name: str
    received: int = 0
    below_min_score: int = 0
    passed: int = 0  # handed on to cleaning
    kept: int = 0  # kept after cleaning and global dedup
    bytes: int = 0  # UTF-8 bytes received (counts toward max_bytes)


def iter_source(src: SourceConfig, stats: SourceStats) -> Iterator[str]:
    """Rows of one source up to max_docs / max_bytes, with the optional score filter."""
    for row in _rows(src, src.streaming):
        if src.max_docs is not None and stats.received >= src.max_docs:
            break
        if src.max_bytes is not None and stats.bytes >= src.max_bytes:
            break
        text = row[src.text_field]
        stats.received += 1
        stats.bytes += len(text.encode("utf-8", errors="surrogatepass"))
        if src.min_score_field is not None:
            score = row.get(src.min_score_field)
            if score is None or score < src.min_score:
                stats.below_min_score += 1
                continue
        stats.passed += 1
        yield text
