"""Raw document sources. Each source yields plain text documents."""
from __future__ import annotations

import json
from collections.abc import Iterator
from itertools import islice

from llm.config import SourceConfig


def _local_jsonl(src: SourceConfig) -> Iterator[str]:
    with open(src.path, encoding="utf-8", errors="surrogatepass") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)[src.text_field]


def _hf(src: SourceConfig) -> Iterator[str]:
    from datasets import load_dataset  # heavy import, only when needed

    for row in load_dataset(src.path, split=src.split, streaming=True):
        yield row[src.text_field]


def iter_source_docs(src: SourceConfig) -> Iterator[str]:
    it = _local_jsonl(src) if src.type == "local_jsonl" else _hf(src)
    return islice(it, src.max_docs) if src.max_docs is not None else it
