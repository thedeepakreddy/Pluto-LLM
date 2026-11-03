"""Benchmark test items (validation splits) for decontamination and multiple-choice evaluation.

Each loader yields the texts a model could have memorised: the question/context and every answer.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator


def _hf(path: str, name: str | None, split: str):
    from datasets import load_dataset  # heavy import, only when needed

    return load_dataset(path, name, split=split)


def _hellaswag() -> Iterator[str]:
    for r in _hf("Rowan/hellaswag", None, "validation"):
        yield r["ctx"]
        yield from (r["ctx"] + " " + e for e in r["endings"])


def _arc(name: str) -> Callable[[], Iterator[str]]:
    def load() -> Iterator[str]:
        for r in _hf("allenai/ai2_arc", name, "test"):
            yield r["question"]
            yield from (r["question"] + " " + c for c in r["choices"]["text"])
    return load


def _piqa() -> Iterator[str]:
    for r in _hf("baber/piqa", None, "validation"):  # parquet copy (the original uses a loading script)
        yield r["goal"]
        yield r["goal"] + " " + r["sol1"]
        yield r["goal"] + " " + r["sol2"]


def _gsm8k() -> Iterator[str]:
    for r in _hf("openai/gsm8k", "main", "test"):
        yield r["question"]
        yield r["answer"]


BENCHMARKS: dict[str, Callable[[], Iterator[str]]] = {
    "hellaswag": _hellaswag, "arc_easy": _arc("ARC-Easy"), "arc_challenge": _arc("ARC-Challenge"),
    "piqa": _piqa, "gsm8k": _gsm8k,
}


def benchmark_texts(names) -> Iterator[str]:
    for n in names:
        if n not in BENCHMARKS:
            raise ValueError(f"unknown benchmark {n!r}; known: {sorted(BENCHMARKS)}")
        yield from BENCHMARKS[n]()
