"""Word / character error rates (Levenshtein distance over normalised text)."""

from __future__ import annotations

import re

_STRIP = re.compile(r"[^\w\s']", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    """Lowercase, drop punctuation except apostrophes, split on whitespace."""
    return _STRIP.sub(" ", text.lower()).split()


def _edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def _rate(ref: list, hyp: list) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def word_error_rate(ref: str, hyp: str) -> float:
    return _rate(normalize_words(ref), normalize_words(hyp))


def char_error_rate(ref: str, hyp: str) -> float:
    """Character-level rate on the normalised text (words re-joined by single spaces)."""
    return _rate(list(" ".join(normalize_words(ref))), list(" ".join(normalize_words(hyp))))
