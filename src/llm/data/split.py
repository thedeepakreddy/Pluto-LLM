"""Deterministic train/val/test assignment by content hash (no randomness)."""
from __future__ import annotations

import hashlib

SPLITS = ("train", "val", "test")


def assign_split(text: str, ratios: dict[str, float]) -> str:
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % 1000
    cum = 0.0
    for name in SPLITS:
        cum += ratios.get(name, 0.0)
        if h < round(cum * 1000):
            return name
    return next(n for n in reversed(SPLITS) if ratios.get(n, 0.0) > 0)
