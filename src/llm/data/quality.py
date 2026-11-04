"""M3 data quality: normalised exact dedup, web-text heuristics, benchmark decontamination and
MinHash near-duplicate removal. Every drop is counted under a named reason (see QUALITY_REASONS).

The heuristics follow the published Gopher rules (Rae et al. 2021, "MassiveText" quality filter),
simplified; the English check is the Gopher stop-word rule, not a language-ID model.
"""
from __future__ import annotations

import hashlib
import re
import zlib
from collections.abc import Iterable

import numpy as np

QUALITY_REASONS = ("not_english", "low_quality", "repetitive", "contaminated", "near_duplicate")

_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = frozenset({"the", "be", "to", "of", "and", "that", "have", "with"})


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def fingerprint(text: str) -> bytes:
    """Exact-dedup key: case-folded, whitespace-collapsed text (the stored text is not changed)."""
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).digest()


def heuristic_reason(text: str) -> str | None:
    """None if the document passes, otherwise the drop reason. For prose, not for code."""
    w = words(text)
    if len(w) < 20 or len(_STOP.intersection(w)) < 2:
        return "not_english"  # too short to judge, or no common English function words
    mean_len = sum(map(len, w)) / len(w)
    if not 3 <= mean_len <= 10:
        return "low_quality"
    if (text.count("#") + text.count("...") + text.count("…")) / len(w) > 0.1:
        return "low_quality"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        if sum(ln.startswith(("•", "-", "*")) for ln in lines) / len(lines) > 0.9:
            return "low_quality"
        if sum(ln.endswith(("...", "…")) for ln in lines) / len(lines) > 0.3:
            return "low_quality"
        if len(lines) >= 4 and 1 - len(set(lines)) / len(lines) > 0.3:
            return "repetitive"
    if len(w) >= 50:
        grams = list(zip(w, w[1:]))
        top = max(_counts(grams).values())
        if top * 2 / len(w) > 0.2:
            return "repetitive"
    return None


def _counts(xs) -> dict:
    c: dict = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def _h64(s: str) -> int:
    """Deterministic 64-bit hash. 32-bit CRC collides far too often for millions of benchmark
    n-grams (~35% false 'contaminated' flags per 1,000-word page); two seeded CRC32s are not
    independent (CRC is affine in its start value), so BLAKE2b is used."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def _gram_hashes(w: list[str], n: int) -> set[int]:
    return {_h64(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


class Decontaminator:
    """Flags documents sharing any word n-gram (default 13, as in GPT-3) with a benchmark item.

    Items shorter than n words are matched on their whole text instead."""

    def __init__(self, texts: Iterable[str], n: int = 13):
        self.n, self.grams, self.items = n, set(), 0
        for t in texts:
            w = words(t)
            if not w:
                continue
            self.items += 1
            self.grams |= _gram_hashes(w, min(n, len(w)))

    def contaminated(self, text: str) -> bool:
        w = words(text)
        return bool(self.grams) and len(w) >= 1 and not self.grams.isdisjoint(
            _gram_hashes(w, self.n) if len(w) >= self.n else _gram_hashes(w, len(w)))


_P = (1 << 61) - 1


class NearDedup:
    """MinHash over word 5-gram shingles with LSH banding (bands x rows permutations).

    A document is a near-duplicate if any band matches an earlier kept document: with 16 bands of
    8 rows, pairs with Jaccard similarity >= ~0.8 are caught with high probability."""

    def __init__(self, bands: int = 16, rows: int = 8, shingle: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        k = bands * rows
        self.a = rng.integers(1, _P, k, dtype=np.uint64)
        self.b = rng.integers(0, _P, k, dtype=np.uint64)
        self.bands, self.rows, self.shingle = bands, rows, shingle
        self.seen: set[int] = set()

    def signature(self, text: str) -> np.ndarray:
        w = words(text)
        sh = np.fromiter(_gram_hashes(w, min(self.shingle, max(len(w), 1))) or {0}, dtype=np.uint64)
        # (a*x + b) mod 2^64, folded to 61 bits: a cheap universal-style family, deterministic
        h = (sh[:, None] * self.a[None, :] + self.b[None, :]) >> np.uint64(3)
        return h.min(axis=0)

    def is_duplicate(self, text: str) -> bool:
        """True if near-duplicate of an earlier document; otherwise records this one."""
        sig = self.signature(text).reshape(self.bands, self.rows)
        keys = [int.from_bytes(hashlib.blake2b(bytes([i]) + sig[i].tobytes(), digest_size=8).digest(), "little")
                for i in range(self.bands)]
        if any(k in self.seen for k in keys):
            return True
        self.seen.update(keys)
        return False
