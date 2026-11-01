"""Document cleaning. Every dropped document is counted under a named reason so
the counts always add up: received == kept + sum(dropped)."""
from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

REASONS = ("invalid_utf8", "too_short", "too_long", "exact_duplicate")


@dataclass
class CleanStats:
    received: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=lambda: {r: 0 for r in REASONS})


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def iter_clean(docs: Iterable[str], min_chars: int, max_chars: int, stats: CleanStats) -> Iterator[str]:
    """Yield cleaned, de-duplicated documents one at a time, counting every drop in `stats`.

    Memory stays bounded by one 32-byte digest per kept document, so a multi-GB corpus streams.
    """
    seen: set[bytes] = set()
    for doc in docs:
        stats.received += 1
        try:
            doc.encode("utf-8")
        except UnicodeEncodeError:
            stats.dropped["invalid_utf8"] += 1
            continue
        text = normalize_text(doc)
        if len(text) < min_chars:
            stats.dropped["too_short"] += 1
            continue
        if len(text) > max_chars:
            stats.dropped["too_long"] += 1
            continue
        h = hashlib.sha256(text.encode("utf-8")).digest()
        if h in seen:
            stats.dropped["exact_duplicate"] += 1
            continue
        seen.add(h)
        stats.kept += 1
        yield text


def clean_docs(docs: Iterable[str], min_chars: int, max_chars: int) -> tuple[list[str], CleanStats]:
    stats = CleanStats()
    kept = list(iter_clean(docs, min_chars, max_chars, stats))
    assert stats.received == stats.kept + sum(stats.dropped.values())
    return kept, stats
