"""Append-only on-disk store for one split of the speech dataset, safe to kill at any point.

Two files per split: ``<split>_codes.bin`` (uint16 row-major ``[frames, K]``) and
``<split>_index.jsonl`` (one record per chunk). A record is appended to the codes file first
and to the index second (both flushed and fsynced), so the index never points at missing codes.
On open, anything a crash left behind (a half-written index line, codes without an index
record) is cut off, so the store always ends on a record boundary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from llm.manifest import atomic_write_bytes

BYTES_PER_CODE = 2  # uint16


class SplitStore:
    def __init__(self, dir: Path, split: str, codebooks: int):
        self.codes_path = Path(dir) / f"{split}_codes.bin"
        self.index_path = Path(dir) / f"{split}_index.jsonl"
        self.codebooks = codebooks
        self.records: list[dict] = []
        self.frames = 0
        self.seconds = 0.0
        self._recover()
        self.ids = {r["id"] for r in self.records}
        self._codes_f = open(self.codes_path, "ab")  # noqa: SIM115 (kept open; see close())
        self._index_f = open(self.index_path, "ab")  # noqa: SIM115

    def _recover(self) -> None:
        records: list[dict] = []
        rewrite = False
        if self.index_path.exists():
            raw = self.index_path.read_bytes()
            lines = raw.split(b"\n")
            if lines and lines[-1] == b"":
                lines.pop()
            else:
                rewrite = True  # last line has no newline (possibly cut short)
            for i, line in enumerate(lines):
                try:
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        raise TypeError("not an object")
                except (ValueError, TypeError):
                    if i == len(lines) - 1:
                        rewrite = True
                        break
                    raise ValueError(f"{self.index_path}: corrupt record on line {i + 1}") from None
                records.append(rec)
        pos = 0
        for rec in records:
            if rec["offset"] != pos:
                raise ValueError(f"{self.index_path}: non-contiguous offset at {rec['id']}")
            pos += rec["frames"]
        codes_size = self.codes_path.stat().st_size if self.codes_path.exists() else 0
        row = self.codebooks * BYTES_PER_CODE
        while records and (records[-1]["offset"] + records[-1]["frames"]) * row > codes_size:
            records.pop()  # codes lost (e.g. disk-level truncation): forget the record
            rewrite = True
        end = sum(r["frames"] for r in records)
        if rewrite:
            atomic_write_bytes(self.index_path, b"".join(self._line(r) for r in records))
        if codes_size != end * row:
            with open(self.codes_path, "ab") as f:
                f.truncate(end * row)
        self.records = records
        self.frames = end
        self.seconds = sum(r["seconds"] for r in records)

    @staticmethod
    def _line(rec: dict) -> bytes:
        return (json.dumps(rec, sort_keys=True) + "\n").encode()

    def append(self, rec: dict, codes: np.ndarray) -> None:
        """Append one chunk. ``rec`` needs id/text/text_ids/token_frames/seconds; offset and
        frames are filled in here."""
        if codes.ndim != 2 or codes.shape[1] != self.codebooks:
            raise ValueError(f"codes must have shape (frames, {self.codebooks}), got {codes.shape}")
        if codes.size and (codes.min() < 0 or codes.max() > np.iinfo(np.uint16).max):
            raise ValueError("codes do not fit in uint16")
        rec = {**rec, "offset": self.frames, "frames": int(codes.shape[0])}
        self._write(self._codes_f, np.ascontiguousarray(codes, dtype="<u2").tobytes())
        self._write(self._index_f, self._line(rec))
        self.records.append(rec)
        self.ids.add(rec["id"])
        self.frames += rec["frames"]
        self.seconds += rec["seconds"]

    @staticmethod
    def _write(f, data: bytes) -> None:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    def close(self) -> None:
        self._codes_f.close()
        self._index_f.close()
