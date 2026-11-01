"""Artifact manifests: every artifact directory records what it contains (file
checksums) and what it was made from (checksums of upstream manifests). Any
mismatch is a hard ProvenanceError, never a warning."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

MANIFEST = "manifest.json"


class ProvenanceError(Exception):
    """An artifact does not match what its manifest or a downstream manifest claims."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode())


def git_sha() -> tuple[str | None, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, False


# Fields that describe *when/how* an artifact was made, not *what* it is. They are kept in the
# manifest for humans but excluded from its checksum, so rebuilding identical inputs on a new
# machine (e.g. after a Kaggle session dies) yields the same checksum and resume still works.
VOLATILE_FIELDS = ("created_at", "git_sha", "git_dirty", "info")


def write_manifest(dir: Path, kind: str, name: str, version: int, files: list[str],
                   inputs: dict[str, str], extra: dict | None = None, info: dict | None = None) -> dict:
    dir = Path(dir)
    sha, dirty = git_sha()
    manifest = {
        "kind": kind,
        "name": name,
        "version": version,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": sha,
        "git_dirty": dirty,
        "files": {f: sha256_file(dir / f) for f in files},
        "inputs": dict(inputs),
        "extra": extra or {},
        "info": info or {},
    }
    atomic_write_json(dir / MANIFEST, manifest)
    return json.loads((dir / MANIFEST).read_text())


def read_manifest(dir: Path) -> dict:
    p = Path(dir) / MANIFEST
    if not p.exists():
        raise ProvenanceError(f"no manifest.json in {dir}")
    return json.loads(p.read_text())


def manifest_checksum(dir: Path) -> str:
    """Checksum of what the artifact *is* (kind, name, version, files, inputs, extra)."""
    identity = {k: v for k, v in read_manifest(dir).items() if k not in VOLATILE_FIELDS}
    return sha256_bytes(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())


def verify_files(dir: Path) -> None:
    dir = Path(dir)
    for rel, expected in read_manifest(dir)["files"].items():
        p = dir / rel
        if not p.exists():
            raise ProvenanceError(f"{dir}: file listed in manifest is missing: {rel}")
        if sha256_file(p) != expected:
            raise ProvenanceError(f"{dir}: checksum mismatch for {rel}")


def verify_input(manifest: dict, label: str, upstream_dir: Path) -> None:
    recorded = manifest.get("inputs", {}).get(label)
    actual = manifest_checksum(upstream_dir)
    if recorded != actual:
        raise ProvenanceError(
            f"input '{label}' changed: this artifact was built from a different version of "
            f"{upstream_dir} (recorded {recorded}, found {actual}). Rebuild downstream artifacts."
        )
