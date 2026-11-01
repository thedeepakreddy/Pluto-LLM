"""Run directories: one folder per training run with its resolved config,
provenance (git, hardware), metrics, samples and checkpoints."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
from pathlib import Path

import yaml

from llm.config import Config, config_to_dict
from llm.manifest import atomic_write_bytes, atomic_write_json, git_sha

LIMITATIONS_TEMPLATE = """<!-- TODO: describe known limitations of this run before sharing results -->
# Known limitations

- 
"""


def hardware_info() -> dict:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda else platform.processor() or "cpu",
        "cpu_count": os.cpu_count(),
    }


def create_run(cfg: Config) -> Path:
    sha, dirty = git_sha()
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"{cfg.run.name}-{stamp}-{sha[:7] if sha else 'nogit'}"
    root = Path(cfg.run.root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir, n = root / base, 1
    while True:
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            n += 1
            run_dir = root / f"{base}-{n}"
    (run_dir / "checkpoints").mkdir()
    (run_dir / "samples").mkdir()
    atomic_write_bytes(run_dir / "config.yaml",
                       yaml.safe_dump(config_to_dict(cfg), sort_keys=False).encode())
    atomic_write_json(run_dir / "run.json", {
        "run_id": run_dir.name, "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": sha, "git_dirty": dirty, "hardware": hardware_info()})
    (run_dir / "limitations.md").write_text(LIMITATIONS_TEMPLATE)
    return run_dir.resolve()


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
