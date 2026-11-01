"""Full-state checkpoints: everything needed to continue a run exactly.

Checkpoints contain only tensors, dicts, lists, tuples and primitives so they
load with torch.load(weights_only=True), which never executes pickled code.
"""
from __future__ import annotations

import dataclasses
import io
import json
import math
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from llm.manifest import atomic_write_bytes, atomic_write_json, git_sha


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    epoch: int = 0
    index: int = 0
    wall_time_s: float = 0.0
    cost_usd: float = 0.0
    best_val_loss: float = math.inf


def get_rng_state() -> dict:
    alg, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [alg, torch.from_numpy(keys.astype(np.int64)), pos, has_gauss, cached],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(state: dict) -> None:
    v, internal, gauss = state["python"]
    random.setstate((v, tuple(internal), gauss))
    alg, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((alg, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        # A run may resume on a machine with fewer GPUs (Kaggle T4x2 -> P100): restore what exists.
        for i, s in enumerate(state["cuda"][: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(s, i)


def _ckpt_dir(run_dir: Path) -> Path:
    d = Path(run_dir) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(run_dir: Path, model, optimizer, scaler, state: TrainState, provenance: dict,
                    config: dict, keep_last: int, is_best: bool) -> Path:
    d = _ckpt_dir(run_dir)
    sha, dirty = git_sha()
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "train_state": dataclasses.asdict(state),
        "rng": get_rng_state(),
        "config": config,
        "provenance": provenance,
        "git_sha": sha,
        "git_dirty": dirty,
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    name = f"step_{state.step:08d}.pt"
    path = d / name
    atomic_write_bytes(path, buf.getvalue())
    atomic_write_json(d / "latest.json", {"path": name, "step": state.step})
    if is_best:
        atomic_write_bytes(d / "best.pt", path.read_bytes())
    steps = sorted(d.glob("step_*.pt"))
    for old in steps[:-keep_last] if keep_last > 0 else []:
        old.unlink()
    return path


def load_checkpoint(path: Path, model, optimizer=None, scaler=None,
                    restore_rng: bool = True) -> tuple[TrainState, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload["scaler"] is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        set_rng_state(payload["rng"])
    return TrainState(**payload["train_state"]), payload


def _loads(path: Path) -> bool:
    try:
        torch.load(path, map_location="cpu", weights_only=True)
        return True
    except Exception as e:  # truncated or corrupt file
        warnings.warn(f"skipping unreadable checkpoint {path.name}: {type(e).__name__}")
        return False


def find_latest_valid(run_dir: Path) -> Path | None:
    d = Path(run_dir) / "checkpoints"
    candidates: list[Path] = []
    latest = d / "latest.json"
    if latest.exists():
        try:
            candidates.append(d / json.loads(latest.read_text())["path"])
        except (ValueError, KeyError, TypeError) as e:
            warnings.warn(f"ignoring unreadable latest.json ({type(e).__name__}); scanning checkpoints")
    candidates += [p for p in sorted(d.glob("step_*.pt"), reverse=True) if p not in candidates]
    for p in candidates:
        if p.exists() and _loads(p):
            return p
    return None


def load_model_for_inference(path_or_run_dir: Path):
    """Rebuild the model and tokenizer from a checkpoint file or a run directory."""
    from llm.config import load_config_dict
    from llm.model.gpt import GPT
    from llm.tokenizer.bpe import BPETokenizer, tokenizer_dir

    p = Path(path_or_run_dir)
    if p.is_dir():
        found = find_latest_valid(p)
        if found is None:
            raise FileNotFoundError(f"no valid checkpoint in {p}")
        p = found
    payload = torch.load(p, map_location="cpu", weights_only=True)
    cfg = load_config_dict(payload["config"])
    model = GPT(cfg.model, cfg.tokenizer.vocab_size)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, BPETokenizer.load(tokenizer_dir(cfg)), payload


_CREATED_REPOS: set[str] = set()


def maybe_upload(run_dir: Path, path: Path, hub_repo: str | None, is_best: bool = False) -> None:
    """Best-effort sync of the run to a private Hugging Face Hub repo; never fails training.

    Uploads the run's small files (config, run.json, metrics, samples, logs) plus the new
    checkpoint and latest.json, so `llm pull-run` can restore the run in a fresh session.
    Authentication comes from the HF_TOKEN environment variable.
    """
    if not hub_repo:
        return
    try:
        from huggingface_hub import HfApi

        api, run_dir = HfApi(), Path(run_dir)
        if hub_repo not in _CREATED_REPOS:
            api.create_repo(hub_repo, private=True, exist_ok=True)
            _CREATED_REPOS.add(hub_repo)
        api.upload_folder(folder_path=str(run_dir), path_in_repo=run_dir.name, repo_id=hub_repo,
                          ignore_patterns=["checkpoints/*", "charts/*"])
        files = [path, run_dir / "checkpoints" / "latest.json"]
        if is_best:
            files.append(run_dir / "checkpoints" / "best.pt")
        for f in files:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"{run_dir.name}/checkpoints/{f.name}",
                            repo_id=hub_repo)
    except Exception as e:
        warnings.warn(f"Hub upload failed (training continues): {e}")


def pull_run(hub_repo: str, run_id: str, runs_root: Path) -> Path:
    """Download a run directory previously synced by maybe_upload."""
    from huggingface_hub import snapshot_download

    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hub_repo, allow_patterns=[f"{run_id}/*"], local_dir=str(runs_root))
    run_dir = runs_root / run_id
    if not (run_dir / "config.yaml").exists():
        raise FileNotFoundError(f"run {run_id} not found in Hub repo {hub_repo}")
    for d in ("checkpoints", "samples"):
        (run_dir / d).mkdir(exist_ok=True)
    return run_dir
