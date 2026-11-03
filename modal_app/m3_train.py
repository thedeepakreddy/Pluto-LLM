"""M3 text brain on Modal: data preparation (CPU), benchmark and training (GPU).

Usage (from the repo root, after `git pull`):
    modal run modal_app/m3_train.py --steps check                       # CPU, ~2 min: 3 rows per source
    modal run --detach modal_app/m3_train.py --steps prepare,tokenizer,shards   # CPU, several hours
    PLUTO_GPU=H100 modal run modal_app/m3_train.py --steps bench --config configs/m3-pilot.yaml
    PLUTO_GPU=H100 modal run --detach modal_app/m3_train.py --steps train --config configs/m3-pilot.yaml
    PLUTO_GPU=H100 modal run --detach modal_app/m3_train.py --steps resume --run runs/<run-id>
    modal run modal_app/m3_train.py --steps summary                     # CPU: metrics and evals

Rehearse on a cheap GPU first (never debug on H100s):
    PLUTO_GPU=L4 modal run modal_app/m3_train.py --steps bench --config configs/m3-ab-base.yaml

The code is cloned from GitHub (main) at run time; data, tokenizer, shards and runs persist on the
`pluto-m3` Volume. Checkpoints sync to the private Hub repo <your-hf-name>/pluto-runs, so a run can
be resumed after a crash or a timeout (training stops cleanly at train.max_wall_time_s, 23.5 h).
Secrets: `pluto-hf` (HF_TOKEN). Cost: see the Modal dashboard; log it in the run notes.
"""
from __future__ import annotations

import os
import re

import modal

REPO_URL = "https://github.com/thedeepakreddy/Pluto-LLM.git"
BRANCH = "main"
GPU = os.environ.get("PLUTO_GPU", "H100")
DATA_CONFIG = "configs/m3-pilot.yaml"  # data/tokenizer/shards sections are shared by all M3 configs
CPU_STEPS = ("check", "prepare", "tokenizer", "shards")
GPU_STEPS = ("bench", "train", "resume")
_SAFE = re.compile(r"^[\w./-]+$")  # config paths and run dirs only: no shell syntax

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch>=2.4", "numpy", "pyyaml", "regex", "datasets", "matplotlib", "huggingface_hub")
)
app = modal.App("pluto-m3-train", image=image)
volume = modal.Volume.from_name("pluto-m3", create_if_missing=True)
WORK = "/vol/Pluto-LLM"
SECRETS = [modal.Secret.from_name("pluto-hf")]


def gpu_command(step: str, config: str, run: str, hub: str) -> str:
    """The fixed `llm` command for a GPU step (inputs are validated paths, never shell text)."""
    for v in (config, run):
        if v and not _SAFE.match(v):
            raise ValueError(f"not a plain path: {v!r}")
    if step == "bench":
        return f"llm bench {config}"
    if step == "train":
        return f"llm train {config} --set train.hub_repo={hub}"
    if step == "resume":
        return f"llm resume {run}"
    raise ValueError(f"unknown GPU step {step!r}")


def _sh(cmd: str, check: bool = True) -> int:
    """Run a fixed command in the repo, streaming its output."""
    import subprocess

    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True, cwd=WORK).returncode
    if check and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc


def _checkout() -> str:
    import subprocess

    from huggingface_hub import HfApi

    if not os.path.exists(f"{WORK}/.git"):
        subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, WORK], check=True)
    else:  # data/, tokenizers/, shards/, runs/ are untracked and survive the reset
        subprocess.run(["git", "-C", WORK, "fetch", "-q", REPO_URL, BRANCH], check=True)
        subprocess.run(["git", "-C", WORK, "reset", "-q", "--hard", "FETCH_HEAD"], check=True)
    _sh("pip install -q --no-deps -e .")
    _sh("git log --oneline -1")
    return HfApi().whoami()["name"] + "/pluto-runs"


def _check_sources() -> None:
    """Stream 3 rows from every source: names, configs, fields and gating fail here, cheaply."""
    import sys

    sys.path.insert(0, f"{WORK}/src")
    from llm.config import load_config
    from llm.data.sources import _rows

    cfg = load_config(f"{WORK}/{DATA_CONFIG}")
    ok = True
    for src in cfg.data.sources:
        try:
            rows = [r for _, r in zip(range(3), _rows(src, streaming=True))]
            fields = sorted(rows[0]) if rows else []
            text = rows[0][src.text_field] if rows else ""
            score = rows[0].get(src.min_score_field) if src.min_score_field and rows else None
            print(f"OK   {src.name}: {len(rows)} rows, fields {fields}, text[:80]={text[:80]!r}"
                  + (f", {src.min_score_field}={score}" if src.min_score_field else ""), flush=True)
            ok &= bool(rows) and src.text_field in rows[0]
        except Exception as e:  # report every source, then fail
            ok = False
            print(f"FAIL {src.name}: {type(e).__name__}: {e}", flush=True)
    from llm.data.benchmarks import BENCHMARKS

    for name in cfg.data.decontaminate:
        try:
            print(f"OK   benchmark {name}: {sum(1 for _ in BENCHMARKS[name]())} texts", flush=True)
        except Exception as e:
            ok = False
            print(f"FAIL benchmark {name}: {type(e).__name__}: {e}", flush=True)
    if not ok:
        raise SystemExit("source check failed (see FAIL lines)")


@app.function(cpu=8, memory=32768, timeout=24 * 3600, volumes={"/vol": volume}, secrets=SECRETS)
def cpu_job(steps: list[str]) -> None:
    _checkout()
    try:
        if "check" in steps:
            _check_sources()
        if "prepare" in steps:
            _sh(f"llm prepare-data {DATA_CONFIG}")
            volume.commit()
            _sh("cat data/m3-mix-v1/data_report.md")
        if "tokenizer" in steps:
            _sh(f"llm train-tokenizer {DATA_CONFIG}")
            volume.commit()
        if "shards" in steps:
            _sh(f"llm tokenize {DATA_CONFIG}")
    finally:
        volume.commit()


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=24 * 3600, volumes={"/vol": volume}, secrets=SECRETS)
def gpu_job(step: str, config: str, run: str) -> None:
    hub = _checkout()
    _sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    try:
        if step == "resume" and not os.path.exists(f"{WORK}/{run}/config.yaml"):
            _sh(f"llm pull-run {hub} {os.path.basename(run)}")  # e.g. after the Volume was reset
        _sh(gpu_command(step, config, run, hub))
    finally:
        volume.commit()


@app.function(timeout=1800, volumes={"/vol": volume})
def summary() -> None:
    """CPU only: data report, tokenizer/shard manifests, and each run's latest metrics."""
    import glob
    import json

    for path in glob.glob(f"{WORK}/data/*/data_report.md"):
        print(f"\n##### {path}\n" + open(path).read(), flush=True)
    for path in glob.glob(f"{WORK}/shards/*/manifest.json"):
        print(f"\n##### {path}\n", json.dumps(json.load(open(path))["extra"]["splits"], indent=1)[:2000])
    for run in sorted(glob.glob(f"{WORK}/runs/*/")):
        rows = [json.loads(line) for line in open(f"{run}metrics.jsonl")] if os.path.exists(
            f"{run}metrics.jsonl") else []
        evals = [r for r in rows if r.get("type") == "eval"]
        train = [r for r in rows if r.get("type") == "train"]
        print(f"\n##### {os.path.basename(run.rstrip('/'))}: last train {train[-1] if train else '-'}")
        for e in evals[-3:]:
            print("  eval", {k: e[k] for k in ("step", "val_loss", "bpb") if k in e})
        for path in glob.glob(f"{run}text_eval_*.json"):
            print("  ", os.path.basename(path), open(path).read()[:300])


@app.local_entrypoint()
def main(steps: str = "check", config: str = "", run: str = ""):
    chosen = [s.strip() for s in steps.split(",") if s.strip()]
    if chosen == ["summary"]:
        summary.remote()
        return
    unknown = set(chosen) - set(CPU_STEPS) - set(GPU_STEPS)
    if unknown:
        raise SystemExit(f"unknown steps {sorted(unknown)}; choose from {CPU_STEPS + GPU_STEPS} or summary")
    cpu = [s for s in CPU_STEPS if s in chosen]
    gpu = [s for s in GPU_STEPS if s in chosen]
    if len(gpu) > 1:
        raise SystemExit("run one GPU step at a time")
    if gpu and gpu[0] in ("bench", "train") and not config:
        raise SystemExit(f"--steps {gpu[0]} needs --config configs/m3-....yaml")
    if gpu and gpu[0] == "resume" and not run:
        raise SystemExit("--steps resume needs --run runs/<run-id>")
    if cpu:
        print(f"CPU steps: {cpu}")
        cpu_job.remote(cpu)
    if gpu:
        gpu_command(gpu[0], config, run, "x/y")  # validate before starting a GPU
        print(f"GPU step {gpu[0]} on {GPU}")
        gpu_job.remote(gpu[0], config, run)
