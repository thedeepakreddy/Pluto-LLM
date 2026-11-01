"""Training loop: mixed precision, gradient accumulation, clipping, warmup + cosine
schedule, periodic evaluation, logging, checkpointing, and exact resume.

A run stops for one of: "max_steps", "stop_at_step", "wall_time" (free-tier
session limit), "signal" (SIGTERM/SIGINT) or "nan" (non-finite loss). Every
stop except "nan" saves a checkpoint first; "nan" keeps the last good one.
"""
from __future__ import annotations

import contextlib
import math
import random
import signal
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from llm.config import Config, ConfigError, TrainConfig, config_to_dict, grad_accum_steps
from llm.data.prepare import data_dir
from llm.eval.metrics import evaluate
from llm.eval.prompts import run_prompt_eval
from llm.manifest import ProvenanceError, manifest_checksum
from llm.model.gpt import GPT
from llm.runs import append_jsonl
from llm.shards import DataPosition, TokenChunks, next_batch, shard_dir, verify_shard_provenance
from llm.tokenizer.bpe import BPETokenizer, tokenizer_dir
from llm.train.checkpoint import (TrainState, find_latest_valid, load_checkpoint, maybe_upload,
                                  save_checkpoint)


# ------------------------------------------------------------------ helpers

def lr_at(step: int, cfg: TrainConfig) -> float:
    min_lr = cfg.lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return min_lr + (cfg.lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def resolve_device(setting: str) -> torch.device:
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)


def resolve_precision(setting: str, device: torch.device) -> str:
    if device.type != "cuda":
        if setting in ("bf16", "fp16"):
            raise ConfigError(f"train.precision={setting} needs a CUDA GPU; use fp32 or auto on {device}")
        return "fp32"
    if setting == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return setting


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    fused = next(model.parameters()).device.type == "cuda"
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=cfg.betas, fused=fused)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(_flatten(v, key) if isinstance(v, dict) else {key: v})
    return out


def _config_diff(a: dict, b: dict) -> list[str]:
    fa, fb = _flatten(a), _flatten(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


class _StopFlag:
    """Turns SIGTERM/SIGINT into a flag checked between steps; restores old handlers on exit."""

    def __init__(self):
        self.set = False
        self._old = {}

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                self._old[sig] = signal.signal(sig, self._handle)
        return self

    def _handle(self, signum, frame):
        self.set = True

    def __exit__(self, *exc):
        for sig, handler in self._old.items():
            signal.signal(sig, handler)


class _Wandb:
    """Optional mirror of metrics to Weights & Biases; local files stay the source of truth."""

    def __init__(self, cfg: Config, run_dir: Path):
        self.run = None
        if cfg.train.wandb_project:
            try:
                import wandb

                self.run = wandb.init(project=cfg.train.wandb_project, id=run_dir.name, name=run_dir.name,
                                      resume="allow", config=config_to_dict(cfg))
            except Exception as e:
                warnings.warn(f"W&B disabled: {e}")

    def log(self, record: dict) -> None:
        if self.run is not None:
            self.run.log({k: v for k, v in record.items() if isinstance(v, (int, float))}, step=record["step"])

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


# ------------------------------------------------------------------ trainer

class Trainer:
    def __init__(self, cfg: Config, run_dir: Path):
        self.cfg, self.t, self.run_dir = cfg, cfg.train, Path(run_dir)
        self.device = resolve_device(self.t.device)
        self.precision = resolve_precision(self.t.precision, self.device)
        verify_shard_provenance(cfg)
        ctx = cfg.model.context_length
        self.train_chunks = TokenChunks(shard_dir(cfg), "train", ctx)
        self.eval_chunks = TokenChunks(shard_dir(cfg), self.t.eval_split, ctx)
        self.tok = BPETokenizer.load(tokenizer_dir(cfg))

        _seed_everything(self.t.seed)
        self.model = GPT(cfg.model, cfg.tokenizer.vocab_size).to(self.device)
        self.optimizer = make_optimizer(self.model, self.t)
        self.scaler = torch.amp.GradScaler("cuda") if self.precision == "fp16" else None
        self.fwd = torch.compile(self.model) if cfg.model.compile else self.model
        self.accum = grad_accum_steps(cfg)
        self.state = TrainState()
        self.provenance = {"shards": manifest_checksum(shard_dir(cfg)),
                           "tokenizer": manifest_checksum(tokenizer_dir(cfg)),
                           "data": manifest_checksum(data_dir(cfg))}
        self.cfg_dict = config_to_dict(cfg)
        self.metrics = self.run_dir / "metrics.jsonl"

    # ---------------------------------------------------------- resume
    def resume(self) -> None:
        path = find_latest_valid(self.run_dir)
        if path is None:
            raise FileNotFoundError(f"no valid checkpoint to resume from in {self.run_dir / 'checkpoints'}")
        state, payload = load_checkpoint(path, self.model, self.optimizer, self.scaler)
        diff = _config_diff(payload["config"], self.cfg_dict)
        if diff:
            raise ConfigError(f"config changed since the checkpoint was saved: {', '.join(diff)}")
        changed = [k for k in self.provenance if payload["provenance"].get(k) != self.provenance[k]]
        if changed:
            raise ProvenanceError(f"inputs changed since the checkpoint was saved: {', '.join(changed)}")
        self.state = state
        append_jsonl(self.run_dir / "resume_log.jsonl",
                     {"type": "resume", "step": state.step, "checkpoint": path.name, "at": time.time()})

    # ---------------------------------------------------------- pieces
    def checkpoint(self, is_best: bool = False) -> None:
        path = save_checkpoint(self.run_dir, self.model, self.optimizer, self.scaler, self.state,
                               self.provenance, self.cfg_dict, self.t.keep_last, is_best)
        maybe_upload(self.run_dir, path, self.t.hub_repo)

    def _optimizer_step(self, pos: DataPosition) -> tuple[DataPosition, float, float, float] | None:
        """One optimizer step over `accum` micro-batches. Returns None on a non-finite loss."""
        lr = lr_at(self.state.step, self.t)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        for _ in range(self.accum):
            x, y, pos = next_batch(self.train_chunks, pos, self.t.micro_batch_size, self.t.seed)
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with _autocast(self.device, self.precision):
                _, loss = self.fwd(x, y)
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                return None
            loss = loss / self.accum
            loss_sum += loss.item()
            (self.scaler.scale(loss) if self.scaler else loss).backward()
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.t.grad_clip).item()
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return pos, loss_sum, grad_norm, lr

    def _evaluate(self, wb: _Wandb) -> None:
        s = self.state
        ev = evaluate(self.model, self.eval_chunks, self.t.eval_batches, self.t.micro_batch_size, self.device,
                      _autocast(self.device, self.precision))
        pe = run_prompt_eval(self.model, self.tok, s.step, self.run_dir / "samples" / f"step_{s.step:08d}.jsonl")
        rec = {"step": s.step, "type": "eval", "split": self.t.eval_split, **ev, **pe}
        append_jsonl(self.metrics, rec)
        wb.log({"step": s.step, **{f"eval/{k}": v for k, v in {**ev, **pe}.items()}})
        is_best = ev["val_loss"] < s.best_val_loss
        if is_best:
            s.best_val_loss = ev["val_loss"]
        self.checkpoint(is_best)

    # ---------------------------------------------------------- main loop
    def run(self, stop_at_step: int | None = None) -> tuple[TrainState, str]:
        t, s = self.t, self.state
        pos = DataPosition(s.epoch, s.index)
        session_start, wall_at_start = time.time(), s.wall_time_s
        wb = _Wandb(self.cfg, self.run_dir)
        self.model.train()
        try:
            with _StopFlag() as stop:
                while s.step < t.max_steps:
                    if t.max_wall_time_s is not None and \
                            wall_at_start + time.time() - session_start >= t.max_wall_time_s:
                        self.checkpoint()
                        return s, "wall_time"
                    step_start = time.time()
                    result = self._optimizer_step(pos)
                    if result is None:
                        append_jsonl(self.metrics, {"step": s.step + 1, "type": "error",
                                                    "reason": "non-finite loss"})
                        return s, "nan"
                    pos, loss, grad_norm, lr = result
                    dt = time.time() - step_start
                    s.step += 1
                    s.tokens_seen += t.tokens_per_step
                    s.epoch, s.index = pos.epoch, pos.index
                    s.wall_time_s += dt
                    s.cost_usd += dt / 3600 * t.cost_per_hour

                    if s.step % t.log_every == 0 or s.step == 1:
                        rec = {"step": s.step, "tokens_seen": s.tokens_seen, "loss": loss, "lr": lr,
                               "grad_norm": grad_norm, "tokens_per_s": t.tokens_per_step / max(dt, 1e-9),
                               "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20
                               if self.device.type == "cuda" else 0.0,
                               "elapsed_s": s.wall_time_s, "cost_usd": s.cost_usd}
                        append_jsonl(self.metrics, rec)
                        wb.log(rec)

                    done = s.step >= t.max_steps
                    if s.step % t.eval_every == 0 or done:
                        self._evaluate(wb)
                    elif s.step % t.ckpt_every == 0:
                        self.checkpoint()
                    if done:
                        break
                    if stop.set:
                        self.checkpoint()
                        return s, "signal"
                    if stop_at_step is not None and s.step >= stop_at_step:
                        self.checkpoint()
                        return s, "stop_at_step"
            return s, "max_steps"
        finally:
            wb.finish()


def train(cfg: Config, run_dir: Path, resume: bool = False,
          stop_at_step: int | None = None) -> tuple[TrainState, str]:
    """Train (or resume) the run in run_dir. Returns the final state and why it stopped."""
    trainer = Trainer(cfg, run_dir)
    if resume:
        trainer.resume()
    return trainer.run(stop_at_step)
