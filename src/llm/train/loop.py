"""Training loop: mixed precision, gradient accumulation, clipping, warmup + cosine
schedule, periodic evaluation, logging, checkpointing, and exact resume.

A run stops for one of: "max_steps", "stop_at_step", "wall_time" (free-tier
session limit), "signal" (SIGTERM/SIGINT) or "nan" (non-finite loss). Every
stop except "nan" saves a checkpoint first; "nan" keeps the last good one.
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import signal
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from llm.config import (Config, ConfigError, TrainConfig, config_diff, config_to_dict,
                        grad_accum_steps)
from llm.data.prepare import data_dir
from llm.eval.metrics import evaluate
from llm.eval.prompts import run_prompt_eval
from llm.manifest import ProvenanceError, atomic_write_json, manifest_checksum, sha256_file
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
        # Native bf16 needs Ampere (sm80) or newer. torch.cuda.is_bf16_supported() also reports
        # slow *emulated* bf16 on T4/P100, which the spec says must use fp16 + GradScaler instead.
        return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    return setting


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
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

class BaseTrainer:
    """Model-agnostic training machinery. Subclasses provide the model, data and evaluation:

    - ``setup()``: load data sources (called before seeding and model construction);
    - ``build_model() -> nn.Module``;
    - ``next_batch(pos, micro) -> (batch, pos)`` where ``micro`` is the global micro-batch index;
    - ``loss(batch) -> (loss tensor, parts dict)`` (forward under autocast);
    - ``evaluate() -> dict`` holding at least ``val_loss``;
    - ``provenance_inputs() -> dict[str, str]`` of manifest checksums;
    - ``on_fresh_start()``: optional, called once when a run starts from scratch.
    """

    def __init__(self, cfg: Config, run_dir: Path):
        self.cfg, self.t, self.run_dir = cfg, cfg.train, Path(run_dir)
        self.device = resolve_device(self.t.device)
        self.precision = resolve_precision(self.t.precision, self.device)
        self.setup()

        _seed_everything(self.t.seed)
        self.model = self.build_model().to(self.device)
        self.optimizer = make_optimizer(self.model, self.t)
        self.scaler = torch.amp.GradScaler("cuda") if self.precision == "fp16" else None
        self.fwd = torch.compile(self.model) if cfg.model.compile else self.model
        self.accum = grad_accum_steps(cfg)
        self.state = TrainState()
        self.provenance = self.provenance_inputs()
        self.cfg_dict = config_to_dict(cfg)
        self.metrics = self.run_dir / "metrics.jsonl"

    # ---------------------------------------------------------- subclass hooks
    def setup(self) -> None:
        pass

    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def next_batch(self, pos: DataPosition, micro: int) -> tuple[object, DataPosition]:
        raise NotImplementedError

    def loss(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def evaluate(self) -> dict:
        raise NotImplementedError

    def provenance_inputs(self) -> dict[str, str]:
        raise NotImplementedError

    def on_fresh_start(self) -> None:
        pass

    def autocast(self):
        return _autocast(self.device, self.precision)

    # ---------------------------------------------------------- resume
    def resume(self, path: Path) -> None:
        state, payload = load_checkpoint(path, self.model, self.optimizer, self.scaler)
        changed = [k for k in self.provenance if payload["provenance"].get(k) != self.provenance[k]]
        if changed:
            raise ProvenanceError(f"inputs changed since the checkpoint was saved: {', '.join(changed)}")
        self.state = state
        append_jsonl(self.run_dir / "resume_log.jsonl",
                     {"type": "resume", "step": state.step, "checkpoint": path.name, "at": time.time()})

    # ---------------------------------------------------------- pieces
    def _account(self) -> None:
        """Add real time since the last call (training, eval, checkpointing) to wall time and cost."""
        now = time.time()
        dt, self._tick = now - self._tick, now
        self.state.wall_time_s += dt
        self.state.cost_usd += dt / 3600 * self.t.cost_per_hour

    def checkpoint(self, is_best: bool = False) -> None:
        self._account()
        path = save_checkpoint(self.run_dir, self.model, self.optimizer, self.scaler, self.state,
                               self.provenance, self.cfg_dict, self.t.keep_last, is_best)
        maybe_upload(self.run_dir, path, self.t.hub_repo, is_best)

    def _optimizer_step(self, pos: DataPosition) -> tuple[DataPosition, float, dict, float, float] | str:
        """One optimizer step over `accum` micro-batches.

        Returns (position, loss, mean loss parts, grad_norm, lr), or an error string if the loss or (without a
        GradScaler, which handles overflow itself) the gradients are non-finite; then no update
        is applied, so the last checkpoint stays clean."""
        lr = lr_at(self.state.step, self.t)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        parts_sum: dict[str, float] = {}
        for i in range(self.accum):
            batch, pos = self.next_batch(pos, self.state.step * self.accum + i)
            loss, parts = self.loss(batch)
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                return "non-finite loss"
            loss = loss / self.accum
            loss_sum += loss.item()
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v / self.accum
            (self.scaler.scale(loss) if self.scaler else loss).backward()
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.t.grad_clip).item()
        if not self.scaler and not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            return "non-finite gradients"
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return pos, loss_sum, parts_sum, grad_norm, lr

    def _evaluate(self, wb: _Wandb) -> None:
        s = self.state
        ev = self.evaluate()
        rec = {"step": s.step, "type": "eval", **ev}
        append_jsonl(self.metrics, rec)
        wb.log({"step": s.step, **{f"eval/{k}": v for k, v in ev.items()}})
        is_best = ev["val_loss"] < s.best_val_loss
        if is_best:
            s.best_val_loss = ev["val_loss"]
        self.checkpoint(is_best)

    # ---------------------------------------------------------- main loop
    def run(self, stop_at_step: int | None = None) -> tuple[TrainState, str]:
        t, s = self.t, self.state
        pos = DataPosition(s.epoch, s.index)
        session_start = self._tick = time.time()
        wb = _Wandb(self.cfg, self.run_dir)
        self.model.train()
        try:
            with _StopFlag() as stop:
                while s.step < t.max_steps:
                    # Per session: each free-tier session gets the full budget before it is killed.
                    if t.max_wall_time_s is not None and time.time() - session_start >= t.max_wall_time_s:
                        self.checkpoint()
                        return s, "wall_time"
                    step_start = time.time()
                    result = self._optimizer_step(pos)
                    if isinstance(result, str):
                        append_jsonl(self.metrics, {"step": s.step + 1, "type": "error", "reason": result})
                        return s, "nan"
                    pos, loss, parts, grad_norm, lr = result
                    dt = time.time() - step_start
                    s.step += 1
                    s.tokens_seen += t.tokens_per_step
                    s.epoch, s.index = pos.epoch, pos.index
                    self._account()

                    if s.step % t.log_every == 0 or s.step == 1:
                        rec = {"step": s.step, "tokens_seen": s.tokens_seen, "loss": loss, "lr": lr,
                               "grad_norm": grad_norm, "tokens_per_s": t.tokens_per_step / max(dt, 1e-9),
                               "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20
                               if self.device.type == "cuda" else 0.0,
                               "elapsed_s": s.wall_time_s, "cost_usd": s.cost_usd}
                        if parts:
                            rec["parts"] = parts
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


class TextTrainer(BaseTrainer):
    """The M1 text-only GPT trainer."""

    def setup(self) -> None:
        cfg = self.cfg
        verify_shard_provenance(cfg)
        ctx = cfg.model.context_length
        self.train_chunks = TokenChunks(shard_dir(cfg), "train", ctx)
        self.eval_chunks = TokenChunks(shard_dir(cfg), self.t.eval_split, ctx)
        self.tok = BPETokenizer.load(tokenizer_dir(cfg))

    def build_model(self) -> GPT:
        return GPT(self.cfg.model, self.cfg.tokenizer.vocab_size)

    def provenance_inputs(self) -> dict[str, str]:
        cfg = self.cfg
        return {"shards": manifest_checksum(shard_dir(cfg)),
                "tokenizer": manifest_checksum(tokenizer_dir(cfg)),
                "data": manifest_checksum(data_dir(cfg))}

    def on_fresh_start(self) -> None:
        if self.t.init_from is None:
            return
        path = Path(self.t.init_from)
        src = torch.load(path, map_location="cpu", weights_only=True)["model"]
        own = self.model.state_dict()
        missing, extra = sorted(set(own) - set(src)), sorted(set(src) - set(own))
        if missing or extra:
            raise ValueError(f"{path}: parameters differ (missing {missing}, unexpected {extra})")
        for k, v in src.items():
            if v.shape != own[k].shape:
                raise ValueError(f"{path}: parameter {k} has shape {tuple(v.shape)}, this model needs "
                                 f"{tuple(own[k].shape)} (a learned-position table cannot change "
                                 "context_length; use model.pos=rope)")
        self.model.load_state_dict(src)
        run_json = self.run_dir / "run.json"
        info = json.loads(run_json.read_text()) if run_json.exists() else {}
        info["init_from"] = {"path": str(path), "sha256": sha256_file(path), "at": time.time()}
        atomic_write_json(run_json, info)

    def next_batch(self, pos: DataPosition, micro: int):
        x, y, pos = next_batch(self.train_chunks, pos, self.t.micro_batch_size, self.t.seed)
        return (x, y), pos

    def loss(self, batch):
        x, y = batch
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        with self.autocast():
            _, loss = self.fwd(x, y)
        return loss, {}

    def evaluate(self) -> dict:
        s = self.state
        ev = evaluate(self.model, self.eval_chunks, self.t.eval_batches, self.t.micro_batch_size, self.device,
                      self.autocast())
        pe = run_prompt_eval(self.model, self.tok, s.step, self.run_dir / "samples" / f"step_{s.step:08d}.jsonl")
        return {"split": self.t.eval_split, **ev, **pe}


Trainer = TextTrainer  # M1 name


def train(cfg: Config, run_dir: Path, resume: bool = False,
          stop_at_step: int | None = None) -> tuple[TrainState, str]:
    """Train (or resume) the run in run_dir. Returns the final state and why it stopped."""
    path = None
    if resume:
        path = find_latest_valid(Path(run_dir))
        if path is None:
            raise FileNotFoundError(f"no valid checkpoint to resume from in {Path(run_dir) / 'checkpoints'}")
        # Compare configs *before* building anything, so an edited config is reported by key name
        # instead of failing later as a shape mismatch or a missing file.
        saved = torch.load(path, map_location="cpu", weights_only=True)["config"]
        diff = config_diff(saved, config_to_dict(cfg))
        if diff:
            raise ConfigError(f"config changed since the checkpoint was saved: {', '.join(diff)}")
    if cfg.audio is not None:
        from llm.train.speech_loop import SpeechTrainer

        trainer: BaseTrainer = SpeechTrainer(cfg, run_dir)
    else:
        trainer = TextTrainer(cfg, run_dir)
    if path is not None:
        trainer.resume(path)
    else:
        trainer.on_fresh_start()
    return trainer.run(stop_at_step)
