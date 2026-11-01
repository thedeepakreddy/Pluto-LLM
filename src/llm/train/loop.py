"""Training loop: mixed precision, gradient accumulation, clipping, warmup + cosine
schedule, periodic evaluation, logging and checkpointing."""
from __future__ import annotations

import contextlib
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from llm.config import Config, ConfigError, TrainConfig, config_to_dict, grad_accum_steps
from llm.data.prepare import data_dir
from llm.eval.metrics import evaluate
from llm.eval.prompts import run_prompt_eval
from llm.manifest import manifest_checksum
from llm.model.gpt import GPT
from llm.runs import append_jsonl
from llm.shards import DataPosition, TokenChunks, next_batch, shard_dir, verify_shard_provenance
from llm.tokenizer.bpe import BPETokenizer, tokenizer_dir
from llm.train.checkpoint import TrainState, maybe_upload, save_checkpoint


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


def _provenance(cfg: Config) -> dict:
    return {"shards": manifest_checksum(shard_dir(cfg)), "tokenizer": manifest_checksum(tokenizer_dir(cfg)),
            "data": manifest_checksum(data_dir(cfg))}


class _Wandb:
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


def train(cfg: Config, run_dir: Path, resume: bool = False,
          stop_at_step: int | None = None) -> tuple[TrainState, str]:
    """Train (or resume) the run in run_dir. Returns the final state and why it stopped."""
    run_dir = Path(run_dir)
    t = cfg.train
    device = resolve_device(t.device)
    precision = resolve_precision(t.precision, device)
    verify_shard_provenance(cfg)
    ctx_len = cfg.model.context_length
    train_chunks = TokenChunks(shard_dir(cfg), "train", ctx_len)
    eval_chunks = TokenChunks(shard_dir(cfg), t.eval_split, ctx_len)
    tok = BPETokenizer.load(tokenizer_dir(cfg))

    _seed_everything(t.seed)
    model = GPT(cfg.model, cfg.tokenizer.vocab_size).to(device)
    optimizer = make_optimizer(model, t)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    state = TrainState()
    provenance, cfg_dict = _provenance(cfg), config_to_dict(cfg)
    fwd = torch.compile(model) if cfg.model.compile else model
    autocast = _autocast(device, precision)
    accum = grad_accum_steps(cfg)
    metrics = run_dir / "metrics.jsonl"
    wb = _Wandb(cfg, run_dir)

    def checkpoint(is_best: bool = False) -> None:
        path = save_checkpoint(run_dir, model, optimizer, scaler, state, provenance, cfg_dict,
                               t.keep_last, is_best)
        maybe_upload(run_dir, path, t.hub_repo)

    pos = DataPosition(state.epoch, state.index)
    session_start = time.time()
    model.train()
    reason = "max_steps"
    while state.step < t.max_steps:
        step_start = time.time()
        lr = lr_at(state.step, t)
        for g in optimizer.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        for _ in range(accum):
            x, y, pos = next_batch(train_chunks, pos, t.micro_batch_size, t.seed)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast:
                _, loss = fwd(x, y)
            loss = loss / accum
            loss_sum += loss.item()
            (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip).item()
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        dt = time.time() - step_start
        state.step += 1
        state.tokens_seen += t.tokens_per_step
        state.epoch, state.index = pos.epoch, pos.index
        state.wall_time_s += dt
        state.cost_usd += dt / 3600 * t.cost_per_hour

        if state.step % t.log_every == 0 or state.step == 1:
            rec = {"step": state.step, "tokens_seen": state.tokens_seen, "loss": loss_sum, "lr": lr,
                   "grad_norm": grad_norm, "tokens_per_s": t.tokens_per_step / max(dt, 1e-9),
                   "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
                   "elapsed_s": state.wall_time_s, "cost_usd": state.cost_usd}
            append_jsonl(metrics, rec)
            wb.log(rec)

        done = state.step >= t.max_steps
        if state.step % t.eval_every == 0 or done:
            ev = evaluate(model, eval_chunks, t.eval_batches, t.micro_batch_size, device, _autocast(device, precision))
            pe = run_prompt_eval(model, tok, state.step, run_dir / "samples" / f"step_{state.step:08d}.jsonl")
            rec = {"step": state.step, "type": "eval", "split": t.eval_split, **ev, **pe}
            append_jsonl(metrics, rec)
            wb.log({**rec, **{f"eval/{k}": v for k, v in ev.items()}})
            is_best = ev["val_loss"] < state.best_val_loss
            if is_best:
                state.best_val_loss = ev["val_loss"]
            checkpoint(is_best)
        elif state.step % t.ckpt_every == 0:
            checkpoint()
        if stop_at_step is not None and state.step >= stop_at_step and not done:
            checkpoint()
            reason = "stop_at_step"
            break
    del session_start
    wb.finish()
    return state, reason
