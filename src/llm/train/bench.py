"""Throughput benchmark on synthetic token batches: tokens/s, step time, memory, MFU, hours.

MFU uses the standard 6 * N * tokens/s estimate of training FLOPs (forward + backward, ignoring
attention FLOPs) divided by the device's dense peak for the precision in use."""
from __future__ import annotations

import time

import torch

from llm.config import Config, grad_accum_steps
from llm.model.gpt import GPT
from llm.train.loop import _autocast, make_optimizer, resolve_device, resolve_precision

# dense peak FLOP/s (no sparsity) by device-name substring and precision
PEAKS = {"H100": {"bf16": 989e12, "fp16": 989e12, "fp32": 67e12},
         "A100": {"bf16": 312e12, "fp16": 312e12, "fp32": 19.5e12},
         "L4": {"bf16": 121e12, "fp16": 121e12, "fp32": 30e12},
         "A10": {"bf16": 125e12, "fp16": 125e12, "fp32": 31e12},
         "T4": {"fp16": 65e12, "fp32": 8.1e12}}


def peak_flops(device_name: str, precision: str) -> float | None:
    for key, table in PEAKS.items():
        if key in device_name:
            return table.get(precision)
    return None


def bench(cfg: Config, steps: int = 20, warmup: int = 3) -> dict:
    t = cfg.train
    device = resolve_device(t.device)
    precision = resolve_precision(t.precision, device)
    torch.manual_seed(0)
    model = GPT(cfg.model, cfg.tokenizer.vocab_size).to(device)
    fwd = torch.compile(model) if cfg.model.compile else model
    opt = make_optimizer(model, t)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    accum = grad_accum_steps(cfg)
    shape = (t.micro_batch_size, cfg.model.context_length)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def step() -> None:
        for _ in range(accum):
            x = torch.randint(0, cfg.tokenizer.vocab_size, shape, device=device)
            with _autocast(device, precision):
                _, loss = fwd(x, x)
            (scaler.scale(loss / accum) if scaler else loss / accum).backward()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):  # compile and allocator warmup, excluded
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync()
    step_s = (time.perf_counter() - t0) / steps
    tps = t.tokens_per_step / step_s
    n = model.num_params()
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    peak = peak_flops(name, precision)
    return {"device": name, "precision": precision, "params": n, "tokens_per_s": tps, "step_s": step_s,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
            "mfu": 6 * n * tps / peak if peak else None,
            "hours_for": lambda tokens: tokens / tps / 3600}
