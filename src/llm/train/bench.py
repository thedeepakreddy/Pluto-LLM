"""Throughput benchmark on synthetic token batches: tokens/s, step time, memory, MFU, hours.

MFU uses the standard 6 * N * tokens/s estimate of training FLOPs (forward + backward, ignoring
attention FLOPs) divided by the device's dense peak for the precision in use."""
from __future__ import annotations

import dataclasses
import time

import torch
import torch.nn.functional as F

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


def attention_backend(cfg: Config, device: torch.device, dtype: torch.dtype) -> str:
    """The fastest SDPA kernel that accepts this model's attention shapes on this device."""
    if cfg.model.attn_impl != "sdpa":
        return "manual"
    if device.type != "cuda":
        return "math"
    from torch.nn.attention import SDPBackend, sdpa_kernel

    dh = cfg.model.d_model // cfg.model.n_head
    q = torch.randn(1, cfg.model.n_head, 256, dh, device=device, dtype=dtype)
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
            return name
        except RuntimeError:
            continue
    return "math"


def sweep_configs(cfg: Config, kv: bool = True, window: bool = True,
                  contexts: tuple[int, ...] = ()) -> list[tuple[str, Config]]:
    """Variants of cfg that change one thing each: KV heads, sliding window, context length.

    Tokens per micro-batch stay constant across context lengths."""
    m, t = cfg.model, cfg.train

    def with_model(**kw) -> Config:
        return dataclasses.replace(cfg, model=dataclasses.replace(m, **kw))

    rows = [("baseline", cfg)]
    if kv:
        for k in sorted({k for k in (m.n_head, 8, 4, 2, 1) if k <= m.n_head and m.n_head % k == 0},
                        reverse=True):
            if k != (m.n_kv_head or m.n_head):
                rows.append((f"kv={k}", with_model(n_kv_head=k)))
    if window and m.sliding_window is None:
        w = min(1024, m.context_length // 4)
        rows.append((f"window={w}/global_every=4", with_model(sliding_window=w, global_every=4)))
    micro_tokens = t.micro_batch_size * m.context_length
    for ctx in contexts:
        if ctx == m.context_length or micro_tokens % ctx:
            continue
        mb = micro_tokens // ctx
        c = dataclasses.replace(cfg, model=dataclasses.replace(m, context_length=ctx),
                                train=dataclasses.replace(t, micro_batch_size=mb))
        rows.append((f"ctx={ctx}", c))
    return rows


@torch.no_grad()
def _decode_speed(model, cfg: Config, device, precision: str, prompt: int, tokens: int) -> float:
    """Batch-1 generation speed with the KV cache: tokens/s after a prompt (the real-time case)."""
    model.eval()
    tokens = min(tokens, cfg.model.context_length // 2)
    prompt = max(1, min(prompt, cfg.model.context_length - tokens))
    x = torch.randint(0, cfg.tokenizer.vocab_size, (1, prompt), device=device)
    with _autocast(device, precision):
        _, cache = model.forward_cached(x, None, 0)
        nxt = x[:, -1:]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(tokens):
            _, cache = model.forward_cached(nxt, cache, prompt + i)
        if device.type == "cuda":
            torch.cuda.synchronize()
    model.train()
    return tokens / (time.perf_counter() - t0)


def bench(cfg: Config, steps: int = 20, warmup: int = 3, decode_tokens: int = 128) -> dict:
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
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)
    attn = attention_backend(cfg, device, dtype)
    if cfg.model.sliding_window is not None and device.type == "cuda":
        attn += "+flex(window)"
    decode = _decode_speed(model, cfg, device, precision, prompt=min(512, cfg.model.context_length // 2),
                           tokens=decode_tokens) if decode_tokens else None
    return {"device": name, "precision": precision, "params": n, "tokens_per_s": tps, "step_s": step_s,
            "attention": attn, "decode_tokens_per_s": decode,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
            "mfu": 6 * n * tps / peak if peak else None,
            "hours_for": lambda tokens: tokens / tps / 3600}
