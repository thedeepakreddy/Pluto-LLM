"""Autoregressive sampling: temperature, top-k, top-p, repetition penalty, stop tokens, seeds."""
from __future__ import annotations

from collections.abc import Iterable

import torch


def filter_logits(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    logits = logits.clone()
    if top_k is not None and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p is not None and top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        # keep tokens until cumulative probability first reaches top_p (always keep the first)
        remove = (cum - sorted_probs) >= top_p
        remove[0] = False
        logits[sorted_idx[remove]] = float("-inf")
    return logits


def apply_repetition_penalty(logits: torch.Tensor, prev_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty == 1.0 or not prev_ids:
        return logits
    logits = logits.clone()
    idx = torch.tensor(sorted(set(prev_ids)), dtype=torch.long, device=logits.device)
    vals = logits[idx]
    logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


@torch.no_grad()
def generate(model, prompt_ids: list[int], max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None, repetition_penalty: float = 1.0,
             stop_ids: Iterable[int] = (), seed: int | None = None) -> list[int]:
    """Return only the newly generated token ids (stop token excluded)."""
    device = next(model.parameters()).device
    ctx = model.cfg.context_length
    stops = set(stop_ids)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    ids = list(prompt_ids)
    new: list[int] = []
    # Models without forward_cached, or whose forward was replaced on the instance (wrappers,
    # test doubles), take the plain full-recompute path.
    use_cache = hasattr(model, "forward_cached") and "forward" not in vars(model)
    cache = None  # per-layer KV for ids[:-1] while the whole sequence fits in the context
    for _ in range(max_new_tokens):
        if use_cache and len(ids) <= ctx:
            if cache is None:  # prefill the whole prompt
                inp, start = ids, 0
            else:  # one new token
                inp, start = ids[-1:], len(ids) - 1
            logits, cache = model.forward_cached(
                torch.tensor([inp], dtype=torch.long, device=device), cache, start)
        else:
            # learned absolute positions: the window shifted, cached keys are stale -> recompute
            cache = None
            window = torch.tensor([ids[-ctx:]], dtype=torch.long, device=device)
            logits, _ = model(window)
        logits = apply_repetition_penalty(logits[0, -1].float().cpu(), ids, repetition_penalty)
        if temperature == 0.0:
            nxt = int(torch.argmax(logits))
        else:
            logits = filter_logits(logits / temperature, top_k, top_p)
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen))
        if nxt in stops:
            break
        ids.append(nxt)
        new.append(nxt)
    return new
