"""Evaluation metrics: validation loss, bits-per-byte, perplexity, repetition."""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch

from llm.shards import TokenChunks, epoch_order


def bits_per_byte(mean_loss_nats: float, bytes_per_token: float) -> float:
    """Loss in bits per UTF-8 byte of source text: comparable across tokenizers."""
    return mean_loss_nats / (math.log(2) * bytes_per_token)


def repetition_rate(ids: list[int], n: int = 4) -> float:
    """Fraction of n-grams that repeat an earlier n-gram (0 = no repetition)."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


@torch.no_grad()
def evaluate(model, chunks: TokenChunks, num_batches: int, batch_size: int, device: torch.device,
             autocast_ctx=None) -> dict:
    """Mean loss over a fixed set of chunks (same chunks every call)."""
    was_training = model.training
    model.eval()
    order = epoch_order(len(chunks), seed=0, epoch=0)[: num_batches * batch_size]
    total, count = 0.0, 0
    for i in range(0, len(order), batch_size):
        batch = torch.from_numpy(np.stack([chunks[int(j)] for j in order[i:i + batch_size]]).astype(np.int64))
        x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
        with autocast_ctx or contextlib.nullcontext():
            _, loss = model(x, y)
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    if was_training:
        model.train()
    mean = total / count
    return {"val_loss": mean, "bpb": bits_per_byte(mean, chunks.bytes_per_token), "ppl": math.exp(mean)}
