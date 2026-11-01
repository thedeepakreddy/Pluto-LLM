"""GPT-2-style decoder-only Transformer, written from scratch.

tokens -> token embedding + learned position embedding -> N pre-LN blocks
(causal self-attention, GELU MLP, residuals) -> final LayerNorm -> LM head
(weight-tied to the token embedding).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.config import ModelConfig


def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference causal attention: softmax(QK^T / sqrt(d) + mask) V. Shapes [B, H, T, Dh]."""
    T, S = q.size(-2), k.size(-2)  # S >= T: the last T keys are the new block, the rest is cache
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    mask = torch.triu(torch.ones(T, S, dtype=torch.bool, device=q.device), diagonal=S - T + 1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head, self.impl, self.dropout = cfg.n_head, cfg.attn_impl, cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if self.impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        else:
            y = manual_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

    def forward_cached(self, x: torch.Tensor, past: tuple[torch.Tensor, torch.Tensor] | None):
        """Attend a block of t new tokens over `past` (K, V: [B,H,P,Dh]) plus themselves.

        New token j sees all P cached positions and new positions <= j.
        Returns (output [B,t,C], (K, V) with T_total = P + t).
        """
        B, t, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (u.view(B, t, self.n_head, C // self.n_head).transpose(1, 2) for u in (q, k, v))
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        P = k.size(2) - t
        if self.impl == "sdpa":
            if t == 1:
                mask = None  # a single query may see every key
            else:
                mask = torch.ones(t, P + t, dtype=torch.bool, device=x.device).tril(diagonal=P)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = manual_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, t, C)
        return self.resid_drop(self.proj(y)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

    def forward_cached(self, x: torch.Tensor, past):
        a, kv = self.attn.forward_cached(self.ln1(x), past)
        x = x + a
        return x + self.mlp(self.ln2(x)), kv


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg, self.vocab_size = cfg, vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for b in self.blocks:
            nn.init.normal_(b.attn.proj.weight, 0.0, resid_std)
            nn.init.normal_(b.mlp.proj.weight, 0.0, resid_std)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.context_length:
            raise ValueError(f"sequence length {T} exceeds context_length {self.cfg.context_length}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def forward_cached(self, idx: torch.Tensor, cache: list | None, start_pos: int):
        """Run t new tokens at absolute positions start_pos..start_pos+t-1 using per-layer KV cache.

        Returns (logits [B,t,V], new cache). Only valid while start_pos + t <= context_length
        (learned absolute positions); callers must recompute over a cropped window beyond that.
        """
        B, t = idx.shape
        if start_pos + t > self.cfg.context_length:
            raise ValueError(
                f"position {start_pos + t} exceeds context_length {self.cfg.context_length}")
        if (cache is None) != (start_pos == 0):
            raise ValueError("cache must be None exactly when start_pos == 0")
        pos = torch.arange(start_pos, start_pos + t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block.forward_cached(x, None if cache is None else cache[i])
            new_cache.append(kv)
        return self.lm_head(self.ln_f(x)), new_cache

    def num_params(self, include_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if not include_embeddings:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n


def expected_num_params(cfg: ModelConfig, vocab_size: int) -> int:
    V, T, L, d = vocab_size, cfg.context_length, cfg.n_layer, cfg.d_model
    return V * d + T * d + L * (12 * d * d + 13 * d) + 2 * d
