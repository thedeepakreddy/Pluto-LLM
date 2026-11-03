"""GPT-2-style decoder-only Transformer, written from scratch.

tokens -> token embedding + learned position embedding -> N pre-LN blocks
(causal self-attention, GELU MLP, residuals) -> final LayerNorm -> LM head
(weight-tied to the token embedding).

M3 switches (``ModelConfig``; defaults keep the layout above): rotary position embedding
(``pos="rope"``, no position table), RMSNorm, SwiGLU feed-forward and grouped-query attention
(``n_kv_head < n_head``; the KV cache stores the smaller K/V).
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


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def make_norm(cfg: ModelConfig) -> nn.Module:
    return RMSNorm(cfg.d_model) if cfg.norm == "rmsnorm" else nn.LayerNorm(cfg.d_model)


def rope_tables(cfg: ModelConfig, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables [context_length, head_dim/2] for rotary position embedding."""
    dh = cfg.d_model // cfg.n_head
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, dh, 2, device=device).float() / dh))
    ang = torch.outer(torch.arange(cfg.context_length, device=device).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[2i], x[2i+1]) of x [B, H, T, Dh] by the angles of cos/sin [T, Dh/2]."""
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head, self.impl, self.dropout = cfg.n_head, cfg.attn_impl, cfg.dropout
        self.n_kv = cfg.n_kv_head or cfg.n_head
        self.dh = cfg.d_model // cfg.n_head
        self.rope = cfg.pos == "rope"
        kvd = self.n_kv * self.dh
        self.split = (cfg.d_model, kvd, kvd)
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model + 2 * kvd)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def _qkv(self, x: torch.Tensor, rope, start: int):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.split, dim=2)
        q = q.view(B, T, self.n_head, self.dh).transpose(1, 2)
        k = k.view(B, T, self.n_kv, self.dh).transpose(1, 2)
        v = v.view(B, T, self.n_kv, self.dh).transpose(1, 2)
        if self.rope:
            cos, sin = rope[0][start:start + T], rope[1][start:start + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return q, k, v

    def _expand(self, t: torch.Tensor) -> torch.Tensor:
        rep = self.n_head // self.n_kv
        return t if rep == 1 else t.repeat_interleave(rep, dim=1)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self._qkv(x, rope, 0)
        k, v = self._expand(k), self._expand(v)
        if self.impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        else:
            y = manual_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

    def forward_cached(self, x: torch.Tensor, past: tuple[torch.Tensor, torch.Tensor] | None,
                       rope=None):
        """Attend a block of t new tokens over `past` (K, V: [B,H_kv,P,Dh]) plus themselves.

        New token j sees all P cached positions and new positions <= j.
        Returns (output [B,t,C], (K, V) with T_total = P + t).
        """
        B, t, C = x.shape
        P0 = 0 if past is None else past[0].size(2)
        q, k, v = self._qkv(x, rope, P0)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        cache = (k, v)
        k, v = self._expand(k), self._expand(v)
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
        return self.resid_drop(self.proj(y)), cache


def swiglu_hidden(cfg: ModelConfig) -> int:
    return cfg.swiglu_hidden or 64 * math.ceil(8 * cfg.d_model / 3 / 64)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.swiglu = cfg.mlp == "swiglu"
        h = swiglu_hidden(cfg) if self.swiglu else 4 * cfg.d_model
        self.fc = nn.Linear(cfg.d_model, 2 * h if self.swiglu else h)  # swiglu: [gate | up]
        self.proj = nn.Linear(h, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.swiglu:
            gate, up = self.fc(x).chunk(2, dim=-1)
            return self.drop(self.proj(F.silu(gate) * up))
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope)
        return x + self.mlp(self.ln2(x))

    def forward_cached(self, x: torch.Tensor, past, rope=None):
        a, kv = self.attn.forward_cached(self.ln1(x), past, rope)
        x = x + a
        return x + self.mlp(self.ln2(x)), kv


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg, self.vocab_size = cfg, vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model) if cfg.pos == "learned" else None
        if cfg.pos == "rope":
            cos, sin = rope_tables(cfg)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = make_norm(cfg)
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
        x = self._embed(idx, 0)
        rope = self._rope()
        for block in self.blocks:
            x = block(x, rope)
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
        x = self._embed(idx, start_pos)
        rope = self._rope()
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block.forward_cached(x, None if cache is None else cache[i], rope)
            new_cache.append(kv)
        return self.lm_head(self.ln_f(x)), new_cache

    def _embed(self, idx: torch.Tensor, start: int) -> torch.Tensor:
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(start, start + idx.size(1), device=idx.device))
        return self.drop(x)

    def _rope(self):
        return (self.rope_cos, self.rope_sin) if self.cfg.pos == "rope" else None

    def num_params(self, include_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if not include_embeddings:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n


def expected_num_params(cfg: ModelConfig, vocab_size: int) -> int:
    V, T, L, d = vocab_size, cfg.context_length, cfg.n_layer, cfg.d_model
    kvd = (cfg.n_kv_head or cfg.n_head) * (d // cfg.n_head)
    norm = d if cfg.norm == "rmsnorm" else 2 * d
    attn = d * (d + 2 * kvd) + (d + 2 * kvd) + d * d + d
    if cfg.mlp == "swiglu":
        h = swiglu_hidden(cfg)
        mlp = d * 2 * h + 2 * h + h * d + d
    else:
        mlp = 8 * d * d + 5 * d
    pos = T * d if cfg.pos == "learned" else 0
    return V * d + pos + L * (attn + mlp + 2 * norm) + norm
