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


def window_mask(t: int, S: int, start: int, window: int | None, device) -> torch.Tensor:
    """Allowed[query, key] for t queries at absolute positions start.. and the last S keys
    (absolute positions start+t-S ..): causal, and within `window` positions if set."""
    q = torch.arange(start, start + t, device=device)[:, None]
    k = torch.arange(start + t - S, start + t, device=device)[None, :]
    ok = k <= q
    return ok & (q - k < window) if window is not None else ok


def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     window: int | None = None) -> torch.Tensor:
    """Reference causal attention: softmax(QK^T / sqrt(d) + mask) V. Shapes [B, H, T, Dh]."""
    T, S = q.size(-2), k.size(-2)  # S >= T: the last T keys are the new block, the rest is cache
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~window_mask(T, S, S - T, window, q.device), float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


_FLEX: dict = {}


def _flex_window(q, k, v, window: int) -> torch.Tensor | None:
    """Sliding-window attention with FlexAttention (fused kernel) on CUDA; None if unavailable."""
    if q.device.type != "cuda":
        return None
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except ImportError:
        return None
    T = q.size(-2)
    key = (T, window, q.device)
    if key not in _FLEX:
        def mask_mod(b, h, qi, ki):
            return (ki <= qi) & (qi - ki < window)
        _FLEX[key] = create_block_mask(mask_mod, None, None, T, T, device=q.device)
    if "fn" not in _FLEX:
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
    return _FLEX["fn"](q, k, v, block_mask=_FLEX[key])


def _attend(q, k, v, impl: str, window: int | None, start: int, dropout: float = 0.0) -> torch.Tensor:
    """Causal (optionally sliding-window) attention of q over k, v (already expanded for GQA)."""
    T, S = q.size(-2), k.size(-2)
    if impl != "sdpa":
        return manual_attention(q, k, v, window)
    if window is None or window >= start + T:  # the window covers everything: plain causal
        if T == S:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout)
        if T == 1:
            return F.scaled_dot_product_attention(q, k, v)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=window_mask(T, S, start, None, q.device))
    if T == S and dropout == 0.0:
        out = _flex_window(q, k, v, window)
        if out is not None:
            return out
    return F.scaled_dot_product_attention(q, k, v, attn_mask=window_mask(T, S, start, window, q.device),
                                          dropout_p=dropout)


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
    def __init__(self, cfg: ModelConfig, window: int | None = None):
        super().__init__()
        self.n_head, self.impl, self.dropout = cfg.n_head, cfg.attn_impl, cfg.dropout
        self.window = window
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
        y = _attend(q, k, v, self.impl, self.window, 0, self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

    def forward_cached(self, x: torch.Tensor, past: tuple[torch.Tensor, torch.Tensor] | None,
                       rope=None, start: int | None = None):
        """Attend a block of t new tokens over `past` (K, V: [B,H_kv,P,Dh]) plus themselves.

        `start` is the absolute position of the first new token (default: the cache length; it
        differs once a sliding-window layer has dropped old keys). Returns (output [B,t,C],
        (K, V)); a windowed layer keeps only its last window - 1 keys.
        """
        B, t, C = x.shape
        P0 = (0 if past is None else past[0].size(2)) if start is None else start
        q, k, v = self._qkv(x, rope, P0)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        cache = (k, v)
        if self.window is not None and k.size(2) > self.window - 1:
            keep = k.size(2) - (self.window - 1)
            cache = (k[:, :, keep:], v[:, :, keep:])
        k, v = self._expand(k), self._expand(v)
        y = _attend(q, k, v, self.impl, self.window, P0)
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
    def __init__(self, cfg: ModelConfig, window: int | None = None):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg, window)
        self.ln2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope)
        return x + self.mlp(self.ln2(x))

    def forward_cached(self, x: torch.Tensor, past, rope=None, start: int | None = None):
        a, kv = self.attn.forward_cached(self.ln1(x), past, rope, start)
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
        self.blocks = nn.ModuleList(Block(cfg, layer_window(cfg, i)) for i in range(cfg.n_layer))
        self.ln_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        if cfg.mtp_heads:  # DeepSeek-V3-style MTP: [norm(h_t); norm(emb(x_t+1))] -> block -> x_t+2
            self.mtp_norm_h, self.mtp_norm_e, self.mtp_ln = make_norm(cfg), make_norm(cfg), make_norm(cfg)
            self.mtp_proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
            self.mtp_block = Block(cfg)
        self.last_losses: dict[str, float] = {}
        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for b in [*self.blocks, *([self.mtp_block] if cfg.mtp_heads else [])]:
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
            if self.cfg.mtp_heads and not torch.compiler.is_compiling():
                self.last_losses = {"main": loss.item()}  # for logging (eager mode only)
            if self.cfg.mtp_heads and self.training and T > 1:
                z = self.mtp_proj(torch.cat([self.mtp_norm_h(x[:, :-1]),
                                             self.mtp_norm_e(self.tok_emb(targets[:, :-1]))], dim=-1))
                lg2 = self.lm_head(self.mtp_ln(self.mtp_block(z, rope)))
                mtp = F.cross_entropy(lg2.float().reshape(-1, lg2.size(-1)), targets[:, 1:].reshape(-1))
                if not torch.compiler.is_compiling():
                    self.last_losses["mtp"] = mtp.item()
                loss = loss + self.cfg.mtp_weight * mtp
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
            x, kv = block.forward_cached(x, None if cache is None else cache[i], rope, start_pos)
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


def layer_window(cfg: ModelConfig, i: int) -> int | None:
    """Sliding window of layer i, or None for full attention."""
    if cfg.sliding_window is None or (cfg.global_every and (i + 1) % cfg.global_every == 0):
        return None
    return cfg.sliding_window


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
    mtp = (3 * norm + 2 * d * d + attn + mlp + 2 * norm) if cfg.mtp_heads else 0
    return V * d + pos + L * (attn + mlp + 2 * norm) + norm + mtp
