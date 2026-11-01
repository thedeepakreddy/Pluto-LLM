"""SpeechGPT: a speech+text model = "brain" Transformer + small "voice box" depth Transformer.

Brain: one position per step (a text slot plus K audio codes). Its input embedding is
``tok_emb(text) + sum_k code_emb_k(codes[..., k]) + pos_emb``; it runs the M1 ``Block`` stack and
emits a hidden state per position. A text head (weight-tied to ``tok_emb``) predicts the next text
slot; the hidden state is handed to the voice box.

Voice box (``DepthTransformer``): for one brain position it predicts the K codebook tokens of the
next position, codebook by codebook. Its input sequence is
``[in_proj(h), emb_0(c_0), ..., emb_{K-2}(c_{K-2})]`` and head k at position k predicts ``c_k``.

Target shift: brain hidden at position t predicts text and codes at position t + 1. Targets whose
mask is False (including every ``EMPTY`` code) are ignored. EMPTY (= codebook_size) is a valid
*input* id (embeddings have codebook_size + 1 rows) but never a target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.config import AudioConfig, ModelConfig
from llm.generate import filter_logits
from llm.model.gpt import Block
from llm.speech.dataset import SpeechBatch

IGNORE = -100


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 0.02)


def _scale_residuals(blocks: nn.ModuleList, n_layer: int) -> None:
    std = 0.02 / math.sqrt(2 * n_layer)
    for b in blocks:
        nn.init.normal_(b.attn.proj.weight, 0.0, std)
        nn.init.normal_(b.mlp.proj.weight, 0.0, std)


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean CE over masked targets; 0 (still attached to the graph) when the mask is empty."""
    if not bool(mask.any()):
        return logits.sum() * 0.0
    t = targets.masked_fill(~mask, IGNORE)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), t.reshape(-1),
                           ignore_index=IGNORE)


class DepthTransformer(nn.Module):
    """The voice box: predicts a frame's K codes from one brain hidden state."""

    def __init__(self, audio: AudioConfig, brain_d: int):
        super().__init__()
        self.K, self.codebook_size = audio.codebooks, audio.codebook_size
        d = audio.depth_d_model
        self.cfg = ModelConfig(n_layer=audio.depth_layers, d_model=d, n_head=audio.depth_n_head,
                               context_length=self.K)
        self.in_proj = nn.Linear(brain_d, d)
        self.code_emb = nn.ModuleList(
            nn.Embedding(audio.codebook_size + 1, d) for _ in range(self.K - 1))
        self.pos_emb = nn.Parameter(torch.zeros(self.K, d))
        self.blocks = nn.ModuleList(Block(self.cfg) for _ in range(audio.depth_layers))
        self.ln_f = nn.LayerNorm(d)
        self.heads = nn.ModuleList(nn.Linear(d, audio.codebook_size) for _ in range(self.K))
        self.apply(_init_weights)
        nn.init.normal_(self.pos_emb, 0.0, 0.02)
        _scale_residuals(self.blocks, audio.depth_layers)

    def _input(self, h: torch.Tensor | None, k: int, code: torch.Tensor | None) -> torch.Tensor:
        """Embedding of depth position k ([N, d]): in_proj(h) for k = 0, else emb_{k-1}(code)."""
        x = self.in_proj(h) if k == 0 else self.code_emb[k - 1](code)
        return x + self.pos_emb[k]

    def forward(self, h: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits [N, K, codebook_size] for brain states h [N, brain_d]."""
        x = torch.stack([self._input(h, 0, None)]
                        + [self._input(None, k, codes[:, k - 1]) for k in range(1, self.K)], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return torch.stack([self.heads[k](x[:, k]) for k in range(self.K)], dim=1)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample one frame's K codes (LongTensor[K]) for h [1, brain_d]; temperature 0 = greedy."""
        if h.size(0) != 1:
            raise ValueError("DepthTransformer.sample expects a single brain state [1, brain_d]")
        out = torch.empty(self.K, dtype=torch.long, device=h.device)
        cache: list | None = None
        prev = None
        for k in range(self.K):
            x = self._input(h, k, prev).unsqueeze(1)  # [1, 1, d]
            new_cache = []
            for i, block in enumerate(self.blocks):
                x, kv = block.forward_cached(x, None if cache is None else cache[i])
                new_cache.append(kv)
            cache = new_cache
            logits = self.heads[k](self.ln_f(x))[0, 0].float()
            if temperature == 0.0:
                c = int(logits.argmax())
            else:
                logits = filter_logits(logits / temperature, top_k, None)
                c = int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator))
            out[k] = c
            prev = out[k:k + 1]
        return out


@dataclass
class SpeechOutput:
    loss: torch.Tensor
    parts: dict[str, float]
    text_logits: torch.Tensor


class SpeechGPT(nn.Module):
    def __init__(self, cfg: ModelConfig, audio: AudioConfig, text_vocab: int):
        super().__init__()
        if len(audio.code_loss_weights) != audio.codebooks:
            raise ValueError("code_loss_weights must have one weight per codebook")
        self.cfg, self.audio, self.text_vocab = cfg, audio, text_vocab
        self.K = audio.codebooks
        d = cfg.d_model
        self.tok_emb = nn.Embedding(text_vocab, d)
        self.code_emb = nn.ModuleList(
            nn.Embedding(audio.codebook_size + 1, d) for _ in range(self.K))
        self.pos_emb = nn.Embedding(cfg.context_length, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.text_head = nn.Linear(d, text_vocab, bias=False)
        self.apply(_init_weights)
        _scale_residuals(self.blocks, cfg.n_layer)
        self.text_head.weight = self.tok_emb.weight
        self.depth = DepthTransformer(audio, d)  # initialises itself

    def _embed(self, text: torch.Tensor, codes: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(text) + self.pos_emb(pos)
        for k in range(self.K):
            x = x + self.code_emb[k](codes[..., k])
        return self.drop(x)

    def hidden(self, text: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Final (post-LayerNorm) brain states [B, T, d]."""
        T = text.size(1)
        if T > self.cfg.context_length:
            raise ValueError(f"sequence length {T} exceeds context_length {self.cfg.context_length}")
        x = self._embed(text, codes, torch.arange(T, device=text.device))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, batch: SpeechBatch) -> SpeechOutput:
        h = self.hidden(batch.text, batch.codes)[:, :-1]
        text_logits = self.text_head(h)
        text_loss = _masked_ce(text_logits, batch.text[:, 1:], batch.text_mask[:, 1:])

        codes_t, cmask_t = batch.codes[:, 1:], batch.code_mask[:, 1:]
        sel = cmask_t.any(dim=-1)  # [B, T-1] positions with at least one code target
        if bool(sel.any()):
            logits = self.depth(h[sel], codes_t[sel])  # [N, K, C]
            targets, masks = codes_t[sel], cmask_t[sel]
            code_losses = [_masked_ce(logits[:, k], targets[:, k], masks[:, k])
                           for k in range(self.K)]
        else:
            zero = h.sum() * 0.0
            code_losses = [zero for _ in range(self.K)]

        w = self.audio.code_loss_weights
        code_total = sum(wk * lk for wk, lk in zip(w, code_losses)) / sum(w)
        loss = self.audio.text_loss_weight * text_loss + code_total
        parts = {"text": float(text_loss.detach())}
        parts.update({f"code_{k + 1}": float(code_losses[k].detach()) for k in range(self.K)})
        parts["total"] = float(loss.detach())
        return SpeechOutput(loss=loss, parts=parts, text_logits=text_logits)

    def step(self, text_t: torch.Tensor, codes_t: torch.Tensor, cache: list | None, pos: int):
        """KV-cached brain step for one position: (h [B, d], new cache).

        ``cache`` must be None exactly when ``pos == 0``; only valid while pos < context_length.
        """
        if pos >= self.cfg.context_length:
            raise ValueError(f"position {pos + 1} exceeds context_length {self.cfg.context_length}")
        if (cache is None) != (pos == 0):
            raise ValueError("cache must be None exactly when pos == 0")
        p = torch.tensor([pos], device=text_t.device)
        x = self._embed(text_t[:, None], codes_t[:, None, :], p)
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block.forward_cached(x, None if cache is None else cache[i])
            new_cache.append(kv)
        return self.ln_f(x)[:, 0], new_cache

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once
