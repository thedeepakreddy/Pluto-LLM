"""KV-cached speech generation with SpeechGPT (M2): ``speak`` (text -> codec frames) and
``continue_speech`` (spoken prompt -> continuation).

Generation loop
---------------
The model input at position ``t`` is ``(text slot, K codes)``. Hidden state ``h_t`` (one cached
brain ``step``) predicts position ``t + 1``: the text head gives its text slot, the voice box
(``depth.sample``) its K codes. So each step: feed position ``t`` -> ``h_t`` -> choose the text
slot of ``t + 1`` and sample its codes -> feed ``t + 1`` -> ...

Position / acoustic-delay convention (identical to ``llm.speech.layout``)
-----------------------------------------------------------------------
Let ``start`` be the position of frame 0's codebook 0 (1 for the aligned layout, ``T + 2`` for
the baseline). Frame position ``p`` holds codebook 0 of frame ``p - start`` and codebooks
``1..K-1`` of frame ``p - start - delay`` (EMPTY when that index is negative). The voice box's
sample at ``p`` is therefore split: ``c[0]`` is the *new* frame's codebook 0, ``c[1:]`` completes
the frame ``delay`` positions earlier (discarded, i.e. EMPTY, for the first ``delay`` positions).

Ending:

* **END_AUDIO chosen** for position ``t + 1``: in the training layouts END_AUDIO directly follows
  the ``delay`` flush positions (codebook 0 EMPTY, codebooks 1.. of the last frames). So the last
  ``delay`` generated positions *are* the flush positions: their codebook 0 is reset to EMPTY
  and the frames are those whose codebook 0 lies before them. Every frame is then complete.
* **Frame limit reached** (``N`` codebook-0 frames, no END_AUDIO): ``delay`` explicit flush
  positions are appended (text PAD_TEXT, codebook 0 EMPTY, codebooks 1.. sampled for the last
  ``delay`` frames); only the last one is not fed (nothing reads its hidden state).

The returned array is ``extract_codes`` of the generated sequence: exactly the de-delayed
``(N, K)`` frames. Callers decode with the codec.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from llm.config import AudioConfig
from llm.model.voice import SpeechGPT
from llm.speech.chunk import token_start_frames
from llm.speech.interfaces import ASR, Codec
from llm.speech.layout import EMPTY, Example, SpecialIds, build_aligned, extract_codes, special_ids


@dataclass
class SpeechSampling:
    temperature: float = 0.8  # voice box (codes); 0 = greedy
    top_k: int = 50
    seed: int = 1234
    text_temperature: float = 0.0  # text head; 0 = greedy


# ---------------------------------------------------------------- helpers


def _specials(model: SpeechGPT, tok) -> SpecialIds:
    if model.text_vocab != tok.vocab_size + 4:
        raise ValueError(f"model text vocab {model.text_vocab} != tokenizer vocab "
                         f"{tok.vocab_size} + 4 specials")
    return special_ids(tok.vocab_size)


def _pick(logits: torch.Tensor, temperature: float, gen: torch.Generator) -> int:
    """Index from 1-D logits: argmax (ties -> lowest index) when temperature is 0, else sample."""
    logits = logits.float()
    if temperature == 0.0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen))


def _forced_choice(model: SpeechGPT, h: torch.Tensor, a: int, b: int, temperature: float,
                   gen: torch.Generator) -> int:
    """Choose between text ids ``a`` and ``b`` with the text head (``a`` wins ties)."""
    logits = model.text_head(h)[0]
    return (a, b)[_pick(torch.stack([logits[a], logits[b]]), temperature, gen)]


def _generate(model: SpeechGPT, prefix_text: list[int], prefix_codes: np.ndarray, start: int,
              max_frames: int, choose: Callable[[torch.Tensor, int], int], sp: SpecialIds,
              delay: int, sampling: SpeechSampling, gen: torch.Generator,
              forced_rest: np.ndarray | None = None) -> np.ndarray:
    """Feed the prefix, then generate up to ``max_frames`` new frame positions.

    ``choose(h, j)`` returns the text slot of the j-th generated position (END_AUDIO stops).
    ``forced_rest[j]`` (optional) overrides codebooks 1.. of the j-th generated position (known
    delayed codes of a prompt). Returns the full ``(text, codes)`` sequence as an Example's
    ``extract_codes`` -> all frames of the sequence (prefix frames included).
    """
    device = next(model.parameters()).device
    K = model.K
    text = list(prefix_text)
    codes = [np.asarray(c, dtype=np.int64) for c in prefix_codes]
    P = len(text)

    def feed(pos: int, cache):
        t = torch.tensor([text[pos]], dtype=torch.long, device=device)
        c = torch.as_tensor(codes[pos], dtype=torch.long, device=device)[None]
        return model.step(t, c, cache, pos)

    cache, h = None, None
    for pos in range(P):
        h, cache = feed(pos, cache)

    def sample_codes(pos: int, j: int, h: torch.Tensor, frame: bool) -> np.ndarray:
        c = model.depth.sample(h, sampling.temperature, sampling.top_k, gen).cpu().numpy()
        out = np.full(K, EMPTY, dtype=np.int64)
        if frame:
            out[0] = c[0]
        if pos - start - delay >= 0:
            out[1:] = c[1:]
        if forced_rest is not None and j < len(forced_rest):
            out[1:] = forced_rest[j]
        return out

    n, ended = 0, False
    while n < max_frames:
        t = len(text) - 1
        j = len(text) - P
        slot = choose(h, j)
        if slot == sp.end_audio:
            # The last `delay` generated positions become the flush positions.
            for p in range(max(P, t - delay + 1), t + 1):
                codes[p][0] = EMPTY
            text.append(sp.end_audio)
            codes.append(np.full(K, EMPTY, dtype=np.int64))
            ended = True
            break
        text.append(slot)
        codes.append(sample_codes(t + 1, j, h, frame=True))
        n += 1
        h, cache = feed(t + 1, cache)

    if not ended:  # flush the delayed codebooks of the last `delay` frames
        for i in range(delay):
            t = len(text) - 1
            j = len(text) - P
            text.append(sp.pad_text)
            codes.append(sample_codes(t + 1, j, h, frame=False))
            if i < delay - 1:
                h, cache = feed(t + 1, cache)

    T = len(text)
    ex = Example(text=np.asarray(text, dtype=np.int64), codes=np.stack(codes),
                 text_mask=np.zeros(T, dtype=bool), code_mask=np.zeros((T, K), dtype=bool))
    return extract_codes(ex, delay)


class _EvalNoGrad:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __enter__(self):
        self.was_training = self.model.training
        self.no_grad = torch.no_grad()
        self.no_grad.__enter__()
        self.model.eval()

    def __exit__(self, *exc):
        self.model.train(self.was_training)
        return self.no_grad.__exit__(*exc)


def _generator(model: SpeechGPT, seed: int) -> torch.Generator:
    device = next(model.parameters()).device
    return torch.Generator(device=device).manual_seed(int(seed))


# ---------------------------------------------------------------- public API


def speak(model: SpeechGPT, text: str, tok, audio: AudioConfig, sampling: SpeechSampling,
          max_frames: int | None = None) -> np.ndarray:
    """Text -> ``(N, K)`` codec codes (int64). The caller decodes them with the codec.

    Aligned layout: the text slot of each frame position is a forced choice by the text head
    between PAD_TEXT and the next token (while tokens remain) or PAD_TEXT and END_AUDIO (all
    placed); END_AUDIO ends generation. Baseline: BOS + tokens + AUDIO, then frames with text
    PAD_TEXT until the text head prefers END_AUDIO over PAD_TEXT. Stops after ``max_frames``
    frames at the latest (default ``min(context - tokens - delay - 3, ceil(2 * words / 2.5 *
    12.5))``, i.e. twice a 2.5 words/s speaking rate at 12.5 frames/s).
    """
    if not text or not text.strip():
        raise ValueError("text is empty")
    sp = _specials(model, tok)
    ids = tok.encode(text)
    delay = audio.acoustic_delay
    ctx = model.cfg.context_length
    # Positions used: aligned 1 + N + delay + 1; baseline T + 2 + N + delay + 1 (incl. END_AUDIO).
    if audio.layout == "aligned":
        cap = ctx - delay - 2
    else:
        cap = ctx - len(ids) - delay - 3
    if max_frames is None:
        words = len(text.split())
        max_frames = min(ctx - len(ids) - delay - 3, math.ceil(2 * words / 2.5 * 12.5))
        if max_frames <= 0:
            raise ValueError(f"text ({len(ids)} tokens) too long for context_length {ctx}")
    if max_frames < 0 or max_frames > cap:
        raise ValueError(f"max_frames {max_frames} does not fit the context "
                         f"(context_length {ctx}, at most {cap} frames)")

    gen = _generator(model, sampling.seed)
    tt = sampling.text_temperature
    K = model.K
    with _EvalNoGrad(model):
        if audio.layout == "aligned":
            nxt = [0]  # index of the next token to place

            def choose(h, j):
                if nxt[0] < len(ids):
                    slot = _forced_choice(model, h, sp.pad_text, ids[nxt[0]], tt, gen)
                    if slot != sp.pad_text:
                        nxt[0] += 1
                    return slot
                return _forced_choice(model, h, sp.pad_text, sp.end_audio, tt, gen)

            prefix_text, start = [sp.bos], 1
        else:
            def choose(h, j):
                return _forced_choice(model, h, sp.pad_text, sp.end_audio, tt, gen)

            prefix_text = [sp.bos, *ids, sp.audio]
            start = len(prefix_text)
        prefix_codes = np.full((len(prefix_text), K), EMPTY, dtype=np.int64)
        return _generate(model, prefix_text, prefix_codes, start, max_frames, choose, sp,
                         delay, sampling, gen)


def continue_speech(model: SpeechGPT, prompt_audio: np.ndarray, sample_rate: int, tok,
                    codec: Codec, asr: ASR, audio: AudioConfig, sampling: SpeechSampling,
                    max_new_frames: int = 250) -> np.ndarray:
    """Spoken prompt -> ``(N_new, K)`` codes of the continuation (prompt frames not included).

    The prompt is encoded with ``codec`` and transcribed/timed with ``asr``; tokens are aligned
    with ``token_start_frames`` and laid out with ``build_aligned``. The prefix fed to the model
    is BOS + the prompt's frame positions (most recent ones kept so that prefix +
    ``max_new_frames`` fits the context). What the layout puts after the last prompt frame
    position (its delay/flush positions and spilled tokens) is not fed as-is: those text slots
    and the prompt's delayed codebooks 1.. are forced onto the first generated positions, whose
    codebook 0 carries the new frames, so codebook 0 stays one contiguous run. After that the
    text head chooses freely (full vocabulary except BOS/AUDIO; END_AUDIO ends) until
    ``max_new_frames``.
    """
    if audio.layout != "aligned":
        raise NotImplementedError("continue-speech requires the aligned layout")
    sp = _specials(model, tok)
    delay = audio.acoustic_delay
    ctx = model.cfg.context_length
    keep = ctx - max_new_frames - max(delay - 1, 0)  # max prefix length (incl. BOS)
    if max_new_frames < 0 or keep < 2:
        raise ValueError(f"max_new_frames {max_new_frames} does not fit context_length {ctx}")

    codes = np.asarray(codec.encode(prompt_audio, sample_rate), dtype=np.int64)
    if codes.shape[0] == 0:
        raise ValueError("prompt audio is shorter than one codec frame")
    tr = asr.transcribe(prompt_audio, sample_rate)
    ids = tok.encode(tr.text) if tr.text.strip() else []
    frames = token_start_frames(tr.text, ids, tok, tr.words, codec.frame_rate) if ids else []
    ex = build_aligned(ids, frames, codes, sp, delay)
    n = codes.shape[0]
    prefix_text = ex.text[: n + 1].tolist()
    prefix_codes = ex.codes[: n + 1]
    forced_text = ex.text[n + 1 : -1].tolist()  # flush positions (+ spilled tokens), no END
    forced_rest = ex.codes[n + 1 : n + 1 + delay, 1:]
    if len(prefix_text) > keep:  # keep BOS + the most recent positions
        drop = len(prefix_text) - keep
        prefix_text = prefix_text[:1] + prefix_text[1 + drop:]
        prefix_codes = np.concatenate([prefix_codes[:1], prefix_codes[1 + drop:]])
    n_prompt = len(prefix_text) - 1  # prompt frames kept (codebook 0 at positions 1..n_prompt)

    banned = torch.tensor([sp.bos, sp.audio])
    gen = _generator(model, sampling.seed)

    def choose(h, j):
        if j < len(forced_text):
            return forced_text[j]
        logits = model.text_head(h)[0].float().clone()
        logits[banned.to(logits.device)] = float("-inf")
        return _pick(logits, sampling.text_temperature, gen)

    with _EvalNoGrad(model):
        out = _generate(model, prefix_text, prefix_codes, 1, max_new_frames, choose, sp, delay,
                        sampling, gen, forced_rest=forced_rest)
    return out[n_prompt:]


def real_time_factor(gen_seconds: float, frames: int, frame_rate: float) -> float:
    """Generation time / duration of the generated audio (< 1 = faster than real time)."""
    if frames <= 0:
        return math.inf
    return gen_seconds / (frames / frame_rate)
