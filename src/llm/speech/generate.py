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

Ending (delay >= 1). In the training layouts the ``delay`` positions after the last frame are
flush positions (codebook 0 EMPTY, delayed codebooks of the last frames) and codebook-0 EMPTY is
a voice-box target there ("stop audio"); END_AUDIO follows the last flush position.

* **Stop sampled**: when the voice box samples codebook 0 = EMPTY for the next position, that
  position is the first flush position. It is fed exactly as in training (text PAD_TEXT,
  codebook 0 EMPTY, delayed codebooks real); the remaining ``delay - 1`` flush positions are fed
  with codebook 0 forced EMPTY. Then the text head chooses between END_AUDIO and PAD_TEXT for
  the next position; generation ends after the flush either way (a PAD preference is logged).
  Stop is masked out while forced text remains (aligned speak: until every token is placed).
* **Frame limit reached**: ``delay`` explicit flush positions are appended the same way (codebook
  0 forced EMPTY); the last one is sampled but not fed (nothing reads its hidden state).
* With ``delay == 0`` the layouts have no flush positions, so the text head's END_AUDIO choice
  ends generation directly.

The returned array is ``extract_codes`` of the generated sequence: exactly the de-delayed
``(N, K)`` frames. Callers decode with the codec.
"""

from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)


@dataclass
class SpeechSampling:
    temperature: float = 0.8  # voice box (codes); 0 = greedy
    top_k: int = 50
    seed: int = 1234
    text_temperature: float = 0.0  # text head; 0 = greedy
    # Aligned speak only: if the text head picks PAD_TEXT (wait) for more than this many frames in
    # a row while words remain, the next word is placed anyway. None = off (pure model choice).
    max_pad_run: int | None = None


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
              max_frames: int, choose: Callable[[torch.Tensor, int], int],
              can_stop: Callable[[int], bool], sp: SpecialIds, delay: int,
              sampling: SpeechSampling, gen: torch.Generator,
              forced_rest: np.ndarray | None = None) -> np.ndarray:
    """Feed the prefix, then generate up to ``max_frames`` new frames; see the module docstring.

    ``choose(h, j)`` gives the text slot of the j-th generated frame position (END_AUDIO is only
    honoured when ``delay == 0``); ``can_stop(j)`` says whether codebook-0 EMPTY may be sampled
    there. ``forced_rest[j]`` overrides codebooks 1.. of the j-th generated position. Returns
    ``extract_codes`` of the whole generated sequence (prefix frames included).
    """
    device = next(model.parameters()).device
    K = model.K
    text: list[int] = []
    codes: list[np.ndarray] = []
    P = len(prefix_text)
    cache, h = None, None

    def feed():
        nonlocal cache, h
        pos = len(text) - 1
        t = torch.tensor([text[pos]], dtype=torch.long, device=device)
        c = torch.as_tensor(codes[pos], dtype=torch.long, device=device)[None]
        h, cache = model.step(t, c, cache, pos)

    for t_id, c in zip(prefix_text, prefix_codes):
        text.append(int(t_id))
        codes.append(np.asarray(c, dtype=np.int64).copy())
        feed()

    def sample(allow_stop: bool, force_stop: bool) -> np.ndarray:
        pos, j = len(text), len(text) - P
        c = model.depth.sample(h, sampling.temperature, sampling.top_k, gen,
                               allow_stop=allow_stop, force_stop=force_stop).cpu().numpy()
        out = np.full(K, EMPTY, dtype=np.int64)
        out[0] = c[0]
        if pos - start - delay >= 0:
            out[1:] = c[1:]
        if forced_rest is not None and j < len(forced_rest):
            out[1:] = forced_rest[j]
        return out

    def flush(first: np.ndarray | None, feed_last: bool) -> None:
        for i in range(delay):
            c = first if (i == 0 and first is not None) else sample(False, True)
            text.append(sp.pad_text)
            codes.append(c)
            if feed_last or i < delay - 1:
                feed()

    n = 0
    while n < max_frames:
        j = len(text) - P
        if delay == 0:
            slot = choose(h, j)
            if slot == sp.end_audio:
                text.append(sp.end_audio)
                codes.append(np.full(K, EMPTY, dtype=np.int64))
                break
            c = sample(False, False)
        else:
            c = sample(can_stop(j), False)
            if c[0] == EMPTY:  # "stop audio": this is the first flush position
                flush(c, feed_last=True)
                end = _forced_choice(model, h, sp.pad_text, sp.end_audio,
                                     sampling.text_temperature, gen)
                if end != sp.end_audio:
                    log.info("text head preferred PAD_TEXT over END_AUDIO after the flush; "
                             "stopping anyway")
                text.append(sp.end_audio)
                codes.append(np.full(K, EMPTY, dtype=np.int64))
                break
            slot = choose(h, j)
        text.append(slot)
        codes.append(c)
        n += 1
        feed()
    else:  # frame limit: flush the delayed codebooks of the last `delay` frames
        flush(None, feed_last=False)

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
          max_frames: int | None = None, stats: dict | None = None) -> np.ndarray:
    """Text -> ``(N, K)`` codec codes (int64). The caller decodes them with the codec.

    Aligned layout: while tokens remain, the text slot of each frame position is a forced choice
    by the text head between PAD_TEXT and the next token, and the voice box may not stop; once
    all are placed the slot is PAD_TEXT and the voice box may sample "stop audio" (codebook 0
    EMPTY), which leads to the flush positions and END_AUDIO (module docstring). Baseline: BOS +
    tokens + AUDIO, then frames with text PAD_TEXT until the voice box stops. (delay 0: no flush
    positions, the text head's PAD/END_AUDIO choice ends instead.) Stops after ``max_frames``
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
    nxt = [0]  # aligned: index of the next token to place
    pad_run = [0]
    forced = [0]
    with _EvalNoGrad(model):
        if audio.layout == "aligned":

            def choose(h, j):
                if nxt[0] < len(ids):
                    slot = _forced_choice(model, h, sp.pad_text, ids[nxt[0]], tt, gen)
                    if (slot == sp.pad_text and sampling.max_pad_run is not None
                            and pad_run[0] >= sampling.max_pad_run):
                        slot = ids[nxt[0]]  # pacing guard: the head waited too long
                        forced[0] += 1
                    if slot != sp.pad_text:
                        nxt[0] += 1
                        pad_run[0] = 0
                    else:
                        pad_run[0] += 1
                    return slot
                if delay == 0:
                    return _forced_choice(model, h, sp.pad_text, sp.end_audio, tt, gen)
                return sp.pad_text  # ending is signalled by the voice box (stop audio)

            def can_stop(j):
                return nxt[0] >= len(ids)

            prefix_text, start = [sp.bos], 1
        else:
            def choose(h, j):
                if delay == 0:
                    return _forced_choice(model, h, sp.pad_text, sp.end_audio, tt, gen)
                return sp.pad_text

            def can_stop(j):
                return True

            prefix_text = [sp.bos, *ids, sp.audio]
            start = len(prefix_text)
        prefix_codes = np.full((len(prefix_text), K), EMPTY, dtype=np.int64)
        out = _generate(model, prefix_text, prefix_codes, start, max_frames, choose, can_stop,
                        sp, delay, sampling, gen)
    if stats is not None:  # diagnostics: did the text get placed, and did the model stop itself?
        stats.update({"tokens_total": len(ids),
                      "tokens_placed": nxt[0] if audio.layout == "aligned" else len(ids),
                      "tokens_forced": forced[0], "max_frames": int(max_frames),
                      "hit_frame_limit": bool(len(out) >= max_frames)})
    return out


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
    text head chooses freely (full vocabulary except BOS/AUDIO, and END_AUDIO when delay >= 1:
    ending comes from the voice box's stop signal, as in ``speak``) until
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

    banned = torch.tensor([sp.bos, sp.audio] + ([sp.end_audio] if delay > 0 else []))
    gen = _generator(model, sampling.seed)

    def choose(h, j):
        if j < len(forced_text):
            return forced_text[j]
        logits = model.text_head(h)[0].float().clone()
        logits[banned.to(logits.device)] = float("-inf")
        return _pick(logits, sampling.text_temperature, gen)

    def can_stop(j):
        return j >= len(forced_text)

    with _EvalNoGrad(model):
        out = _generate(model, prefix_text, prefix_codes, 1, max_new_frames, choose, can_stop, sp,
                        delay, sampling, gen, forced_rest=forced_rest)
    return out[n_prompt:]


def real_time_factor(gen_seconds: float, frames: int, frame_rate: float) -> float:
    """Generation time / duration of the generated audio (< 1 = faster than real time)."""
    if frames <= 0:
        return math.inf
    return gen_seconds / (frames / frame_rate)
