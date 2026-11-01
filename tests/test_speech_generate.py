import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from llm.config import AudioConfig, ModelConfig
from llm.model.voice import SpeechGPT
from llm.speech.fakes import FakeASR, FakeCodec, FakeTTS
from llm.speech.generate import SpeechSampling, continue_speech, real_time_factor, speak
from llm.speech.layout import EMPTY, special_ids
from llm.tokenizer.bpe import BPETokenizer

K = 8
TEXT = "Tom found a little boat under a tree."


@pytest.fixture(scope="module")
def tok(stories_texts):
    return BPETokenizer.train(stories_texts, vocab_size=300)


def _audio(layout="aligned", delay=1):
    return AudioConfig(layout=layout, acoustic_delay=delay, depth_layers=2, depth_d_model=32,
                       depth_n_head=4)


def _model(tok, layout="aligned", ctx=256, delay=1):
    torch.manual_seed(0)
    cfg = ModelConfig(n_layer=2, d_model=64, n_head=4, context_length=ctx)
    audio = _audio(layout, delay)
    return SpeechGPT(cfg, audio, tok.vocab_size + 4), audio


class _HeadOverride(nn.Module):
    """Wraps the text head; ``fn(logits, call_index)`` edits the logits in place."""

    def __init__(self, head, fn):
        super().__init__()
        self.head, self.fn, self.calls = head, fn, 0

    def forward(self, h):
        logits = self.head(h).clone()
        self.fn(logits, self.calls)
        self.calls += 1
        return logits


def _never_end(model, sp):
    def fn(logits, i):
        logits[..., sp.end_audio] = -1e9
    model.text_head = _HeadOverride(model.text_head, fn)


def _spy_steps(model):
    fed = []
    orig = model.step

    def step(text_t, codes_t, cache, pos):
        fed.append((int(text_t[0]), codes_t[0].tolist(), pos))
        return orig(text_t, codes_t, cache, pos)

    model.step = step
    return fed


SAMPLING = SpeechSampling(temperature=0.8, top_k=50, seed=7)


def test_speak_rejects_empty_text(tok):
    m, audio = _model(tok)
    for t in ["", "   \n"]:
        with pytest.raises(ValueError, match="text is empty"):
            speak(m, t, tok, audio, SAMPLING)


def test_speak_deterministic_with_seed(tok):
    m, audio = _model(tok)
    a = speak(m, TEXT, tok, audio, SAMPLING, max_frames=12)
    b = speak(m, TEXT, tok, audio, SAMPLING, max_frames=12)
    c = speak(m, TEXT, tok, audio, SpeechSampling(seed=8), max_frames=12)
    assert a.ndim == 2 and a.shape[1] == K and a.dtype == np.int64
    assert ((a >= 0) & (a < 2048)).all()
    np.testing.assert_array_equal(a, b)
    assert a.shape != c.shape or not np.array_equal(a, c)
    hot = SpeechSampling(seed=3, text_temperature=1.0)
    np.testing.assert_array_equal(speak(m, TEXT, tok, audio, hot, max_frames=12),
                                  speak(m, TEXT, tok, audio, hot, max_frames=12))


def test_speak_restores_training_mode(tok):
    m, audio = _model(tok)
    m.train()
    speak(m, TEXT, tok, audio, SAMPLING, max_frames=4)
    assert m.training


def test_speak_places_all_tokens_aligned(tok):
    m, audio = _model(tok)
    sp = special_ids(tok.vocab_size)
    ids = tok.encode(TEXT)

    def fn(logits, i):  # alternate: PAD wins on even calls, the non-PAD candidate on odd ones
        logits[..., sp.pad_text] = 1e9 if i % 2 == 0 else -1e9

    m.text_head = _HeadOverride(m.text_head, fn)
    fed = _spy_steps(m)
    codes = speak(m, TEXT, tok, audio, SAMPLING, max_frames=200)
    slots = [t for t, _, _ in fed]
    assert slots[0] == sp.bos
    assert [t for t in slots if t < tok.vocab_size] == ids  # every token once, in order
    assert set(slots[1:]) <= set(ids) | {sp.pad_text}
    # END_AUDIO was chosen right after the last token's PAD, long before max_frames
    assert len(fed) < 200
    # positions: every fed frame position carries codebook 0; the last `delay` positions before
    # END_AUDIO are flush positions, so frames = fed frame positions - delay
    assert codes.shape == (len(fed) - 1 - audio.acoustic_delay, K)
    # returned codes are the de-delayed frames: codebook 0 of frame f was fed at position f + 1,
    # codebooks 1.. of frame f at position f + 1 + delay
    d = audio.acoustic_delay
    for f in range(codes.shape[0]):
        assert fed[f + 1][1][0] == codes[f, 0]
        assert fed[f + 1 + d][1][1:] == codes[f, 1:].tolist()
    # the first `delay` frame positions have no later-codebook content yet
    for p in range(1, 1 + d):
        assert fed[p][1][1:] == [EMPTY] * (K - 1)


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_speak_stops_at_max_frames(tok, delay):
    m, audio = _model(tok, delay=delay)
    _never_end(m, special_ids(tok.vocab_size))
    fed = _spy_steps(m)
    codes = speak(m, TEXT, tok, audio, SAMPLING, max_frames=9)
    assert codes.shape == (9, K)
    # fed positions: BOS + 9 frame positions + (delay - 1) flush positions (the last flush
    # position only needs sampling, not feeding)
    assert len(fed) == 1 + 9 + max(delay - 1, 0)
    for _, c, p in fed[1:]:
        f0, fr = p - 1, p - 1 - delay  # frame of codebook 0 / of codebooks 1..
        assert c[0] == (codes[f0, 0] if f0 < 9 else EMPTY)  # flush positions: codebook 0 EMPTY
        assert c[1:] == (codes[fr, 1:].tolist() if fr >= 0 else [EMPTY] * (K - 1))


def test_speak_default_max_frames(tok):
    m, audio = _model(tok, ctx=64)
    _never_end(m, special_ids(tok.vocab_size))
    n_tok = len(tok.encode(TEXT))
    words = len(TEXT.split())
    expected = min(64 - n_tok - audio.acoustic_delay - 3, math.ceil(2 * words / 2.5 * 12.5))
    assert speak(m, TEXT, tok, audio, SAMPLING).shape == (expected, K)


def test_speak_rejects_max_frames_beyond_context(tok):
    m, audio = _model(tok, ctx=64)
    with pytest.raises(ValueError, match="context"):
        speak(m, TEXT, tok, audio, SAMPLING, max_frames=200)


def test_speak_baseline_layout_runs(tok):
    m, audio = _model(tok, layout="text_then_audio")
    sp = special_ids(tok.vocab_size)
    ids = tok.encode(TEXT)
    _never_end(m, sp)
    fed = _spy_steps(m)
    codes = speak(m, TEXT, tok, audio, SAMPLING, max_frames=10)
    assert codes.shape == (10, K)
    slots = [t for t, _, _ in fed]
    assert slots[: len(ids) + 2] == [sp.bos, *ids, sp.audio]
    assert set(slots[len(ids) + 2:]) == {sp.pad_text}
    assert all(c == [EMPTY] * K for _, c, _ in fed[: len(ids) + 2])


def test_speak_baseline_stops_when_end_preferred(tok):
    m, audio = _model(tok, layout="text_then_audio")
    sp = special_ids(tok.vocab_size)
    n_prefix = len(tok.encode(TEXT)) + 2

    def fn(logits, i):  # END_AUDIO beats PAD from the 6th frame decision on
        logits[..., sp.end_audio] = 1e9 if i >= 5 else -1e9

    m.text_head = _HeadOverride(m.text_head, fn)
    fed = _spy_steps(m)
    codes = speak(m, TEXT, tok, audio, SAMPLING, max_frames=50)
    # decisions 0..4 chose PAD -> 5 frame positions; the last `delay` of them are flush positions
    assert len(fed) == n_prefix + 5
    assert codes.shape == (5 - audio.acoustic_delay, K)


@pytest.mark.parametrize("layout", ["aligned", "text_then_audio"])
def test_speak_uses_cache(tok, layout):
    m, audio = _model(tok, layout=layout)
    _never_end(m, special_ids(tok.vocab_size))
    fed = _spy_steps(m)

    def boom(*a, **k):
        raise AssertionError("full-sequence recompute")

    m.hidden = boom
    m.forward = boom
    codes = speak(m, TEXT, tok, audio, SAMPLING, max_frames=15)
    prefix = 1 if layout == "aligned" else len(tok.encode(TEXT)) + 2
    assert len(fed) == codes.shape[0] + prefix  # delay 1
    assert [p for _, _, p in fed] == list(range(len(fed)))


def test_continue_crops_long_prompt(tok, stories_texts):
    m, audio = _model(tok, ctx=256)
    text = " ".join(stories_texts[:2])  # > 100 words -> > 500 frames, far beyond the context
    syn = FakeTTS().synthesize(text)
    codec = FakeCodec()
    assert len(syn.audio) / 1920 > 256
    fed = _spy_steps(m)
    out = continue_speech(m, syn.audio, syn.sample_rate, tok, codec, FakeASR(), audio,
                          SAMPLING, max_new_frames=20)
    assert out.ndim == 2 and out.shape[1] == K and out.shape[0] <= 20
    assert ((out >= 0) & (out < 2048)).all()
    assert max(p for _, _, p in fed) < 256
    sp = special_ids(tok.vocab_size)
    assert fed[0][0] == sp.bos and fed[0][2] == 0
    # the prompt (cropped) was fed: its real codec codes appear as codebook 0 of the prefix
    prompt_codes = codec.encode(syn.audio, syn.sample_rate)
    assert fed[1][1][0] in set(prompt_codes[:, 0].tolist())


def test_continue_hits_limit_exactly(tok):
    m, audio = _model(tok, ctx=128)
    _never_end(m, special_ids(tok.vocab_size))
    syn = FakeTTS().synthesize("Tom found a little boat")
    prompt = FakeCodec().encode(syn.audio, syn.sample_rate)
    n = prompt.shape[0]
    fed = _spy_steps(m)
    out = continue_speech(m, syn.audio, syn.sample_rate, tok, FakeCodec(), FakeASR(), audio,
                          SAMPLING, max_new_frames=12)
    assert out.shape == (12, K)
    # prefix = BOS + n prompt frame positions; the first generated position carries the new
    # frame's codebook 0 and the prompt's last frame's (delayed) codebooks 1..
    assert [c[0] for _, c, _ in fed[1 : n + 1]] == prompt[:, 0].tolist()
    assert fed[n + 1][1][0] == out[0, 0]
    assert fed[n + 1][1][1:] == prompt[-1, 1:].tolist()
    assert fed[n + 2][1][1:] == out[0, 1:].tolist()
    assert len(fed) == n + 1 + 12  # delay 1: the flush position is sampled, not fed


def test_continue_requires_aligned(tok):
    m, audio = _model(tok, layout="text_then_audio")
    syn = FakeTTS().synthesize("Tom found a little boat")
    with pytest.raises(NotImplementedError, match="continue-speech requires the aligned layout"):
        continue_speech(m, syn.audio, syn.sample_rate, tok, FakeCodec(), FakeASR(), audio,
                        SAMPLING)


def test_real_time_factor():
    assert real_time_factor(2.0, 25, 12.5) == pytest.approx(1.0)
    assert real_time_factor(0.5, 25, 12.5) == pytest.approx(0.25)
    assert real_time_factor(1.0, 0, 12.5) == math.inf
