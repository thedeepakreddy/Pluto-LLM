import math

import pytest
import torch

from llm.config import AudioConfig, ModelConfig
from llm.model.voice import DepthTransformer, SpeechGPT, SpeechOutput
from llm.speech.dataset import SpeechBatch
from llm.speech.layout import EMPTY

K, CB, V = 8, 2048, 300


def _audio(**kw):
    base = dict(depth_layers=2, depth_d_model=32, depth_n_head=4)
    base.update(kw)
    return AudioConfig(**base)


def _model(impl="sdpa", ctx=32):
    torch.manual_seed(0)
    cfg = ModelConfig(n_layer=2, d_model=48, n_head=4, context_length=ctx, attn_impl=impl)
    return SpeechGPT(cfg, _audio(), V).eval()


def _batch(B=2, T=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    text = torch.randint(0, V, (B, T), generator=g)
    codes = torch.randint(0, CB, (B, T, K), generator=g)
    codes[:, :3] = EMPTY  # some positions carry no audio
    text_mask = torch.ones(B, T, dtype=torch.bool)
    code_mask = codes != EMPTY
    return SpeechBatch(text, codes, text_mask, code_mask)


def test_shapes_and_parts():
    m = _model()
    b = _batch()
    out = m(b)
    assert isinstance(out, SpeechOutput)
    assert torch.isfinite(out.loss)
    assert out.text_logits.shape == (2, 15, V)
    assert set(out.parts) == {"text", "total", *(f"code_{k}" for k in range(1, K + 1))}
    assert all(type(v) is float for v in out.parts.values())
    out.loss.backward()


def test_param_count_m2_config():
    cfg = ModelConfig(n_layer=8, d_model=384, n_head=6, context_length=640)
    m = SpeechGPT(cfg, AudioConfig(), 4100)
    n = m.num_params()
    assert 25e6 <= n <= 31e6, n


def test_brain_causal():
    m = _model()
    b = _batch()
    with torch.no_grad():
        h1 = m.hidden(b.text, b.codes)
        codes2 = b.codes.clone()
        codes2[:, 10] = (codes2[:, 10] + 7) % CB
        text2 = b.text.clone()
        text2[:, 10] = (text2[:, 10] + 1) % V
        h2 = m.hidden(text2, codes2)
    assert torch.allclose(h1[:, :10], h2[:, :10], atol=1e-6)
    assert not torch.allclose(h1[:, 10], h2[:, 10])


def test_depth_causal_within_frame():
    torch.manual_seed(0)
    d = DepthTransformer(_audio(), brain_d=48).eval()
    h = torch.randn(5, 48)
    codes = torch.randint(0, CB, (5, K))
    with torch.no_grad():
        l1 = d(h, codes)
        codes2 = codes.clone()
        codes2[:, 3] = (codes2[:, 3] + 11) % CB
        l2 = d(h, codes2)
    assert l1.shape == (5, K, CB + 1)
    assert torch.allclose(l1[:, :4], l2[:, :4], atol=1e-6)
    assert not torch.allclose(l1[:, 4], l2[:, 4])


def test_masked_positions_ignored():
    m = _model()
    b = _batch()
    b = SpeechBatch(b.text, b.codes, torch.zeros_like(b.text_mask), torch.zeros_like(b.code_mask))
    out = m(b)
    assert out.loss.item() == 0.0
    assert all(v == 0.0 for v in out.parts.values())
    out.loss.backward()  # must not raise


def test_masked_targets_do_not_change_loss():
    """Changing a masked-out target code leaves that codebook's loss unchanged.

    The last position is used: its codes are inputs only to the brain's final state (dropped,
    it has no next-position target) and to depth heads after head 5 within its own frame.
    """
    m = _model()
    b = _batch()
    cm = b.code_mask.clone()
    cm[:, -1, 5] = False
    codes2 = b.codes.clone()
    codes2[:, -1, 5] = (codes2[:, -1, 5] + 5) % CB
    with torch.no_grad():
        p1 = m(SpeechBatch(b.text, b.codes, b.text_mask, cm)).parts
        p2 = m(SpeechBatch(b.text, codes2, b.text_mask, cm)).parts
    assert p1["code_6"] == pytest.approx(p2["code_6"], abs=1e-6)
    assert p1["code_7"] != p2["code_7"]  # sanity: the change is visible downstream


def test_initial_loss_near_ln():
    m = _model()
    with torch.no_grad():
        out = m(_batch())
    assert abs(out.parts["text"] - math.log(V)) < 0.3
    assert abs(out.parts["code_1"] - math.log(CB + 1)) < 0.3  # head 0 includes "stop"
    for k in range(2, K + 1):
        assert abs(out.parts[f"code_{k}"] - math.log(CB)) < 0.3


@pytest.mark.parametrize("impl", ["sdpa", "manual"])
def test_step_matches_hidden(impl):
    m = _model(impl)
    b = _batch(T=12)
    with torch.no_grad():
        full = m.hidden(b.text, b.codes)
        cache = None
        for t in range(12):
            h, cache = m.step(b.text[:, t], b.codes[:, t], cache, t)
            assert torch.allclose(h, full[:, t], atol=1e-5)


def test_depth_sample_matches_forward():
    torch.manual_seed(0)
    d = DepthTransformer(_audio(), brain_d=48).eval()
    h = torch.randn(1, 48)
    with torch.no_grad():
        sampled = d.sample(h, temperature=0.0, top_k=None, generator=None)
        assert sampled.shape == (K,)
        # argmax chain with teacher forcing on its own choices
        chain = torch.zeros(1, K, dtype=torch.long)
        for k in range(K):
            logits = d(h, chain)
            chain[0, k] = logits[0, k].argmax()
    assert torch.equal(sampled, chain[0])


def test_depth_sample_stochastic_in_range():
    torch.manual_seed(0)
    d = DepthTransformer(_audio(), brain_d=48).eval()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        s = d.sample(torch.randn(1, 48), temperature=1.0, top_k=50, generator=g)
    assert s.dtype == torch.long and s.shape == (K,)
    assert int(s.min()) >= 0 and int(s.max()) < CB


def test_depth_stop_class():
    torch.manual_seed(0)
    d = DepthTransformer(_audio(), brain_d=48).eval()
    h = torch.randn(1, 48)
    logits = d(h, torch.zeros(1, K, dtype=torch.long))
    assert logits.shape == (1, K, CB + 1)
    assert torch.isinf(logits[0, 1:, CB]).all() and torch.isfinite(logits[0, 0]).all()
    with torch.no_grad():
        d.heads[0].bias[CB] = 1e4  # stop dominates
        s = d.sample(h, temperature=0.0)
        assert int(s[0]) == CB and int(s[1:].max()) < CB  # delayed codebooks still sampled
        assert int(d.sample(h, temperature=1.0, allow_stop=False)[0]) < CB
        d.heads[0].bias[CB] = -1e4
        assert int(d.sample(h, temperature=0.0, force_stop=True)[0]) == CB


def test_stop_target_trains():
    m = _model()
    b = _batch()
    codes = b.codes.clone()
    codes[:, 5, 0] = EMPTY
    cm = b.code_mask.clone()
    cm[:, 5, 0] = True  # EMPTY as a codebook-0 target
    out = m(SpeechBatch(b.text, codes, b.text_mask, cm))
    assert torch.isfinite(out.loss)
    out.loss.backward()
