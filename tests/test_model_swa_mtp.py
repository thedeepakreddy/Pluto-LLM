"""Sliding-window attention and multi-token prediction (M3 add-ons). Defaults are unchanged."""
import pytest
import torch

from llm.config import ConfigError, ModelConfig, _validate_model
from llm.model.gpt import GPT, expected_num_params, manual_attention, window_mask

V = 50
BASE = dict(n_layer=4, d_model=32, n_head=4, context_length=24, pos="rope", norm="rmsnorm",
            mlp="swiglu", n_kv_head=2)
SWA = dict(sliding_window=5, global_every=2)


def _model(**kw):
    torch.manual_seed(0)
    return GPT(ModelConfig(**{**BASE, **kw}), V).eval()


def test_window_mask():
    m = window_mask(4, 6, start=2, window=3, device="cpu")  # queries at 2..5, keys at 0..5
    assert m.tolist() == [[True, True, True, False, False, False],
                          [False, True, True, True, False, False],
                          [False, False, True, True, True, False],
                          [False, False, False, True, True, True]]


def test_window_limits_receptive_field():
    m = _model(sliding_window=3, global_every=0)  # every layer local: field = L*(w-1)+1 = 9
    x = torch.randint(0, V, (1, 20))
    y = x.clone()
    y[0, 0] = (x[0, 0] + 1) % V
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(y)
    assert torch.allclose(a[0, 9:], b[0, 9:]) and not torch.allclose(a[0, :9], b[0, :9])


def test_window_equals_manual_reference():
    q, k, v = (torch.randn(1, 2, 7, 8) for _ in range(3))
    from llm.model.gpt import _attend
    out = _attend(q, k, v, impl="sdpa", window=3, start=0)
    torch.testing.assert_close(out, manual_attention(q, k, v, window=3), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kw", [SWA, {**SWA, "mtp_heads": 1}])
def test_cached_matches_uncached_with_window(kw):
    m = _model(**kw)
    x = torch.randint(0, V, (1, 20))
    with torch.no_grad():
        full, _ = m(x)
        cache, outs = None, []
        for i in range(0, 20, 3):  # blocks of 3 new tokens
            lg, cache = m.forward_cached(x[:, i:i + 3], cache, i)
            outs.append(lg)
    torch.testing.assert_close(torch.cat(outs, dim=1), full, atol=1e-5, rtol=1e-5)
    local = [c[0].size(2) for i, c in enumerate(cache) if (i + 1) % 2 != 0]
    assert local and max(local) <= 5  # windowed layers keep only the window


def test_mtp_param_count_and_training_loss():
    cfg = ModelConfig(**{**BASE, "mtp_heads": 1, "mtp_weight": 0.5})
    torch.manual_seed(0)
    m = GPT(cfg, V)
    assert m.num_params() == expected_num_params(cfg, V)
    x = torch.randint(0, V, (2, 16))
    m.train()
    _, total = m(x[:, :-1], x[:, 1:])
    main = m.last_losses["main"]
    assert total.item() == pytest.approx(main + 0.5 * m.last_losses["mtp"], rel=1e-5)
    m.eval()
    _, ev = m(x[:, :-1], x[:, 1:])
    assert ev.item() == pytest.approx(main, rel=1e-5)  # eval loss is next-token only


def test_defaults_have_no_new_params():
    cfg = ModelConfig(**BASE)
    assert GPT(cfg, V).num_params() == expected_num_params(cfg, V)


def test_window_validation():
    with pytest.raises(ConfigError):
        _validate_model(ModelConfig(**{**BASE, "sliding_window": 0}))
    with pytest.raises(ConfigError):
        _validate_model(ModelConfig(**{**BASE, "mtp_heads": 2}))
