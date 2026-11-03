"""Modern layer switches (M3): RoPE, RMSNorm, SwiGLU, grouped-query attention.

Defaults must reproduce the M1/M2 architecture exactly.
"""
import pytest
import torch

from llm.config import ConfigError, ModelConfig, _validate_model
from llm.model.gpt import GPT, expected_num_params

V = 50
BASE = dict(n_layer=2, d_model=32, n_head=4, context_length=16)
VARIANTS = {
    "rope": dict(pos="rope"),
    "rmsnorm": dict(norm="rmsnorm"),
    "swiglu": dict(mlp="swiglu"),
    "gqa": dict(n_kv_head=2),
    "all": dict(pos="rope", norm="rmsnorm", mlp="swiglu", n_kv_head=2),
}


def _model(**kw):
    torch.manual_seed(0)
    return GPT(ModelConfig(**{**BASE, **kw}), V).eval()


def test_defaults_identical_to_m1():
    # checksums of the pre-M3 GPT with this seed/config (recorded before the change)
    m = _model()
    x = torch.arange(12).view(1, 12) % V
    with torch.no_grad():
        logits, _ = m(x)
    assert float(logits.double().sum()) == pytest.approx(5.195866830224986, abs=1e-6)
    assert float(logits.double().abs().sum()) == pytest.approx(55.3225340803765, abs=1e-6)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variants_param_count(name):
    cfg = ModelConfig(**{**BASE, **VARIANTS[name]})
    assert GPT(cfg, V).num_params() == expected_num_params(cfg, V)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_cached_matches_uncached(name):
    m = _model(**VARIANTS[name])
    x = torch.randint(0, V, (1, 10))
    with torch.no_grad():
        full, _ = m(x)
        cache, outs = None, []
        for i in range(10):
            lg, cache = m.forward_cached(x[:, i:i + 1], cache, i)
            outs.append(lg)
    torch.testing.assert_close(torch.cat(outs, dim=1), full, atol=1e-5, rtol=1e-5)


def test_rope_has_no_position_table():
    m = _model(pos="rope")
    assert not any("pos_emb" in n for n, _ in m.named_parameters())
    with pytest.raises(ValueError):
        m(torch.zeros(1, 17, dtype=torch.long))


def test_gqa_requires_divisor():
    with pytest.raises(ConfigError):
        _validate_model(ModelConfig(n_layer=1, d_model=48, n_head=6, context_length=8, n_kv_head=4))
