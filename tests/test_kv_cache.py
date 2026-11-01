import time

import pytest
import torch

from llm.config import ModelConfig
from llm.generate import generate
from llm.model.gpt import GPT


def _model(impl="sdpa", ctx=64, n_layer=2, d_model=64, n_head=2, vocab=300):
    torch.manual_seed(0)
    return GPT(ModelConfig(n_layer=n_layer, d_model=d_model, n_head=n_head,
                           context_length=ctx, attn_impl=impl), vocab).eval()


@pytest.mark.parametrize("impl", ["sdpa", "manual"])
def test_cached_equals_full_forward(impl):
    m = _model(impl)
    ids = torch.randint(0, 300, (2, 15))
    with torch.no_grad():
        logits, cache = m.forward_cached(ids[:, :10], None, 0)
        full, _ = m(ids[:, :10])
        assert torch.allclose(logits, full, atol=1e-5)
        for i in range(10, 15):
            logits, cache = m.forward_cached(ids[:, i:i + 1], cache, i)
            full, _ = m(ids[:, :i + 1])
            assert torch.allclose(logits[:, -1], full[:, -1], atol=1e-5)
    assert cache[0][0].shape == (2, 2, 15, 32)


@pytest.mark.parametrize("impl", ["sdpa", "manual"])
def test_cached_block_append_matches_full(impl):
    m = _model(impl)
    ids = torch.randint(0, 300, (1, 12))
    with torch.no_grad():
        _, cache = m.forward_cached(ids[:, :7], None, 0)
        logits, cache = m.forward_cached(ids[:, 7:], cache, 7)
        full, _ = m(ids)
    assert torch.allclose(logits, full[:, 7:], atol=1e-5)


def _reference_generate(model, prompt, n):
    ids = list(prompt)
    ctx = model.cfg.context_length
    for _ in range(n):
        logits, _ = model(torch.tensor([ids[-ctx:]]))
        ids.append(int(logits[0, -1].argmax()))
    return ids[len(prompt):]


@pytest.mark.parametrize("prompt_len", [5, 80])
def test_generate_unchanged(prompt_len):
    m = _model(ctx=64)
    prompt = list(range(1, prompt_len + 1))
    # 5-token prompt + 90 new crosses the context boundary mid-generation
    assert generate(m, prompt, 90, temperature=0.0) == _reference_generate(m, prompt, 90)


@pytest.mark.slow
def test_generate_faster_with_cache():
    m = _model(ctx=256, n_layer=4, d_model=128, n_head=4)
    prompt = [1, 2, 3, 4, 5]
    t0 = time.perf_counter()
    ref = _reference_generate(m, prompt, 200)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = generate(m, prompt, 200, temperature=0.0)
    t_new = time.perf_counter() - t0
    print(f"recompute {t_ref:.2f}s cached {t_new:.2f}s speedup {t_ref / t_new:.2f}x")
    assert out == ref
    assert t_ref / t_new >= 2.0
