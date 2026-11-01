import dataclasses
import math

import pytest
import torch

from llm.config import ModelConfig
from llm.model.gpt import GPT, expected_num_params, manual_attention

CFG = ModelConfig(n_layer=2, d_model=64, n_head=2, context_length=64)


def test_output_shape():
    logits, loss = GPT(CFG, 300)(torch.randint(0, 300, (3, 17)))
    assert logits.shape == (3, 17, 300) and loss is None


def test_param_count():
    m = GPT(CFG, 300)
    assert m.num_params() == expected_num_params(CFG, 300)
    assert m.num_params(include_embeddings=False) == expected_num_params(CFG, 300) - 300 * 64 - 64 * 64


def test_weight_tying():
    m = GPT(CFG, 300)
    assert m.lm_head.weight is m.tok_emb.weight


@pytest.mark.parametrize("impl", ["sdpa", "manual"])
def test_causal_mask(impl):
    torch.manual_seed(0)
    m = GPT(dataclasses.replace(CFG, attn_impl=impl), 300).eval()
    x = torch.randint(0, 300, (2, 20))
    x2 = x.clone()
    x2[:, 10] = (x[:, 10] + 1) % 300
    a, _ = m(x)
    b, _ = m(x2)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)
    assert not torch.allclose(a[:, 10:], b[:, 10:], atol=1e-6)


def test_sdpa_matches_manual():
    torch.manual_seed(0)
    a = GPT(CFG, 300).eval()
    b = GPT(dataclasses.replace(CFG, attn_impl="manual"), 300).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 300, (2, 33))
    assert torch.allclose(a(x)[0], b(x)[0], atol=1e-5)


def test_manual_attention_is_causal_softmax():
    q = k = v = torch.randn(1, 1, 4, 8)
    out = manual_attention(q, k, v)
    assert torch.allclose(out[0, 0, 0], v[0, 0, 0], atol=1e-6)  # first position attends only to itself


def test_initial_loss_near_ln_vocab():
    torch.manual_seed(0)
    m = GPT(CFG, 300)
    x = torch.randint(0, 300, (8, 64))
    _, loss = m(x, torch.randint(0, 300, (8, 64)))
    assert abs(loss.item() - math.log(300)) < 0.3


def test_too_long_sequence_raises():
    with pytest.raises(ValueError, match="context"):
        GPT(CFG, 300)(torch.randint(0, 300, (1, 65)))


def test_residual_projection_init_scaled():
    torch.manual_seed(0)
    m = GPT(ModelConfig(n_layer=8, d_model=256, n_head=4, context_length=64), 300)
    std = m.blocks[0].attn.proj.weight.std().item()
    assert abs(std - 0.02 / math.sqrt(16)) < 0.001
    assert abs(m.blocks[0].mlp.fc.weight.std().item() - 0.02) < 0.002


def test_m1_tinystories_size():
    assert 5.5e6 < expected_num_params(ModelConfig(6, 256, 8, 256), 4096) < 6.5e6
