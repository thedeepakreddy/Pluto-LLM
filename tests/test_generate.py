import math

import pytest
import torch

from llm.config import ModelConfig
from llm.generate import apply_repetition_penalty, filter_logits, generate
from llm.model.gpt import GPT


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(ModelConfig(n_layer=2, d_model=64, n_head=2, context_length=64), 300).eval()


def test_same_seed_same_output(model):
    a = generate(model, [1, 2, 3], 20, temperature=1.0, seed=5)
    assert a == generate(model, [1, 2, 3], 20, temperature=1.0, seed=5)
    assert a != generate(model, [1, 2, 3], 20, temperature=1.0, seed=6)


def test_greedy_is_argmax(model):
    ids = [1, 2, 3]
    expected = []
    for _ in range(5):
        logits, _ = model(torch.tensor([ids + expected]))
        expected.append(int(logits[0, -1].argmax()))
    assert generate(model, ids, 5, temperature=0.0) == expected


def test_top_k_only_allowed():
    out = filter_logits(torch.arange(10.0), top_k=3, top_p=None)
    assert torch.isfinite(out).nonzero().flatten().tolist() == [7, 8, 9]


def test_top_p_minimal_set():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    out = filter_logits(logits, top_k=None, top_p=0.8)
    assert torch.isfinite(out).nonzero().flatten().tolist() == [0, 1]


def test_top_p_keeps_at_least_one():
    out = filter_logits(torch.log(torch.tensor([0.9, 0.1])), top_k=None, top_p=0.01)
    assert torch.isfinite(out).nonzero().flatten().tolist() == [0]


def test_stops_at_stop_token(model, monkeypatch):
    def peaked(idx, targets=None):
        logits = torch.full((idx.shape[0], idx.shape[1], 300), -10.0)
        logits[..., 7] = 10.0
        return logits, None

    monkeypatch.setattr(model, "forward", peaked)
    assert generate(model, [1], 10, temperature=1.0, seed=0, stop_ids=[7]) == []


def test_max_length(model):
    assert len(generate(model, [1], 13, temperature=1.0, seed=0)) == 13


def test_generate_crops_long_prompt(model):
    assert len(generate(model, list(range(100)), 5, temperature=0.0)) == 5


def test_repetition_penalty():
    out = apply_repetition_penalty(torch.tensor([2.0, -2.0, 1.0]), [0, 1], 2.0)
    assert out.tolist() == [1.0, -4.0, 1.0]
