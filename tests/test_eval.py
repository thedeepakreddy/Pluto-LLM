import hashlib
import json
import math

import torch

from llm.config import ModelConfig
from llm.eval.metrics import bits_per_byte, evaluate, repetition_rate
from llm.eval.prompts import MAX_NEW_TOKENS, PROMPT_SET_VERSION, PROMPTS, SAMPLERS, run_prompt_eval
from llm.model.gpt import GPT
from llm.shards import TokenChunks
from llm.tokenizer.bpe import BPETokenizer, tokenizer_dir

PROMPTS_SHA = "0c916ec354cd31606344f464b855d5d8ccce51aeda670393213113cd3d839f43"


def test_bpb_formula():
    assert math.isclose(bits_per_byte(math.log(2) * 4, 4.0), 1.0)


def test_repetition_rate():
    assert math.isclose(repetition_rate([1, 2, 3, 4] * 3), 1 - 4 / 9)
    assert repetition_rate([1, 2, 3]) == 0.0
    assert repetition_rate(list(range(20))) == 0.0


def _model(vocab=300):
    torch.manual_seed(0)
    return GPT(ModelConfig(n_layer=2, d_model=64, n_head=2, context_length=64), vocab)


def test_evaluate_deterministic(prepared_tokenized):
    _, sdir = prepared_tokenized
    chunks, m = TokenChunks(sdir, "train", 64), _model()
    a = evaluate(m, chunks, num_batches=2, batch_size=4, device=torch.device("cpu"))
    b = evaluate(m, chunks, num_batches=2, batch_size=4, device=torch.device("cpu"))
    assert a == b
    assert set(a) == {"val_loss", "bpb", "ppl"}
    assert math.isclose(a["ppl"], math.exp(a["val_loss"]), rel_tol=1e-6)
    assert math.isclose(a["bpb"], bits_per_byte(a["val_loss"], chunks.bytes_per_token), rel_tol=1e-9)


def test_evaluate_restores_train_mode(prepared_tokenized):
    _, sdir = prepared_tokenized
    m = _model().train()
    evaluate(m, TokenChunks(sdir, "train", 64), 1, 2, torch.device("cpu"))
    assert m.training


def test_prompt_eval_writes_all(prepared_tokenized, tmp_path):
    cfg, _ = prepared_tokenized
    tok = BPETokenizer.load(tokenizer_dir(cfg))
    m = _model()
    out = tmp_path / "samples.jsonl"
    summary = run_prompt_eval(m, tok, step=7, out_path=out, max_new_tokens=8)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == len(PROMPTS) * len(SAMPLERS) == 20
    assert all(r["prompt_set"] == PROMPT_SET_VERSION == "tinystories-v1" and r["step"] == 7 for r in rows)
    assert {r["sampler"] for r in rows} == {"greedy", "t0.8_k50"}
    assert 0.0 <= summary["repetition_mean"] <= 1.0
    assert m.training


def test_prompts_frozen():
    assert len(PROMPTS) == 10 and MAX_NEW_TOKENS == 128
    assert hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest() == PROMPTS_SHA
    assert SAMPLERS[1] == {"name": "t0.8_k50", "temperature": 0.8, "top_k": 50, "seed": 1234}
