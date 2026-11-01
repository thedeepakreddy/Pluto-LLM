import json

import numpy as np
import pytest
import torch
from conftest import setup_speech_config

from llm import cli
from llm.config import load_config
from llm.model.gpt import GPT
from llm.model.voice import SpeechGPT
from llm.runs import create_run
from llm.speech.build import build_speech_data
from llm.speech.fakes import FakeASR, FakeCodec, FakeTTS
from llm.train.checkpoint import find_latest_valid
from llm.train.loop import train

AUDIO = {"layout": "aligned"}
SPEECH = {"max_seconds": 4.5}  # tiny context_length is 64


def _rows(run, kind="train"):
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    return [r for r in rows if r.get("type", "train") == kind]


def _speech_cfg(tiny_config, train_over=None, audio=None):
    cfg = setup_speech_config(tiny_config, speech=SPEECH, audio={**AUDIO, **(audio or {})})
    build_speech_data(cfg, FakeTTS(), FakeCodec(), FakeASR())
    overrides = [f"train.{k}={v}" for k, v in (train_over or {}).items()]
    return load_config(tiny_config, overrides)


@pytest.fixture
def cfg(tiny_config):
    return _speech_cfg(tiny_config, {"max_steps": 20, "eval_every": 10, "ckpt_every": 5,
                                     "micro_batch_size": 4, "tokens_per_step": 256})


def test_speech_resume_bitwise(cfg):
    a = create_run(cfg)
    _, reason = train(cfg, a)
    assert reason == "max_steps"
    b = create_run(cfg)
    _, reason = train(cfg, b, stop_at_step=7)
    assert reason == "stop_at_step"
    state, reason = train(cfg, b, resume=True)
    assert reason == "max_steps" and state.step == 20
    la, lb = [r["loss"] for r in _rows(a)], [r["loss"] for r in _rows(b)]
    assert la == lb and len(la) == 20
    assert [r["parts"] for r in _rows(a)] == [r["parts"] for r in _rows(b)]
    sa = torch.load(find_latest_valid(a), weights_only=True)["model"]
    sb = torch.load(find_latest_valid(b), weights_only=True)["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    # eval records and demo samples
    ev = _rows(a, "eval")
    assert ev and "val_loss" in ev[0] and "code_1" in ev[0]
    samples = sorted((a / "samples" / "step_00000010").glob("*.npy"))
    assert samples and np.load(samples[0]).shape[1] == 8


def test_text_only_steps_logged(cfg):
    run = create_run(cfg)
    train(cfg, run)
    rows = {r["step"]: r for r in _rows(run)}
    text_only = rows[10]["parts"]  # text_every=10, accum=1: optimizer step 10 is text-only
    assert text_only["text"] > 0
    assert all(text_only.get(f"code_{k}", 0.0) == 0.0 for k in range(1, 9))
    assert rows[9]["parts"]["code_1"] > 0


@pytest.mark.slow
def test_speech_overfit(tiny_config):
    cfg = _speech_cfg(tiny_config, {"max_steps": 300, "eval_every": 1000, "ckpt_every": 1000,
                                    "log_every": 1, "warmup_steps": 10, "micro_batch_size": 4,
                                    "tokens_per_step": 256})
    run = create_run(cfg)
    train(cfg, run)
    rows = _rows(run)
    speech = [r["parts"] for r in rows if r["parts"]["code_1"] > 0]
    first, last = speech[0], speech[-5:]
    assert np.mean([p["code_1"] for p in last]) < 0.3 * first["code_1"]
    assert np.mean([p["text"] for p in last]) < 0.3 * first["text"]


def _gpt_ckpt(tmp_path, cfg, **model_over):
    from dataclasses import replace

    mc = replace(cfg.model, **model_over)
    torch.manual_seed(0)
    gpt = GPT(mc, cfg.tokenizer.vocab_size)
    path = tmp_path / "m1.pt"
    torch.save({"model": gpt.state_dict()}, path)
    return gpt, path


def test_warm_start_copies_weights(cfg, tmp_path):
    from llm.train.speech_loop import warm_start_from_text

    V = cfg.tokenizer.vocab_size
    gpt, path = _gpt_ckpt(tmp_path, cfg)
    model = SpeechGPT(cfg.model, cfg.audio, V + 4)
    new_rows = model.tok_emb.weight[V:].detach().clone()
    loaded = warm_start_from_text(model, path)
    src, dst = gpt.state_dict(), model.state_dict()
    for k in src:
        if k.startswith(("blocks.", "ln_f.")):
            assert torch.equal(src[k], dst[k]) and k in loaded
    assert torch.equal(dst["pos_emb.weight"], src["pos_emb.weight"])
    assert torch.equal(dst["tok_emb.weight"][:V], src["tok_emb.weight"])
    assert torch.equal(dst["tok_emb.weight"][V:], new_rows)
    assert not torch.equal(dst["tok_emb.weight"][V:], torch.zeros_like(new_rows))
    assert "tok_emb.weight" in loaded and "pos_emb.weight" in loaded
    assert not any(k.startswith("depth.") or k.startswith("code_emb") for k in loaded)


def test_warm_start_dim_mismatch(cfg, tmp_path):
    from llm.train.speech_loop import warm_start_from_text

    _, path = _gpt_ckpt(tmp_path, cfg, d_model=32)
    model = SpeechGPT(cfg.model, cfg.audio, cfg.tokenizer.vocab_size + 4)
    with pytest.raises(ValueError, match="tok_emb.weight|pos_emb.weight|blocks"):
        warm_start_from_text(model, path)


def test_warm_start_applied_at_fresh_start(tiny_config, tmp_path):
    base = _speech_cfg(tiny_config, {"max_steps": 2, "eval_every": 1000, "warmup_steps": 1})
    gpt, path = _gpt_ckpt(tmp_path, base)
    cfg = load_config(tiny_config, ["train.max_steps=2", "train.eval_every=1000", "train.warmup_steps=1",
                                    f"audio.init_from={path}"])
    run = create_run(cfg)
    train(cfg, run, stop_at_step=1)
    info = json.loads((run / "run.json").read_text())["warm_start"]
    assert info["path"] == str(path) and "blocks.0.attn.qkv.weight" in info["loaded"]


def test_cli_train_dispatches_speech(tiny_config):
    _speech_cfg(tiny_config)
    assert cli.main(["train", str(tiny_config), "--set", "train.max_steps=2",
                     "--set", "train.eval_every=1000", "--set", "train.warmup_steps=1"]) == 0
    from pathlib import Path

    run = next(Path("runs").iterdir())
    rows = _rows(run)
    assert len(rows) == 2 and "code_1" in rows[0]["parts"]
    payload = torch.load(find_latest_valid(run), weights_only=True)
    assert any(k.startswith("depth.") for k in payload["model"])
    assert set(payload["provenance"]) == {"speech", "tokenizer", "shards"}


def test_speech_needs_complete_build(tiny_config):
    cfg = setup_speech_config(tiny_config, speech=SPEECH, audio=AUDIO)
    with pytest.raises(FileNotFoundError, match="speech"):
        train(cfg, create_run(cfg))
