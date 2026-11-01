import dataclasses
import json
import math

import pytest
import torch

from conftest import run_pipeline
from llm import cli
from llm.config import ConfigError, ModelConfig, load_config
from llm.model.gpt import GPT
from llm.runs import create_run
from llm.train.loop import lr_at, make_optimizer, resolve_precision, train


def _records(run_dir, kind=None):
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return [r for r in rows if r.get("type", "train") == kind] if kind else rows


def test_lr_schedule(tiny_config):
    t = dataclasses.replace(load_config(tiny_config).train, lr=1.0, warmup_steps=10, max_steps=110)
    assert math.isclose(lr_at(0, t), 0.1) and math.isclose(lr_at(9, t), 1.0)
    assert math.isclose(lr_at(10, t), 1.0)
    assert math.isclose(lr_at(60, t), 0.55)
    assert math.isclose(lr_at(110, t), 0.1) and math.isclose(lr_at(500, t), 0.1)


def test_resolve_precision_cpu():
    assert resolve_precision("auto", torch.device("cpu")) == "fp32"
    with pytest.raises(ConfigError):
        resolve_precision("fp16", torch.device("cpu"))


def test_weight_decay_groups(tiny_config):
    m = GPT(ModelConfig(2, 64, 2, 64), 300)
    opt = make_optimizer(m, load_config(tiny_config).train)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"]) and all(p.ndim < 2 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(m.parameters()))
    assert opt.defaults["betas"] == (0.9, 0.95)


@pytest.mark.slow
def test_overfit(tiny_config):
    cfg, _ = run_pipeline(tiny_config, ["train.max_steps=1500", "train.lr=0.003", "train.warmup_steps=20",
                                       "train.eval_every=500", "train.ckpt_every=500", "train.log_every=50"])
    run = create_run(cfg)
    state, reason = train(cfg, run)
    assert reason == "max_steps" and state.step == 1500
    assert _records(run, "train")[-1]["loss"] < 0.1


def test_two_runs_identical(tiny_config):
    cfg, _ = run_pipeline(tiny_config, ["train.max_steps=30"])
    a, b = create_run(cfg), create_run(cfg)
    train(cfg, a), train(cfg, b)
    la = [r["loss"] for r in _records(a, "train")]
    assert len(la) == 30 and la == [r["loss"] for r in _records(b, "train")]


def test_metrics_and_samples_written(prepared_tokenized):
    cfg, _ = prepared_tokenized
    run = create_run(cfg)
    state, reason = train(cfg, run)
    assert reason == "max_steps" and state.step == 40 and state.tokens_seen == 40 * 512
    first = _records(run, "train")[0]
    for key in ("step", "tokens_seen", "loss", "lr", "grad_norm", "tokens_per_s", "peak_mem_mb",
                "elapsed_s", "cost_usd"):
        assert key in first
    evals = _records(run, "eval")
    assert [e["step"] for e in evals] == [20, 40]
    assert all({"val_loss", "bpb", "ppl", "repetition_mean"} <= set(e) for e in evals)
    assert (run / "samples" / "step_00000040.jsonl").exists()
    assert (run / "checkpoints" / "step_00000040.pt").exists()


def test_loss_decreases(prepared_tokenized):
    cfg, _ = prepared_tokenized
    run = create_run(cfg)
    train(cfg, run)
    losses = [r["loss"] for r in _records(run, "train")]
    assert losses[-1] < losses[0] - 1.0


def test_cli_train(tiny_config, workspace, capsys):
    for c in ("prepare-data", "train-tokenizer", "tokenize"):
        assert cli.main([c, str(tiny_config)]) == 0
    assert cli.main(["train", str(tiny_config), "--set", "train.max_steps=5"]) == 0
    out = capsys.readouterr().out
    assert "max_steps" in out and str(workspace / "runs") in out


@pytest.mark.parametrize("capability,expected", [((7, 5), "fp16"), ((6, 0), "fp16"), ((8, 0), "bf16")])
def test_auto_precision_uses_hardware_bf16_only(monkeypatch, capability, expected):
    # torch's is_bf16_supported() also counts slow emulated bf16 on T4/P100; we must not.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    assert resolve_precision("auto", torch.device("cuda")) == expected


def test_wall_time_and_cost_include_eval_and_checkpoint_time(prepared_tokenized):
    import time

    cfg, _ = prepared_tokenized
    from llm.config import load_config_dict, config_to_dict
    d = config_to_dict(cfg)
    d["train"].update(cost_per_hour=3600.0, eval_every=10)  # $1 per second
    cfg = load_config_dict(d)
    run = create_run(cfg)
    t0 = time.time()
    state, _ = train(cfg, run)
    elapsed = time.time() - t0
    assert state.wall_time_s >= 0.8 * elapsed
    assert math.isclose(state.cost_usd, state.wall_time_s, rel_tol=1e-6)


def test_nonfinite_gradients_stop_without_poisoned_checkpoint(tiny_config, monkeypatch):
    cfg, _ = run_pipeline(tiny_config, ["train.ckpt_every=5"])
    real, calls = torch.nn.utils.clip_grad_norm_, {"n": 0}

    def poison_after_5(params, max_norm, *a, **k):
        params = list(params)
        calls["n"] += 1
        if calls["n"] > 5:
            for p in params:
                if p.grad is not None:
                    p.grad.fill_(float("nan"))
        return real(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", poison_after_5)
    run = create_run(cfg)
    state, reason = train(cfg, run)
    assert reason == "nan" and state.step == 5
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["type"] == "error" and "gradient" in rows[-1]["reason"]
    from llm.train.checkpoint import find_latest_valid
    sd = torch.load(find_latest_valid(run), weights_only=True)["model"]
    assert all(torch.isfinite(v).all() for v in sd.values())
