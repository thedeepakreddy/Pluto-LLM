import json
import os
import signal

import pytest
import torch

from conftest import edit_yaml, run_pipeline
from llm import cli
from llm.config import ConfigError, load_config
from llm.manifest import ProvenanceError
from llm.model.gpt import GPT
from llm.runs import create_run
from llm.shards import write_shards
from llm.tokenizer.bpe import train_tokenizer
from llm.train import loop
from llm.train.checkpoint import find_latest_valid
from llm.train.loop import train


def _losses(run):
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    return [r["loss"] for r in rows if r.get("type", "train") == "train"]


def _latest_step(run):
    return json.loads((run / "checkpoints" / "latest.json").read_text())["step"]


@pytest.fixture
def cfg(prepared_tokenized):
    return prepared_tokenized[0]


def test_resume_bitwise_equal(cfg):
    a = create_run(cfg)
    train(cfg, a)
    b = create_run(cfg)
    _, reason = train(cfg, b, stop_at_step=17)
    assert reason == "stop_at_step" and _latest_step(b) == 17
    state, reason = train(cfg, b, resume=True)
    assert reason == "max_steps" and state.step == 40
    assert _losses(a) == _losses(b) and len(_losses(a)) == 40
    sa = torch.load(find_latest_valid(a), weights_only=True)["model"]
    sb = torch.load(find_latest_valid(b), weights_only=True)["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_resume_log_written(cfg):
    run = create_run(cfg)
    train(cfg, run, stop_at_step=5)
    train(cfg, run, resume=True, stop_at_step=8)
    log = [json.loads(l) for l in (run / "resume_log.jsonl").read_text().splitlines()]
    assert len(log) == 1 and log[0]["step"] == 5 and log[0]["type"] == "resume"


def test_resume_without_checkpoint_fails(cfg):
    with pytest.raises(FileNotFoundError):
        train(cfg, create_run(cfg), resume=True)


def test_resume_refuses_changed_config(cfg):
    run = create_run(cfg)
    train(cfg, run, stop_at_step=5)
    edit_yaml(run / "config.yaml", lambda d: d["train"].update(lr=0.1))
    with pytest.raises(ConfigError, match="train.lr"):
        train(load_config(run / "config.yaml"), run, resume=True)


def test_resume_after_rebuilding_identical_artifacts(cfg):
    # A dead Kaggle session loses data/, tokenizer/, shards/: rebuilding them from the same
    # source must be accepted, because only timestamps differ.
    from llm.data.prepare import prepare_data

    run = create_run(cfg)
    train(cfg, run, stop_at_step=5)
    prepare_data(cfg), train_tokenizer(cfg), write_shards(cfg)
    state, reason = train(cfg, run, resume=True)
    assert reason == "max_steps" and state.step == 40


def test_resume_refuses_changed_inputs(tiny_config, workspace, stories_path):
    from llm.data.prepare import prepare_data

    src = workspace / "stories_copy.jsonl"
    src.write_text(stories_path.read_text())
    edit_yaml(tiny_config, lambda d: d["data"]["source"].update(path=str(src)))
    cfg, _ = run_pipeline(tiny_config)
    run = create_run(cfg)
    train(cfg, run, stop_at_step=5)
    with open(src, "a") as f:
        f.write(json.dumps({"text": "A brand new story about a purple whale who sang every night."}) + "\n")
    prepare_data(cfg), train_tokenizer(cfg), write_shards(cfg)
    with pytest.raises(ProvenanceError, match="data"):
        train(cfg, run, resume=True)


def test_wall_time_stop(tiny_config):
    cfg, _ = run_pipeline(tiny_config, ["train.max_wall_time_s=0.001"])
    run = create_run(cfg)
    _, reason = train(cfg, run)
    assert reason == "wall_time" and find_latest_valid(run) is not None


def test_signal_stop(cfg, monkeypatch):
    real = loop.lr_at

    def lr_and_signal(step, t):
        if step == 2:  # during step 3
            os.kill(os.getpid(), signal.SIGTERM)
        return real(step, t)

    monkeypatch.setattr(loop, "lr_at", lr_and_signal)
    before = signal.getsignal(signal.SIGTERM)
    run = create_run(cfg)
    state, reason = train(cfg, run)
    assert reason == "signal" and state.step == 3 and _latest_step(run) == 3
    assert signal.getsignal(signal.SIGTERM) == before


def test_nan_guard(tiny_config, monkeypatch):
    cfg, _ = run_pipeline(tiny_config, ["train.ckpt_every=5"])
    real, calls = GPT.forward, {"n": 0}

    def nan_after_5(self, idx, targets=None):
        logits, loss = real(self, idx, targets)
        if targets is not None and self.training:
            calls["n"] += 1
            if calls["n"] > 5:
                loss = loss * float("nan")
        return logits, loss

    monkeypatch.setattr(GPT, "forward", nan_after_5)
    run = create_run(cfg)
    state, reason = train(cfg, run)
    assert reason == "nan" and state.step == 5 and _latest_step(run) == 5
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["type"] == "error" and rows[-1]["step"] == 6


def test_fp16_path_runs(tiny_config):
    if not torch.cuda.is_available():
        pytest.skip("fp16 + GradScaler needs a CUDA GPU (covered by the Kaggle smoke test)")
    cfg, _ = run_pipeline(tiny_config, ["train.device=cuda", "train.precision=fp16", "train.max_steps=5"])
    run = create_run(cfg)
    state, reason = train(cfg, run)
    assert reason == "max_steps" and all(torch.isfinite(torch.tensor(_losses(run))))


def test_cli_resume(cfg, workspace, tiny_config, capsys):
    assert cli.main(["train", str(tiny_config), "--stop-at-step", "4"]) == 0
    run = next((workspace / "runs").iterdir())
    assert cli.main(["resume", str(run), "--stop-at-step", "6"]) == 0
    assert "stop_at_step at step 6" in capsys.readouterr().out


def test_resume_refuses_changed_model_config_by_name(cfg):
    run = create_run(cfg)
    train(cfg, run, stop_at_step=5)
    edit_yaml(run / "config.yaml", lambda d: d["model"].update(n_layer=3))
    with pytest.raises(ConfigError, match="model.n_layer"):
        train(load_config(run / "config.yaml"), run, resume=True)


def test_wall_time_limit_is_per_session(tiny_config):
    # Each Kaggle session gets the full budget: a resumed session must make progress
    # even when the earlier sessions together already used more than the limit.
    cfg, _ = run_pipeline(tiny_config, ["train.max_wall_time_s=1.0", "train.max_steps=100000",
                                        "train.eval_every=100000", "train.ckpt_every=100000"])
    run = create_run(cfg)
    s1, r1 = train(cfg, run)
    assert r1 == "wall_time" and s1.step > 0 and s1.wall_time_s >= 1.0
    s2, r2 = train(cfg, run, resume=True)
    assert r2 == "wall_time" and s2.step > s1.step
