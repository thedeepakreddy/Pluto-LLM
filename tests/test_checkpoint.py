import json
import random

import numpy as np
import pytest
import torch
import yaml

from llm.config import ModelConfig, load_config
from llm.model.gpt import GPT
from llm.runs import create_run
from llm.train.checkpoint import (TrainState, find_latest_valid, get_rng_state, load_checkpoint,
                                  maybe_upload, save_checkpoint, set_rng_state)

MCFG = ModelConfig(n_layer=1, d_model=32, n_head=2, context_length=16)


def _setup(seed=0):
    torch.manual_seed(seed)
    m = GPT(MCFG, 300)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return m, opt


def _step(m, opt):
    x = torch.randint(0, 300, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    opt.step()
    opt.zero_grad()


def _save(run_dir, m, opt, step, keep_last=3, is_best=False):
    return save_checkpoint(run_dir, m, opt, None, TrainState(step=step), provenance={"x": "y"},
                           config={"a": 1}, keep_last=keep_last, is_best=is_best)


def test_create_run_files(tiny_config, workspace):
    cfg = load_config(tiny_config)
    run = create_run(cfg)
    assert run.parent == workspace / "runs" and run.name.startswith("tiny-")
    assert load_config(run / "config.yaml") == cfg
    info = json.loads((run / "run.json").read_text())
    assert "git_sha" in info and "hardware" in info and info["run_id"] == run.name
    assert (run / "limitations.md").read_text().startswith("<!-- TODO")
    assert (run / "checkpoints").is_dir() and (run / "samples").is_dir()
    assert create_run(cfg) != run  # unique even within the same second


def test_save_load_identical_outputs(tmp_path):
    m, opt = _setup()
    _step(m, opt), _step(m, opt)
    path = _save(tmp_path, m, opt, 2)
    m2, opt2 = _setup(seed=1)
    state, payload = load_checkpoint(path, m2, opt2)
    assert state.step == 2 and payload["config"] == {"a": 1} and payload["provenance"] == {"x": "y"}
    x = torch.randint(0, 300, (1, 16))
    assert torch.equal(m.eval()(x)[0], m2.eval()(x)[0])
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert all(torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"]) for k in s1)


def test_rng_restored():
    state = get_rng_state()
    a = (torch.rand(3), np.random.rand(3), random.random())
    set_rng_state(state)
    b = (torch.rand(3), np.random.rand(3), random.random())
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_rng_restored_from_checkpoint(tmp_path):
    m, opt = _setup()
    path = _save(tmp_path, m, opt, 1)
    a = torch.rand(3)
    load_checkpoint(path, m, opt)
    assert torch.equal(a, torch.rand(3))


def test_checkpoint_loads_with_weights_only(tmp_path):
    m, opt = _setup()
    path = _save(tmp_path, m, opt, 1)
    torch.load(path, weights_only=True)  # no arbitrary pickles


def test_keep_last_and_best(tmp_path):
    m, opt = _setup()
    for s in range(1, 6):
        _save(tmp_path, m, opt, s, keep_last=2, is_best=(s == 2))
    names = sorted(p.name for p in (tmp_path / "checkpoints").glob("*.pt"))
    assert names == ["best.pt", "step_00000004.pt", "step_00000005.pt"]
    assert load_checkpoint(tmp_path / "checkpoints" / "best.pt", m)[0].step == 2


def test_latest_json_points_to_newest(tmp_path):
    m, opt = _setup()
    _save(tmp_path, m, opt, 1), _save(tmp_path, m, opt, 2)
    assert json.loads((tmp_path / "checkpoints" / "latest.json").read_text()) == {"path": "step_00000002.pt", "step": 2}
    assert find_latest_valid(tmp_path).name == "step_00000002.pt"


def test_load_latest_skips_corrupt_checkpoint(tmp_path):
    m, opt = _setup()
    _save(tmp_path, m, opt, 1), _save(tmp_path, m, opt, 2)
    newest = tmp_path / "checkpoints" / "step_00000002.pt"
    newest.write_bytes(newest.read_bytes()[:10])
    with pytest.warns(UserWarning, match="step_00000002"):
        assert find_latest_valid(tmp_path).name == "step_00000001.pt"


def test_no_checkpoint_returns_none(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    assert find_latest_valid(tmp_path) is None


def test_no_tmp_files_left(tmp_path):
    m, opt = _setup()
    _save(tmp_path, m, opt, 1, is_best=True)
    assert not list((tmp_path / "checkpoints").glob("*.tmp"))


def test_required_hub_upload_failure_is_reported(tmp_path, monkeypatch):
    import huggingface_hub

    class BrokenApi:
        def create_repo(self, *args, **kwargs):
            raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "HfApi", BrokenApi)
    checkpoint = tmp_path / "step.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match="required Hub upload failed"):
        maybe_upload(tmp_path, checkpoint, "private/pluto-runs", required=True)


def test_cli_generate(prepared_tokenized, capsys):
    from llm import cli
    from llm.config import config_to_dict

    cfg, _ = prepared_tokenized
    run = create_run(cfg)
    torch.manual_seed(0)
    m = GPT(cfg.model, cfg.tokenizer.vocab_size)
    save_checkpoint(run, m, None, None, TrainState(step=1), {}, config_to_dict(cfg), 3, False)
    args = ["generate", str(run), "--prompt", "Once upon a time", "--max-new-tokens", "10", "--seed", "1"]
    assert cli.main(args) == 0
    out1 = capsys.readouterr().out
    assert out1.startswith("Once upon a time") and len(out1) > len("Once upon a time")
    assert cli.main(args) == 0 and capsys.readouterr().out == out1


def test_corrupt_latest_json_falls_back(tmp_path):
    m, opt = _setup()
    _save(tmp_path, m, opt, 1), _save(tmp_path, m, opt, 2)
    (tmp_path / "checkpoints" / "latest.json").write_text('{"pa')
    with pytest.warns(UserWarning, match="latest.json"):
        assert find_latest_valid(tmp_path).name == "step_00000002.pt"


def test_restore_cuda_rng_with_fewer_gpus(monkeypatch):
    restored = []
    state = get_rng_state()
    state["cuda"] = [torch.zeros(16, dtype=torch.uint8), torch.ones(16, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda s, device=0: restored.append(device))
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: (_ for _ in ()).throw(IndexError("1")))
    set_rng_state(state)
    assert restored == [0]
