"""train.init_from for text runs (M3): weights only, fresh optimizer, provenance recorded."""
import dataclasses
import json

import pytest
import torch

from conftest import run_pipeline
from llm.config import load_config
from llm.runs import create_run
from llm.train.checkpoint import find_latest_valid
from llm.train.loop import TextTrainer, train


def _with(cfg, **train_kw):
    return dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **train_kw))


def _model(cfg, **kw):
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, **kw))


@pytest.fixture
def source_ckpt(tiny_config):
    cfg, _ = run_pipeline(tiny_config, ["model.pos=rope", "train.max_steps=10", "train.ckpt_every=10"])
    run = create_run(cfg)
    train(cfg, run)
    return cfg, find_latest_valid(run)


def test_init_from_loads_weights(source_ckpt):
    cfg, ckpt = source_ckpt
    cfg2 = _with(cfg, init_from=str(ckpt))
    run = create_run(cfg2)
    tr = TextTrainer(cfg2, run)
    tr.on_fresh_start()
    src = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    for k, v in tr.model.state_dict().items():
        assert torch.equal(v.cpu(), src[k]), k
    assert tr.state.step == 0 and not tr.optimizer.state
    info = json.loads((run / "run.json").read_text())["init_from"]
    assert info["path"] == str(ckpt) and len(info["sha256"]) == 64


def test_init_from_rope_context_extension(source_ckpt):
    cfg, ckpt = source_ckpt
    cfg2 = _with(_model(cfg, context_length=128), init_from=str(ckpt), tokens_per_step=1024)
    tr = TextTrainer(cfg2, create_run(cfg2))
    tr.on_fresh_start()  # no position table under RoPE: loads at the longer context


def test_init_from_learned_pos_mismatch_errors(tiny_config):
    cfg, _ = run_pipeline(tiny_config, ["train.max_steps=10", "train.ckpt_every=10"])
    run = create_run(cfg)
    train(cfg, run)
    cfg2 = _with(_model(cfg, context_length=128), init_from=str(find_latest_valid(run)), tokens_per_step=1024)
    tr = TextTrainer(cfg2, create_run(cfg2))
    with pytest.raises(ValueError, match="pos_emb"):
        tr.on_fresh_start()


def test_resume_ignores_init_from(source_ckpt, tmp_path):
    cfg, ckpt = source_ckpt
    copy = tmp_path / "init.pt"
    copy.write_bytes(ckpt.read_bytes())
    cfg2 = _with(cfg, init_from=str(copy), max_steps=20, ckpt_every=10)
    run = create_run(cfg2)
    train(cfg2, run, stop_at_step=10)
    copy.unlink()  # resume must not need the init checkpoint any more
    state, _ = train(cfg2, run, resume=True)
    assert state.step == 20
