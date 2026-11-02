import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import edit_yaml, run_pipeline

from llm import cli
from llm.config import config_diff, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
M2 = ["m2-stage0", "m2-warmstart", "m2-speech-aligned", "m2-speech-baseline"]


def _raw(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


@pytest.mark.parametrize("name", M2)
def test_m2_configs_load(name):
    cfg = load_config(CONFIGS / f"{name}.yaml")
    assert cfg.identity is None  # no committed creator identity
    m1 = _raw("m1-tinystories")
    raw = _raw(name)
    for k in ("data", "tokenizer", "shards"):
        assert raw[k] == m1[k]  # M1 tokenizer and shards are reused
    m = cfg.model
    assert (m.n_layer, m.d_model, m.n_head, m.context_length) == (8, 384, 6, 640)


def test_m2_aligned_vs_baseline_differ_only_in_layout():
    a, b = _raw("m2-speech-aligned"), _raw("m2-speech-baseline")
    assert config_diff(a, b) == ["audio.layout", "run.name"]
    ca, cb = load_config(CONFIGS / "m2-speech-aligned.yaml"), load_config(CONFIGS / "m2-speech-baseline.yaml")
    assert (ca.audio.layout, cb.audio.layout) == ("aligned", "text_then_audio")
    assert (ca.run.name, cb.run.name) == ("m2-aligned", "m2-baseline")
    t = ca.train
    assert (t.micro_batch_size, t.tokens_per_step, t.max_steps, t.lr, t.warmup_steps, t.eval_every) == (
        16, 40960, 6000, 6e-4, 300, 500)
    sp = ca.speech
    assert (sp.name, sp.version, sp.voice, sp.hours) == ("tinystories-speech", 1, "bf_emma", 200)
    assert ca.audio.acoustic_delay == 1 and ca.audio.init_from


def test_m2_warmstart_and_stage0():
    w = load_config(CONFIGS / "m2-warmstart.yaml")
    assert w.speech is None and w.audio is None and w.run.name == "m2-warmstart"
    t = w.train
    assert (t.max_steps, t.tokens_per_step, t.micro_batch_size, t.lr, t.warmup_steps) == (
        2400, 81920, 32, 1e-3, 200)
    assert config_diff(_raw("m2-speech-aligned"), _raw("m2-stage0")) == [
        "run.name", "speech.hours", "speech.test_chunks", "speech.val_chunks", "train.ckpt_every",
        "train.eval_every", "train.max_steps", "train.warmup_steps"]
    s = load_config(CONFIGS / "m2-stage0.yaml")
    assert (s.speech.hours, s.speech.val_chunks, s.speech.test_chunks, s.train.max_steps) == (0.3, 20, 20, 50)


TINY_MODEL = {"n_layer": 2, "d_model": 64, "n_head": 2, "context_length": 64}


@pytest.mark.slow
def test_m2_pipeline_cli(workspace, tiny_config, capsys):
    """build-speech -> warm-start text train -> aligned train with init_from, interrupted -> resume
    -> eval-speech -> speak --codes-only -> report (with a Speech section). Fake components only."""

    def fn(d):
        d["data"]["split"] = {"train": 0.6, "val": 0.2, "test": 0.2}
        d["model"] = dict(TINY_MODEL)
        d["train"].update({"max_steps": 6, "eval_every": 3, "ckpt_every": 2, "micro_batch_size": 4,
                           "tokens_per_step": 256, "warmup_steps": 1})
        d["run"]["name"] = "tiny-warm"

    edit_yaml(tiny_config, fn)
    run_pipeline(tiny_config)
    c = str(tiny_config)
    assert cli.main(["train", c]) == 0
    warm = next((workspace / "runs").glob("tiny-warm-*"))
    best = warm / "checkpoints" / "best.pt"
    assert best.exists()

    speech = workspace / "speech.yaml"
    d = yaml.safe_load(tiny_config.read_text())
    d["speech"] = {"name": "tiny-speech", "version": 1, "tts": "fake", "codec": "fake", "asr": "fake",
                   "hours": 0.02, "val_chunks": 3, "test_chunks": 3, "qa_fraction": 0.0, "keep_wavs": 2,
                   "max_seconds": 4.5}
    d["audio"] = {"layout": "aligned", "init_from": str(best)}
    d["run"]["name"] = "tiny-aligned"
    d["train"]["max_steps"] = 8
    speech.write_text(yaml.safe_dump(d, sort_keys=False))
    s = str(speech)
    assert cli.main(["build-speech", s]) == 0
    assert "complete" in capsys.readouterr().out
    assert cli.main(["train", s, "--stop-at-step", "4"]) == 0
    run = next((workspace / "runs").glob("tiny-aligned-*"))
    assert json.loads((run / "run.json").read_text())["warm_start"]["path"] == str(best)
    assert cli.main(["resume", str(run)]) == 0
    assert "stopped: max_steps at step 8" in capsys.readouterr().out
    assert cli.main(["eval-speech", str(run), "--n", "2", "--floor"]) == 0
    ev = json.loads((run / "speech_eval.json").read_text())
    assert "floor" in ev
    out = workspace / "codes.npy"
    assert cli.main(["speak", str(run), "--text", "Once upon a time", "--out", str(out),
                     "--codes-only", "--seed", "1"]) == 0
    assert np.load(out).shape[1] == 8
    with pytest.warns(UserWarning):
        assert cli.main(["report", str(run)]) == 0
    report = (run / "report.md").read_text()
    assert "## Speech" in report and "resumed from step 4" in report
