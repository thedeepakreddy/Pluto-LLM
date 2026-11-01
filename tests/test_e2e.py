from pathlib import Path

import pytest

from llm import cli
from llm.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["m1-overfit", "m1-tinystories"])
def test_configs_load(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg.run.name == name


def test_tinystories_config_is_about_6m_params():
    from llm.model.gpt import expected_num_params

    cfg = load_config(ROOT / "configs" / "m1-tinystories.yaml")
    assert 5.5e6 < expected_num_params(cfg.model, cfg.tokenizer.vocab_size) < 6.5e6


@pytest.mark.slow
def test_full_pipeline_cli(workspace, tiny_config, capsys):
    c = str(tiny_config)
    for cmd in ("prepare-data", "train-tokenizer", "tokenize"):
        assert cli.main([cmd, c]) == 0
    assert cli.main(["train", c, "--set", "train.max_steps=40", "--stop-at-step", "20"]) == 0
    run = next((workspace / "runs").iterdir())
    assert cli.main(["resume", str(run)]) == 0
    capsys.readouterr()
    assert cli.main(["generate", str(run), "--prompt", "Once upon a time", "--max-new-tokens", "20",
                     "--seed", "1"]) == 0
    assert len(capsys.readouterr().out.strip()) > len("Once upon a time")
    with pytest.warns(UserWarning, match="limitations"):
        assert cli.main(["report", str(run)]) == 0
    report = (run / "report.md").read_text()
    assert "resumed from step 20" in report
