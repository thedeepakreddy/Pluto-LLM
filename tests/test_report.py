import warnings

import pytest

from llm import cli
from llm.report import build_report, compare_runs
from llm.runs import create_run
from llm.train.loop import train

SECTIONS = ["## Run card", "## Data card", "## Tokenizer card", "## Charts", "## Results",
            "## Sample progression", "## Resume log", "## Known limitations"]


@pytest.fixture
def run(prepared_tokenized):
    cfg, _ = prepared_tokenized
    r = create_run(cfg)
    train(cfg, r)
    return r


def test_report_sections(run):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = build_report(run)
    text = path.read_text()
    positions = [text.index(s) for s in SECTIONS]
    assert positions == sorted(positions)
    for c in ("loss", "bpb", "lr", "grad_norm", "throughput"):
        assert (run / "charts" / f"{c}.png").stat().st_size > 0
    assert "test-fixture" in text  # data license from the data card
    assert "Once upon a time, there was a little dog named" in text
    assert "| final val_loss |" in text


def test_report_warns_unfilled_limitations(run):
    with pytest.warns(UserWarning, match="limitations.md"):
        text = build_report(run).read_text()
    assert "Not yet written" in text
    (run / "limitations.md").write_text("# Known limitations\n\n- Tiny model on 40 synthetic stories.\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "Tiny model on 40" in build_report(run).read_text()


def test_report_includes_resume_log(prepared_tokenized):
    cfg, _ = prepared_tokenized
    r = create_run(cfg)
    train(cfg, r, stop_at_step=13)
    train(cfg, r, resume=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = build_report(r).read_text()
    resume_section = text[text.index("## Resume log"):text.index("## Known limitations")]
    assert "13" in resume_section


def test_compare_lists_config_diff(tiny_config, workspace):
    from conftest import run_pipeline

    cfg_a, _ = run_pipeline(tiny_config, ["train.max_steps=10", "train.eval_every=10"])
    a = create_run(cfg_a)
    train(cfg_a, a)
    from llm.config import load_config
    cfg_b = load_config(tiny_config, ["train.max_steps=10", "train.eval_every=10", "train.lr=0.001"])
    b = create_run(cfg_b)
    train(cfg_b, b)
    out = compare_runs(a, b, workspace / "cmp")
    text = out.read_text()
    assert "train.lr" in text and "train.max_steps" not in text
    assert (workspace / "cmp" / "loss.png").exists()


def test_cli_report_and_compare(run, workspace):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert cli.main(["report", str(run)]) == 0
        assert cli.main(["compare", str(run), str(run), "--out", str(workspace / "c")]) == 0
    assert (run / "report.md").exists() and (workspace / "c" / "comparison.md").exists()
