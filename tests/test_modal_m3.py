"""The M3 Modal app imports and exposes the planned steps (skipped without `modal`)."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("modal")


def _load():
    path = Path(__file__).parent.parent / "modal_app" / "m3_train.py"
    spec = importlib.util.spec_from_file_location("m3_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_steps_and_functions():
    m = _load()
    assert m.CPU_STEPS == ("check", "prepare", "tokenizer", "shards")
    assert m.GPU_STEPS == ("bench", "train", "resume", "ab")
    for name in ("cpu_job", "gpu_job", "summary", "main"):
        assert hasattr(m, name)


def test_commands_are_built_from_fixed_steps():
    m = _load()
    assert m.gpu_command("bench", "configs/m3-pilot.yaml", "", "u/pluto-runs") == \
        "llm bench configs/m3-pilot.yaml"
    assert "--set train.hub_repo=u/pluto-runs" in m.gpu_command("train", "configs/m3-pilot.yaml", "",
                                                                 "u/pluto-runs")
    assert m.gpu_command("resume", "", "runs/abc", "u/pluto-runs") == "llm resume runs/abc"
    with pytest.raises(ValueError):
        m.gpu_command("train", "configs/x.yaml; rm -rf /", "", "u/pluto-runs")


def test_ab_step_variants():
    m = _load()
    names = [n for n, _ in m.AB_VARIANTS]
    assert names[0] == "base" and {"rope", "rmsnorm", "swiglu", "gqa", "window", "mtp", "all"} <= set(names)
    cmds = m.ab_commands("u/pluto-runs")
    assert len(cmds) == len(m.AB_VARIANTS)
    assert all(c.startswith("llm train configs/m3-ab-base.yaml") and "run.name=m3-ab-" in c for c in cmds)
    assert "--set model.mtp_heads=1" in cmds[names.index("mtp")]


def test_bench_sweep_command():
    m = _load()
    assert m.gpu_command("bench", "configs/m3-pilot.yaml", "", "u/p", sweep=True) == \
        "llm bench configs/m3-pilot.yaml --sweep"
