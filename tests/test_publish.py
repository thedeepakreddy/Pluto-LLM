"""Public results export must never leak creator identity (fictional fixture only)."""
import json
from pathlib import Path

import pytest
import yaml
from conftest import FIXTURES

from llm.identity.pcid import build_probes, load_pcid
from llm.publish import export_public_results

PCID = FIXTURES / "pcid_sample"
HF_ID = "someone-example/private-identity-xyz"


def _make_run(root: Path, with_identity: bool) -> Path:
    run = root / "runs" / "m2-aligned-x"
    (run / "charts").mkdir(parents=True)
    (run / "charts" / "loss.png").write_bytes(b"\x89PNG fake")
    (run / "samples" / "step_00000010").mkdir(parents=True)
    (run / "samples" / "step_00000010" / "0.wav").write_bytes(b"RIFF")
    cfg = {"run": {"name": "m2-aligned"}, "train": {"max_steps": 8}}
    ev = {"model": {"wer": 0.5, "cer": 0.3, "duration_ratio": 1.0, "rtf": 0.2, "n": 2}}
    report = ["# Report", "", "## Speech", "", "| system | WER |", "| model | 0.500 |"]
    if with_identity:
        cfg["identity"] = {"hf_dataset": HF_ID}
        probes = build_probes(load_pcid(PCID))
        items = [{"prompt": p.prompt, "expected": list(p.expected), "fact": p.fact,
                  "output": p.expected[0], "hit": True} for p in probes]
        facts = {p.fact: 1.0 for p in probes}
        ev["identity"] = {"n_probes": len(items),
                          "text": {"recall": 0.75, "n": len(items), "per_fact": facts, "items": items},
                          "speech": {"recall": 0.25, "n": len(items), "per_fact": facts, "items": items}}
        report += ["", "### Identity recall", "", "| fact | text |", "|---|---:|"]
        report += [f"| {f} | 1.00 |" for f in facts] + ["| **all** | 0.75 |"]
    report += ["", "Demo samples (`speech-demo-v1`, step 10):", "", "- [0.wav](samples/step_00000010/0.wav)"]
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run / "speech_eval.json").write_text(json.dumps(ev))
    (run / "report.md").write_text("\n".join(report) + "\n")
    (run / "metrics.jsonl").write_text('{"step": 1, "loss": 3.0}\n')
    (run / "run.json").write_text("{}")
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "best.pt").write_bytes(b"weights")
    return run


def _all_text(dest: Path) -> str:
    return "\n".join(p.read_bytes().decode("utf-8", "ignore") for p in dest.rglob("*") if p.is_file())


def test_export_redacts_identity(tmp_path):
    run = _make_run(tmp_path, True)
    secret = set()
    for p in build_probes(load_pcid(PCID)):
        secret |= set(p.expected)
    assert secret
    files = export_public_results(run, tmp_path / "out")
    dest = tmp_path / "out"
    assert files and all(f.is_relative_to(dest) for f in files)
    text = _all_text(dest).lower()
    for s in secret | {HF_ID, "Ada Quill", "per_fact", "creator_name"}:
        assert s.lower() not in text, s
    ev = json.loads((dest / "speech_eval.json").read_text())
    assert ev["identity"] == {"n_probes": ev["identity"]["n_probes"], "text": {"recall": 0.75},
                              "speech": {"recall": 0.25}}
    assert "identity" not in yaml.safe_load((dest / "config.yaml").read_text())
    rep = (dest / "report.md").read_text()
    assert "Identity recall: text 75%, speech 25% (details private)" in rep
    assert "Demo samples" in rep and "## Speech" in rep
    assert not (dest / "checkpoints").exists()
    assert (dest / "charts" / "loss.png").exists() and (dest / "samples" / "step_00000010" / "0.wav").exists()


def test_export_without_identity_unchanged(tmp_path):
    run = _make_run(tmp_path, False)
    dest = tmp_path / "out"
    export_public_results(run, dest)
    for name in ("report.md", "speech_eval.json", "config.yaml", "metrics.jsonl", "run.json"):
        assert (dest / name).read_bytes() == (run / name).read_bytes()


def test_export_final_scan_raises(tmp_path):
    run = _make_run(tmp_path, True)
    (run / "limitations.md").write_text(f"trained on {HF_ID}\n")
    with pytest.raises(ValueError, match="private identity"):
        export_public_results(run, tmp_path / "out")
    assert not (tmp_path / "out" / "limitations.md").exists()
