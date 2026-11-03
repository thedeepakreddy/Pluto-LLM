import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_speech_generate import _audio, _model, _never_end
from test_speech_train import _speech_cfg

from llm import cli
from llm.report import build_report, compare_runs
from llm.runs import create_run
from llm.speech.dataset import SpeechShards
from llm.speech.evaluate import (
    DEMO_SENTENCES,
    DEMO_SET_VERSION,
    ceiling_wer,
    evaluate_speech,
)
from llm.speech.fakes import FakeASR, FakeCodec, FakeTTS
from llm.speech.generate import SpeechSampling
from llm.speech.layout import special_ids
from llm.tokenizer.bpe import BPETokenizer
from llm.train.loop import train

SAMPLING = SpeechSampling(seed=7)
DEMO_SHA256 = "9417d0abfc1b950614ce4bef8b038292cc1d06a61f580386ea9e5bc5e2bc0fca"


@pytest.fixture(scope="module")
def tok(stories_texts):
    return BPETokenizer.train(stories_texts, vocab_size=300)


def _examples(texts):
    tts, codec = FakeTTS(), FakeCodec()
    out = []
    for t in texts:
        s = tts.synthesize(t)
        out.append((t, codec.encode(s.audio, s.sample_rate), s.audio))
    return out


class IdentityCodec(FakeCodec):
    """Decode returns the original FakeTTS audio (keyed by the codes' bytes)."""

    def __init__(self, examples):
        self.table = {c.tobytes(): a for _, c, a in examples}

    def decode(self, codes):
        return self.table[np.asarray(codes, dtype=np.int64).tobytes()]


def test_demo_set_frozen():
    assert DEMO_SET_VERSION == "speech-demo-v1"
    assert len(DEMO_SENTENCES) == 10 and len(set(DEMO_SENTENCES)) == 10
    assert hashlib.sha256(json.dumps(DEMO_SENTENCES).encode()).hexdigest() == DEMO_SHA256


def test_ceiling_with_fakes():
    ex = _examples(["the cat sat", "a big red ball"])
    pairs = [(t, c) for t, c, _ in ex]
    assert ceiling_wer(FakeCodec(), FakeASR(), pairs)["wer"] == 1.0
    r = ceiling_wer(IdentityCodec(ex), FakeASR(), pairs)
    assert r["wer"] == 0.0 and r["cer"] == 0.0 and r["n"] == 2


def test_evaluate_speech_keys_and_determinism(tok):
    m, audio = _model(tok)
    ex = [(t, c) for t, c, _ in _examples(["the cat sat on the mat", "a big red ball"])]
    a = evaluate_speech(m, tok, FakeCodec(), FakeASR(), ex, audio, SAMPLING)
    b = evaluate_speech(m, tok, FakeCodec(), FakeASR(), ex, audio, SAMPLING)
    assert {"wer", "cer", "duration_ratio", "rtf", "n", "per_item"} <= set(a)
    assert a["n"] == 2 and len(a["per_item"]) == 2
    assert {"text", "hypothesis", "wer", "cer", "frames", "ref_frames"} <= set(a["per_item"][0])
    for k in ("wer", "cer", "duration_ratio"):
        assert a[k] == b[k]
    assert a["per_item"] == b["per_item"]
    assert a["rtf"] > 0


def test_duration_ratio_flags_runaway(tok):
    m, audio = _model(tok)
    _never_end(m, special_ids(tok.vocab_size))
    ex = [(t, c) for t, c, _ in _examples(["the cat sat on the mat now"])]
    r = evaluate_speech(m, tok, FakeCodec(), FakeASR(), ex, audio, SAMPLING)
    assert r["duration_ratio"] > 1.5


@pytest.fixture
def run(tiny_config):
    cfg = _speech_cfg(tiny_config, {"max_steps": 2, "eval_every": 1000, "warmup_steps": 1,
                                    "micro_batch_size": 4, "tokens_per_step": 256})
    r = create_run(cfg)
    train(cfg, r)
    return r


def test_cli_eval_speech(run):
    assert cli.main(["eval-speech", str(run), "--n", "2", "--floor"]) == 0
    d = json.loads((run / "speech_eval.json").read_text())
    assert {"model", "ceiling", "floor"} <= set(d)
    assert d["model"]["n"] == 2 and d["floor"]["n"] == 2 and "wer" in d["ceiling"]
    assert cli.main(["eval-speech", str(run), "--n", "1"]) == 0   # different size: own file
    assert "floor" in json.loads((run / "speech_eval.json").read_text())
    assert "floor" not in json.loads((run / "speech_eval_n1.json").read_text())


def test_shards_text_accessor(run, tiny_config):
    from llm.config import load_config
    from llm.speech.build import speech_dir

    ds = SpeechShards(speech_dir(load_config(tiny_config)), "test")
    assert isinstance(ds.text(0), str) and ds.text(0)


def test_report_speech_section(run):
    assert "## Speech" not in build_report(run).read_text()
    (run / "speech_eval.json").write_text(json.dumps({
        "set": DEMO_SET_VERSION,
        "model": {"wer": 0.5, "cer": 0.3, "duration_ratio": 1.1, "rtf": 0.2, "n": 5, "per_item": []},
        "ceiling": {"wer": 0.05, "cer": 0.02, "n": 5},
        "floor": {"wer": 1.0, "cer": 0.9, "duration_ratio": 2.0, "rtf": 0.3, "n": 5, "per_item": []}}))
    text = build_report(run).read_text()
    assert "## Speech" in text
    assert "| ceiling |" in text and "| model |" in text and "| floor |" in text
    assert "0.500" in text and "RTF" in text


def test_compare_includes_wer(run, tmp_path):
    for name, w in (("a", 0.5), ("b", 0.25)):
        d = Path(run.parent / f"cmp-{name}")
        import shutil

        shutil.copytree(run, d)
        (d / "speech_eval.json").write_text(json.dumps({"model": {"wer": w}, "ceiling": {"wer": 0.0}}))
    out = compare_runs(run.parent / "cmp-a", run.parent / "cmp-b", tmp_path / "out").read_text()
    assert "speech WER" in out and "0.500" in out and "0.250" in out


def test_eval_speech_reuses_saved_phases_and_floor_n(run, monkeypatch):
    """An interrupted evaluation continues: phases saved for the same checkpoint and --n are reused,
    and --floor-n scores the slow untrained floor on fewer chunks."""
    import llm.speech.evaluate as ev

    assert cli.main(["eval-speech", str(run), "--n", "2"]) == 0
    first = json.loads((run / "speech_eval.json").read_text())
    calls = []
    real = ev.evaluate_speech
    monkeypatch.setattr(ev, "evaluate_speech", lambda *a, **k: calls.append(k.get("label")) or real(*a, **k))
    monkeypatch.setattr(ev, "ceiling_wer", lambda *a, **k: pytest.fail("ceiling should be reused"))
    assert cli.main(["eval-speech", str(run), "--n", "2", "--floor", "--floor-n", "1"]) == 0
    d = json.loads((run / "speech_eval.json").read_text())
    assert calls == ["floor"]                      # only the missing phase ran
    assert d["model"] == first["model"] and d["ceiling"] == first["ceiling"]
    assert d["floor"]["n"] == 1


def test_read_along_and_summary():
    from llm.speech.evaluate import read_along, summarize_items

    assert read_along("the cat sat", "the cat sat on the mat and more") == (0.0, 1.0)  # extra speech ignored
    p, c = read_along("the cat sat down", "the dog sat down")
    assert p == 0.25 and c == 0.25
    items = [{"text": "a b c", "hypothesis": "a b c x y", "duration_ratio": 2.0},
             {"text": " ".join(["w"] * 25), "hypothesis": "w", "duration_ratio": 0.1}]
    d = summarize_items(items)
    assert d["all"]["n"] == 2 and set(d["by_length_words"]) == {"1-20", "21-40"}
    assert d["by_length_words"]["1-20"]["prefix_wer"] == 0.0


def test_eval_speech_temperature_writes_separate_file(run):
    assert cli.main(["eval-speech", str(run), "--n", "1", "--temperature", "0"]) == 0
    other = run / "speech_eval_t0_k50.json"
    d = json.loads(other.read_text())
    assert d["sampling"] == {"temperature": 0.0, "top_k": 50}
    assert "diagnostics" in d["model"]
    assert cli.main(["speech-diagnose", str(other)]) == 0


def test_eval_records_text_placement_and_pacing_guard(run):
    """Per-item generation stats (words placed, frame limit) are recorded, and --max-pad-run
    writes its own file with the pacing guard in the sampling settings."""
    assert cli.main(["eval-speech", str(run), "--n", "2", "--max-pad-run", "3"]) == 0
    d = json.loads((run / "speech_eval_t0.8_k50_pad3.json").read_text())
    assert d["sampling"]["max_pad_run"] == 3
    it = d["model"]["per_item"][0]
    assert {"tokens_total", "tokens_placed", "tokens_forced", "hit_frame_limit"} <= set(it)
    assert it["tokens_placed"] <= it["tokens_total"]
    assert "hit_frame_limit_frac" in d["model"]["diagnostics"]["all"]


def test_eval_speech_text_temperature_tag(run):
    assert cli.main(["eval-speech", str(run), "--n", "1", "--text-temperature", "1"]) == 0
    d = json.loads((run / "speech_eval_t0.8_k50_tt1.json").read_text())
    assert d["sampling"]["text_temperature"] == 1.0
