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
    assert cli.main(["eval-speech", str(run), "--n", "1"]) == 0
    assert "floor" not in json.loads((run / "speech_eval.json").read_text())


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
