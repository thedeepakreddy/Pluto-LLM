import json
from pathlib import Path

import numpy as np
import pytest
from conftest import SPEECH_FAKE, edit_yaml, run_pipeline, setup_speech_config

from llm.config import load_config
from llm.manifest import manifest_checksum, read_manifest, verify_files
from llm.speech import build as build_mod
from llm.speech.build import build_speech_data, speech_dir
from llm.speech.fakes import FakeASR, FakeCodec, FakeTTS
from llm.speech.interfaces import Transcript

K = 8
SPEECH = SPEECH_FAKE


def _setup(tiny_config, speech=None, stories=None):
    return setup_speech_config(tiny_config, speech=speech, stories=stories)


@pytest.fixture
def cfg(tiny_config):
    return _setup(tiny_config)


def _build(cfg, **kw):
    return build_speech_data(cfg, FakeTTS(), FakeCodec(), FakeASR(), **kw)


def _index(d, split):
    p = Path(d) / f"{split}_index.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()]


def _check_contiguous(d, split):
    rows = _index(d, split)
    pos = 0
    for r in rows:
        assert r["offset"] == pos
        assert len(r["token_frames"]) == len(r["text_ids"])
        pos += r["frames"]
    codes = Path(d) / f"{split}_codes.bin"
    assert (codes.stat().st_size if codes.exists() else 0) == pos * K * 2
    return rows


def test_speech_dir(cfg):
    assert speech_dir(cfg) == Path("speech") / "tiny-speech-v1"


def test_build_outputs_and_manifest(cfg):
    d, complete = _build(cfg)
    assert complete and d == speech_dir(cfg)
    verify_files(d)
    m = read_manifest(d)
    assert m["kind"] == "speech"
    assert set(m["inputs"]) == {"data", "tokenizer"}
    ex = m["extra"]
    assert ex["voice"] == "bf_emma" and ex["frame_rate"] == 12.5 and ex["codebooks"] == 8
    assert (ex["tts"], ex["codec"], ex["asr"]) == ("fake", "fake", "fake")
    assert ex["splits"]["test"]["chunks"] == 3 and ex["splits"]["val"]["chunks"] == 3
    assert ex["splits"]["train"]["hours"] * 3600 >= 72
    for split in ("train", "val", "test"):
        rows = _check_contiguous(d, split)
        assert len(rows) == ex["splits"][split]["chunks"]
        assert sum(r["frames"] for r in rows) == ex["splits"][split]["frames"]
        for r in rows:
            assert r["id"].startswith(split + "-")
            assert r["seconds"] == pytest.approx(r["frames"] / 12.5)
    first = _index(d, "train")[0]
    codes = np.fromfile(d / "train_codes.bin", dtype=np.uint16).reshape(-1, K)
    assert codes.shape[0] > 0 and int(codes.max()) < 2048
    assert first["token_frames"] == sorted(first["token_frames"])
    assert (d / "stats.json").exists()
    wavs = sorted(p.name for p in (d / "samples").glob("*.wav"))
    assert wavs == sorted(f"{r['id']}.wav" for r in _index(d, "train")[:2])


def test_build_writes_valid_wav(cfg):
    import wave

    d, _ = _build(cfg)
    rid = _index(d, "train")[0]["id"]
    with wave.open(str(d / "samples" / f"{rid}.wav")) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        assert w.getnframes() > 0


def test_build_split_isolation(cfg):
    from llm.data.prepare import data_dir

    d, _ = _build(cfg)
    docs = {s: [json.loads(l)["text"] for l in (data_dir(cfg) / f"{s}.jsonl").read_text().splitlines()]
            for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        for r in _index(d, s):
            doc_i = int(r["id"].split("-")[1])
            assert r["id"].startswith(f"{s}-") and r["text"] in docs[s][doc_i]
    all_ids = [r["id"] for s in docs for r in _index(d, s)]
    assert len(all_ids) == len(set(all_ids))
    # sentence-level overlap between near-duplicate fixture stories is not leakage; whole docs must differ
    assert not (set(docs["train"]) & set(docs["val"])) and not (set(docs["train"]) & set(docs["test"]))


STORIES = [
    f"Mr. Smith had a {w} cat. He fed it every day. The cat purred loudly and slept all afternoon."
    if i % 2 == 0 else
    f"Lily had a {w} dog. She walked it every day. The dog barked happily and played all afternoon."
    for i, w in enumerate(["red", "blue", "green", "tiny", "huge", "soft", "fast", "slow", "old", "new",
                           "calm", "wild", "warm", "cold", "shy", "bold", "kind", "wise", "grey", "gold"])
]


def test_build_counts_timing_mismatch(tiny_config, tmp_path):
    stories = tmp_path / "mr.jsonl"
    stories.write_text("".join(json.dumps({"text": t}) + "\n" for t in STORIES))
    cfg = _setup(tiny_config, stories=stories)

    class MisterTTS(FakeTTS):
        def synthesize(self, text):
            return super().synthesize(text.replace("Mr.", "mister"))

    d, complete = build_speech_data(cfg, MisterTTS(), FakeCodec(), FakeASR())
    assert complete
    stats = json.loads((d / "stats.json").read_text())
    assert stats["drops"].get("timing_mismatch", 0) > 0
    for s in ("train", "val", "test"):
        assert all("Mr." not in r["text"] for r in _index(d, s))


def test_build_qa_drops_bad_audio(tiny_config):
    cfg = _setup(tiny_config, speech={"qa_fraction": 1.0})

    class WrongASR:
        def transcribe(self, audio, sample_rate):
            return Transcript(text="wrong words", words=[])

    d, complete = build_speech_data(cfg, FakeTTS(), FakeCodec(), WrongASR())
    assert complete  # sources exhausted
    stats = json.loads((d / "stats.json").read_text())
    assert stats["kept"] == 0 and stats["attempted"] > 0
    assert stats["drops"] == {"qa_wer": stats["attempted"]}
    assert sum(stats["drops"].values()) == stats["attempted"]
    assert not any((d / f"{s}_index.jsonl").read_text() for s in ("train", "val", "test"))


def test_build_drops_too_long(tiny_config):
    cfg = _setup(tiny_config, speech={"max_seconds": 5.0})
    d, complete = _build(cfg)
    stats = json.loads((d / "stats.json").read_text())
    assert complete
    for s in ("train", "val", "test"):
        assert all(r["seconds"] <= 5.0 for r in _index(d, s))
    assert stats["drops"].get("too_long", 0) + stats["kept"] == stats["attempted"]


class _Clock:
    """Advances one 'second' per reading, so a session budget of N allows about N chunks."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def test_build_resume_after_partial_write(cfg, monkeypatch):
    monkeypatch.setattr(build_mod, "_clock", _Clock())
    d, complete = _build(cfg, max_seconds_this_session=5)
    assert not complete
    assert not (d / "manifest.json").exists()
    n_before = sum(len(_index(d, s)) for s in ("train", "val", "test"))
    assert 0 < n_before
    split = next(s for s in ("test", "val", "train") if _index(d, s))
    with open(d / f"{split}_index.jsonl", "a") as f:
        f.write('{"id": "half-written", "tex')
    with open(d / f"{split}_codes.bin", "ab") as f:
        f.write(b"\x07" * 100)

    monkeypatch.setattr(build_mod, "_clock", _Clock())
    d2, complete = _build(cfg)
    assert complete and d2 == d
    verify_files(d)
    resumed = {}
    for s in ("train", "val", "test"):
        rows = _check_contiguous(d, s)
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))
        assert "half-written" not in ids
        resumed[s] = rows
    stats_resumed = json.loads((d / "stats.json").read_text())
    sum_resumed = manifest_checksum(d)

    import shutil

    shutil.rmtree(d)
    d3, complete = _build(cfg)
    assert complete
    for s in ("train", "val", "test"):
        assert _index(d3, s) == resumed[s]
    assert json.loads((d3 / "stats.json").read_text()) == stats_resumed
    assert manifest_checksum(d3) == sum_resumed


def test_build_resume_does_not_double_count_drops(tiny_config, tmp_path, monkeypatch):
    stories = tmp_path / "mr.jsonl"
    stories.write_text("".join(json.dumps({"text": t}) + "\n" for t in STORIES))
    cfg = _setup(tiny_config, stories=stories)

    class MisterTTS(FakeTTS):
        def synthesize(self, text):
            return super().synthesize(text.replace("Mr.", "mister"))

    run = lambda **kw: build_speech_data(cfg, MisterTTS(), FakeCodec(), FakeASR(), **kw)
    monkeypatch.setattr(build_mod, "_clock", _Clock())
    d, complete = run(max_seconds_this_session=6)
    assert not complete
    monkeypatch.setattr(build_mod, "_clock", _Clock())
    d, complete = run()
    assert complete
    resumed = json.loads((d / "stats.json").read_text())
    import shutil

    shutil.rmtree(d)
    d, _ = run()
    assert json.loads((d / "stats.json").read_text()) == resumed


def test_build_deterministic(cfg):
    d, _ = _build(cfg)
    first = manifest_checksum(d)
    import shutil

    shutil.rmtree(d)
    d, _ = _build(cfg)
    assert manifest_checksum(d) == first


def test_build_complete_is_idempotent(cfg):
    d, _ = _build(cfg)
    before = manifest_checksum(d)
    d2, complete = _build(cfg)
    assert complete and manifest_checksum(d2) == before


def test_build_requires_speech_section(tiny_config):
    cfg, _ = run_pipeline(tiny_config)
    with pytest.raises(ValueError, match="speech"):
        _build(cfg)


def test_build_refuses_changed_recipe(cfg, tiny_config, monkeypatch):
    monkeypatch.setattr(build_mod, "_clock", _Clock())
    _build(cfg, max_seconds_this_session=3)
    edit_yaml(tiny_config, lambda d: d["speech"].update(voice="af_heart"))
    from llm.manifest import ProvenanceError

    with pytest.raises(ProvenanceError):
        _build(load_config(tiny_config))


def test_cli_build_speech(tiny_config, capsys):
    from llm import cli

    cfg = _setup(tiny_config)
    assert cli.main(["build-speech", str(tiny_config)]) == 0
    out = capsys.readouterr().out
    assert "complete" in out.lower()
    assert (speech_dir(cfg) / "manifest.json").exists()


def test_cli_build_speech_partial(tiny_config, capsys):
    from llm import cli

    cfg = _setup(tiny_config)
    assert cli.main(["build-speech", str(tiny_config), "--max-hours-session", "0"]) == 0
    assert "incomplete" in capsys.readouterr().out.lower()
    assert not (speech_dir(cfg) / "manifest.json").exists()
