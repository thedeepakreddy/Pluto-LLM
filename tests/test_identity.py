"""Creator identity (PCID) pipeline: statements, probes, mixing and recall.

Uses ONLY the fictional fixture in tests/fixtures/pcid_sample ("Ada Quill Example")."""
import json
import re
import shutil
import sys
import types
from pathlib import Path

import pytest
from conftest import FIXTURES, edit_yaml, setup_speech_config

from llm.config import ConfigError, load_config
from llm.identity.pcid import (
    Probe,
    build_probes,
    build_statements,
    identity_dir,
    load_pcid,
    prepare_identity,
)
from llm.shards import DataPosition
from llm.speech.layout import special_ids

PCID = FIXTURES / "pcid_sample"
ASPIRATION_SOURCES = ("pluto/long_term_vision.json", "creator_philosophy.json")


@pytest.fixture(scope="module")
def pcid():
    return load_pcid(PCID)


@pytest.fixture(scope="module")
def statements(pcid):
    return build_statements(pcid, ("current_location",), 0.2)


def _raw_text() -> str:
    return " ".join(p.read_text() for p in PCID.rglob("*") if p.is_file() and p.name != "README.md")


def test_statements_deterministic(pcid, statements):
    again = build_statements(load_pcid(PCID), ("current_location",), 0.2)
    assert again == statements
    assert len(statements) > 20
    assert {s.status for s in statements} <= {"creator_stated", "aspiration", "source-derived"}
    creator = [s for s in statements if s.fact == "creator_name"]
    assert len(creator) == 4 and len({s.text for s in creator}) == 4
    assert any(s.text == "Pluto was created by Ada Quill Example." for s in creator)
    qa = [s for s in statements if s.source == "training/creator_qa.jsonl"]
    assert qa and qa[0].text == "Question: Who created Pluto? Answer: Pluto was created by Ada Quill Example."


def test_statements_faithful(statements):
    """Every name-like word (capitalised) and every number in a statement appears verbatim in the
    source records: nothing invented."""
    raw = _raw_text()
    template_words = {"Question", "Answer", "The", "One", "A", "Why"}
    for s in statements:
        for w in re.findall(r"[A-Za-z0-9.'\-]+", s.text):
            w = w.strip(".'-")
            w = w.removesuffix("'s")
            if not w or w in template_words:
                continue
            if w[0].isupper() or any(c.isdigit() for c in w):
                assert w in raw, f"{w!r} in {s.text!r} is not in the source records"


def test_aspirations_stay_goals(statements):
    asp = [s for s in statements if s.source in ASPIRATION_SOURCES]
    assert len(asp) >= 4
    for s in asp:
        assert "wants" in s.text or "aims" in s.text, s.text
    for s in statements:
        if s.source == "pluto/long_term_vision.json":
            assert s.status == "aspiration"
    goals = [s.text for s in asp if "wants Pluto to" in s.text]
    assert "Pluto's creator wants Pluto to hold a spoken conversation." in goals


def test_exclude_fields(pcid, statements):
    assert all("Harborview" not in s.text for s in statements)
    assert all("Testville" not in s.text for s in statements)
    # excluding another field removes it too
    no_origin = build_statements(pcid, ("current_location", "origin_country"), 0.2)
    assert any("Freedonia" in s.text for s in statements)
    assert all("Freedonia" not in s.text for s in no_origin)


def test_holdout_keeps_each_fact_trainable(pcid, statements):
    by_fact: dict[str, list] = {}
    for s in statements:
        by_fact.setdefault(s.fact, []).append(s)
    assert all(any(not s.holdout for s in v) for v in by_fact.values())
    assert any(s.holdout for s in statements)
    # even at a huge holdout fraction every fact keeps a training phrasing
    heavy = build_statements(pcid, ("current_location",), 0.99)
    facts = {s.fact for s in heavy}
    assert all(any(s.fact == f and not s.holdout for s in heavy) for f in facts)


def test_probes_from_records(pcid):
    probes = {p.fact: p for p in build_probes(pcid)}
    assert probes["creator_name"] == Probe("Pluto was created by", ("ada",), "creator_name")
    assert probes["naming"].prompt == "Pluto is named after"
    assert set(probes["naming"].expected) == {"universe", "space", "planet"}
    assert probes["education"].prompt == "Pluto's creator studied"
    assert set(probes["education"].expected) == {"mathematics", "bsc", "machine learning", "certificate"}
    assert probes["projects"].prompt == "One of the creator's AI projects is"
    assert set(probes["projects"].expected) == {"lumen", "tidewatch"}  # Inkpot is not an AI project


def test_identity_config_requires_one_source(tiny_config):
    def neither(d):
        d["identity"] = {"name": "pcid"}

    edit_yaml(tiny_config, neither)
    with pytest.raises(ConfigError, match="exactly one"):
        load_config(tiny_config)

    def both(d):
        d["identity"] = {"path": str(PCID), "hf_dataset": "me/pcid"}

    edit_yaml(tiny_config, both)
    with pytest.raises(ConfigError, match="exactly one"):
        load_config(tiny_config)

    def ok(d):
        d["identity"] = {"path": str(PCID)}

    edit_yaml(tiny_config, ok)
    cfg = load_config(tiny_config)
    assert cfg.identity.exclude_fields == ("current_location",)
    assert cfg.identity.text_every == 50 and cfg.identity.speech_every == 20


def _identity_cfg(tiny_config, **extra):
    def fn(d):
        d["identity"] = {"path": str(PCID), **extra}

    edit_yaml(tiny_config, fn)
    return load_config(tiny_config)


def test_prepare_identity_manifest(tiny_config):
    cfg = _identity_cfg(tiny_config)
    out = prepare_identity(cfg)
    assert out == identity_dir(cfg) == Path("identity/pcid-v1")
    man = json.loads((out / "manifest.json").read_text())
    assert set(man["files"]) == {"statements.jsonl", "probes.jsonl", "identity_report.md"}
    rows = [json.loads(l) for l in (out / "statements.jsonl").read_text().splitlines()]
    assert rows and set(rows[0]) == {"text", "source", "status", "fact", "holdout"}
    ex = man["extra"]
    assert sum(ex["by_holdout"].values()) == len(rows)
    assert ex["by_source"]["identity.json"] >= 4 and "aspiration" in ex["by_status"]
    from llm.manifest import manifest_checksum

    first = manifest_checksum(out)
    shutil.rmtree(out)
    assert manifest_checksum(prepare_identity(cfg)) == first  # deterministic
    assert "Harborview" not in (out / "identity_report.md").read_text()


def test_prepare_identity_from_hf(tiny_config, monkeypatch, tmp_path):
    calls = []

    def snapshot_download(repo_id, repo_type=None, **kw):
        calls.append((repo_id, repo_type))
        dst = tmp_path / "hf_snapshot"
        shutil.copytree(PCID, dst, dirs_exist_ok=True)
        return str(dst)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    cfg = _identity_cfg(tiny_config)
    local = prepare_identity(cfg)
    shutil.rmtree(local)
    cfg_hf = load_config(tiny_config, ["identity.path=null", "identity.hf_dataset=me/private-pcid"])
    out = prepare_identity(cfg_hf)
    assert calls == [("me/private-pcid", "dataset")]
    assert (out / "statements.jsonl").exists()


def test_cli_prepare_identity(tiny_config, capsys):
    from llm import cli

    _identity_cfg(tiny_config)
    assert cli.main(["prepare-identity", str(tiny_config)]) == 0
    assert (Path("identity/pcid-v1") / "manifest.json").exists()


# ---------------------------------------------------------------- speech data + mixing


@pytest.fixture
def speech_identity_cfg(tiny_config):
    from llm.speech.build import build_speech_data
    from llm.speech.fakes import FakeASR, FakeCodec, FakeTTS

    setup_speech_config(tiny_config, speech={"max_seconds": 4.5}, audio={"layout": "aligned"})
    cfg = _identity_cfg(tiny_config)
    _, complete = build_speech_data(cfg, FakeTTS(), FakeCodec(), FakeASR())
    assert complete
    return cfg


def test_build_speech_identity_splits(speech_identity_cfg):
    from llm.speech.build import speech_dir
    from llm.speech.dataset import SpeechShards

    cfg = speech_identity_cfg
    d = speech_dir(cfg)
    man = json.loads((d / "manifest.json").read_text())
    assert "identity" in man["inputs"]
    rows = [json.loads(l) for l in (identity_dir(cfg) / "statements.jsonl").read_text().splitlines()]
    train = SpeechShards(d, "identity")
    held = SpeechShards(d, "identity_eval")
    texts = {train.text(i) for i in range(len(train))} | {held.text(i) for i in range(len(held))}
    assert len(train) > 0 and len(held) > 0
    assert texts <= {r["text"] for r in rows}
    assert {held.text(i) for i in range(len(held))} <= {r["text"] for r in rows if r["holdout"]}
    assert man["extra"]["splits"]["identity"]["chunks"] == len(train)


def test_identity_mixing_schedule(speech_identity_cfg):
    from llm.shards import TokenChunks, shard_dir
    from llm.speech.build import speech_dir
    from llm.speech.dataset import SpeechShards, next_speech_batch

    cfg = speech_identity_cfg
    sp = special_ids(cfg.tokenizer.vocab_size)
    d = speech_dir(cfg)
    ds = SpeechShards(d, "train")
    id_speech = SpeechShards(d, "identity")
    id_text = [[5, 6, 7], [8, 9]]
    chunks = TokenChunks(shard_dir(cfg), "train", cfg.model.context_length - 1)
    id_speech_texts = {tuple(id_speech[i][0]) for i in range(len(id_speech))}

    def kind(step):
        b, new = next_speech_batch(ds, DataPosition(), 2, 7, cfg.audio, sp, step, chunks,
                                   identity=cfg.identity, identity_text=id_text,
                                   identity_speech=id_speech)
        if not b.code_mask.any():
            first = b.text[0, 1:].tolist()
            ident = any(first[: len(t)] == t and all(x == sp.pad_text for x in first[len(t):])
                        for t in id_text)
            return ("identity_text" if ident else "text"), new
        texts = [r for r in b.text.tolist()]
        speech_ids = {tuple(t for t in row if t < sp.bos) for row in texts}
        return ("identity_speech" if speech_ids <= id_speech_texts else "speech"), new

    for step in (49, 99):
        k, new = kind(step)
        assert k == "identity_text" and new == DataPosition()
    for step in (19, 39, 59):
        k, new = kind(step)
        assert k == "identity_speech" and new == DataPosition()
    assert kind(9)[0] == "text"
    assert kind(3)[0] == "speech"
    # without identity sources the old schedule is unchanged
    b, _ = next_speech_batch(ds, DataPosition(), 2, 7, cfg.audio, sp, 18, chunks)  # 19 would be text
    assert b.code_mask.any()
    b, _ = next_speech_batch(ds, DataPosition(), 2, 7, cfg.audio, sp, 49, chunks)
    assert not b.code_mask.any()  # plain text_every step


def test_trainer_and_eval_with_identity(speech_identity_cfg, tiny_config):
    from llm.report import build_report
    from llm.runs import create_run
    from llm.speech.evaluate import run_eval_speech
    from llm.train.loop import train

    cfg = load_config(tiny_config, ["train.max_steps=4", "train.warmup_steps=1", "train.eval_every=1000", "train.ckpt_every=4",
                                    "train.micro_batch_size=4", "train.tokens_per_step=256",
                                    "identity.text_every=2", "identity.speech_every=3"])
    run = create_run(cfg)
    _, reason = train(cfg, run)
    assert reason == "max_steps"
    path = run_eval_speech(run, n=2)
    d = json.loads(path.read_text())
    ident = d["identity"]
    assert 0.0 <= ident["text"]["recall"] <= 1.0
    assert set(ident["text"]["per_fact"]) >= {"creator_name", "naming", "education", "projects"}
    assert "speech" in ident
    report = build_report(run).read_text()
    assert "Identity recall" in report


@pytest.mark.slow
def test_identity_recall_text_with_memorising_model(tiny_config):
    """A tiny SpeechGPT overfit on the train statements (held-out phrasings excluded) recalls the
    creator's name through ``identity_recall`` (text mode) from the canonical probe.

    Held-out *paraphrase* recall is measured and reported but not asserted: a from-scratch
    2-layer/64-d model generalised "Pluto was built by" -> "Ada" in only 1 of 3 seeds (see the
    task report), so asserting it would make the test flaky."""
    import torch
    from conftest import run_pipeline

    from llm.config import AudioConfig
    from llm.model.voice import SpeechGPT
    from llm.speech.dataset import collate
    from llm.speech.evaluate import identity_recall
    from llm.speech.layout import build_text_only
    from llm.tokenizer.bpe import BPETokenizer, tokenizer_dir

    cfg, _ = run_pipeline(tiny_config)
    tok = BPETokenizer.load(tokenizer_dir(cfg))
    pcid = load_pcid(PCID)
    sts = build_statements(pcid, ("current_location",), 0.2)
    texts = [s.text for s in sts if not s.holdout]
    held = [s for s in sts if s.fact == "creator_name" and s.holdout]
    assert held and all(s.text not in texts for s in held)
    canonical = [p for p in build_probes(pcid) if p.fact == "creator_name"]
    paraphrases = [Probe(s.text.split("Ada Quill Example")[0].strip(), ("ada",), "creator_name_heldout")
                   for s in held]
    sp = special_ids(tok.vocab_size)
    torch.manual_seed(0)
    model = SpeechGPT(cfg.model, AudioConfig(), tok.vocab_size + 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    for step in range(800):
        # statements packed in pairs, so each phrasing is seen at many positions
        g = torch.Generator().manual_seed(step)
        a = torch.randperm(len(texts), generator=g)[:16].tolist()
        b = torch.randperm(len(texts), generator=g)[:16].tolist()
        ex = [build_text_only(tok.encode(texts[i] + " " + texts[j])[: cfg.model.context_length - 1], sp)
              for i, j in zip(a, b)]
        loss = model(collate(ex, sp)).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    r = identity_recall(model, tok, canonical + paraphrases, "text")
    assert r["per_fact"]["creator_name"] >= 0.9, r["items"]
    assert "creator_name_heldout" in r["per_fact"]  # measured (reported), not asserted
    # an untrained model does not recall the name: the probe is not trivially satisfied
    torch.manual_seed(0)
    fresh = SpeechGPT(cfg.model, AudioConfig(), tok.vocab_size + 4)
    assert identity_recall(fresh, tok, canonical, "text")["recall"] == 0.0


def test_private_outputs_are_git_ignored():
    """Prepared identity statements and synthesised speech (creator data) must never be committed."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    paths = ["identity/pcid-v1/statements.jsonl", "speech/x-v1/train_codes.bin"]
    r = subprocess.run(["git", "check-ignore", *paths], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.split() == paths, r.stdout + r.stderr


def test_prepare_identity_deterministic_across_runs(tiny_config):
    from llm.manifest import manifest_checksum

    cfg = _identity_cfg(tiny_config)
    out = prepare_identity(cfg)
    first = manifest_checksum(out)
    files = {f: (out / f).read_bytes() for f in ("statements.jsonl", "probes.jsonl", "identity_report.md")}
    shutil.rmtree(out)
    out2 = prepare_identity(cfg)
    assert manifest_checksum(out2) == first
    assert {f: (out2 / f).read_bytes() for f in files} == files
