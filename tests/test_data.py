import json
import random

from llm import cli
from llm.config import load_config
from llm.data.clean import clean_docs, normalize_text
from llm.data.prepare import prepare_data
from llm.data.sources import iter_source_docs
from llm.data.split import assign_split
from llm.manifest import read_manifest, verify_files


def test_clean_counts_sum(stories_path, tiny_config):
    docs = list(iter_source_docs(load_config(tiny_config).data.source))
    kept, stats = clean_docs(docs, min_chars=20, max_chars=20000)
    assert stats.received == 43 == len(docs)
    assert stats.received == stats.kept + sum(stats.dropped.values())
    assert stats.dropped["exact_duplicate"] == 2 and stats.dropped["too_short"] == 1
    assert stats.kept == len(kept) == 40


def test_clean_drops_too_long():
    kept, stats = clean_docs(["a" * 50, "b" * 500], min_chars=5, max_chars=100)
    assert kept == ["a" * 50] and stats.dropped["too_long"] == 1


def test_clean_drops_lone_surrogate():
    kept, stats = clean_docs(["ok text that is long enough", "bad \ud800 text that is long"], 5, 1000)
    assert len(kept) == 1 and stats.dropped["invalid_utf8"] == 1


def test_normalize_nfc_and_newlines():
    assert normalize_text("é\r\nx\r ") == "é\nx"


def test_split_deterministic_and_disjoint():
    rng = random.Random(0)
    texts = ["".join(rng.choices("abcdefgh ", k=30)) for _ in range(1000)]
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    a = [assign_split(t, ratios) for t in texts]
    assert a == [assign_split(t, ratios) for t in texts]
    assert set(a) <= {"train", "val", "test"}
    assert 700 <= a.count("train") <= 900 and a.count("val") > 0 and a.count("test") > 0


def test_split_all_train():
    assert {assign_split(str(i), {"train": 1.0, "val": 0.0, "test": 0.0}) for i in range(500)} == {"train"}


def test_prepare_data_outputs(tiny_config):
    out = prepare_data(load_config(tiny_config))
    for s in ("train", "val", "test"):
        assert (out / f"{s}.jsonl").exists()
    verify_files(out)
    m = read_manifest(out)
    assert m["kind"] == "data" and m["extra"]["clean"]["kept"] == 40
    assert m["extra"]["splits"]["train"]["docs"] == 40
    rows = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 40 and all("text" in r for r in rows)
    report = (out / "data_report.md").read_text()
    assert "exact_duplicate" in report and "test-fixture" in report


def test_cli_prepare_data(tiny_config, workspace):
    assert cli.main(["prepare-data", str(tiny_config)]) == 0
    assert (workspace / "data" / "tiny-v1" / "manifest.json").exists()
