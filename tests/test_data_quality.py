"""M3 data quality: heuristics, normalised dedup, decontamination, near-dedup, source checks."""
import json

import pytest

from conftest import tiny_config_dict
from llm.config import load_config_dict
from llm.data import benchmarks
from llm.data.prepare import prepare_data
from llm.data.quality import Decontaminator, NearDedup, fingerprint, heuristic_reason
from llm.manifest import read_manifest

GOOD = ("The cat sat on the mat and looked at the birds that were singing in the tall tree near "
        "the old house, and then it went to sleep with a happy face in the warm afternoon sun.")


def test_heuristics():
    assert heuristic_reason(GOOD) is None
    assert heuristic_reason("Kaufen Sie jetzt billig online " * 10) == "not_english"
    assert heuristic_reason("\n".join(["Click here to read the rest of the story and more"] * 10)) == "repetitive"
    assert heuristic_reason(GOOD + " " + "# " * 200) == "low_quality"


def test_fingerprint_normalises_case_and_space():
    assert fingerprint("Hello  World\n") == fingerprint("hello world")
    assert fingerprint("hello world") != fingerprint("hello there world")


def test_decontaminator():
    item = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
            "clips in May. How many clips did Natalia sell altogether in April and May?")
    d = Decontaminator([item])
    assert d.contaminated("Here is a puzzle. " + item + " Think about it.")
    assert not d.contaminated(GOOD)


def test_near_dedup():
    nd = NearDedup()
    long = " ".join(f"word{i} is here" for i in range(200))
    assert not nd.is_duplicate(long)
    assert nd.is_duplicate(long.replace("word3 is", "word3 was"))  # one small edit
    assert not nd.is_duplicate(GOOD)


def _jsonl(path, docs):
    path.write_text("".join(json.dumps(d) + "\n" for d in docs))
    return str(path)


def _cfg(workspace, stories_path, sources, **data):
    d = tiny_config_dict(stories_path)
    d["data"].pop("source")
    d["data"].update({"sources": sources, "root": str(workspace / "data"), **data})
    return load_config_dict(d)


def _src(name, path, **kw):
    return {"name": name, "type": "local_jsonl", "path": path, "license": "test-fixture", **kw}


def test_prepare_quality_pipeline(workspace, stories_path, monkeypatch):
    bench = "What is the capital city of the country of France and why is it famous in the world"
    monkeypatch.setitem(benchmarks.BENCHMARKS, "fake", lambda: iter([bench]))
    docs = [{"text": GOOD}, {"text": GOOD.upper()},  # exact dup after normalisation
            {"text": "Kaufen Sie jetzt billig online " * 10},
            {"text": "Quiz time. " + bench + "? " + GOOD},
            {"text": GOOD.replace("tall tree", "big tree") + " " + GOOD},
            {"text": GOOD + " " + GOOD.replace("cat", "dog")}]
    code = [{"text": "def f(x):\n    return x + 1  # tiny python function for tests\n"}]
    cfg = _cfg(workspace, stories_path,
               [_src("web", _jsonl(workspace / "w.jsonl", docs)),
                _src("code", _jsonl(workspace / "c.jsonl", code), prose=False)],
               quality_filters=True, decontaminate=["fake"], near_dedup=True)
    m = read_manifest(prepare_data(cfg))["extra"]
    dropped = m["clean"]["dropped"]
    assert dropped["exact_duplicate"] == 1 and dropped["not_english"] == 1
    assert dropped["contaminated"] == 1
    assert m["sources"][1]["kept"] == 1  # code skips prose heuristics
    assert m["clean"]["received"] == m["clean"]["kept"] + sum(dropped.values())


def test_chunk_chars_splits_long_docs(workspace, stories_path):
    book = "\n\n".join(f"Paragraph {i}. " + GOOD for i in range(30))
    cfg = _cfg(workspace, stories_path, [_src("book", _jsonl(workspace / "b.jsonl", [{"text": book}]),
                                              chunk_chars=1000)])
    m = read_manifest(prepare_data(cfg))["extra"]
    assert m["sources"][0]["received"] == 1 and m["clean"]["kept"] > 4


def test_bad_score_field_errors(workspace, stories_path):
    cfg = _cfg(workspace, stories_path, [_src("a", _jsonl(workspace / "a.jsonl", [{"text": GOOD, "s": "hi"}]),
                                              min_score_field="s", min_score=1)])
    with pytest.raises(ValueError, match="numeric"):
        prepare_data(cfg)


def test_source_with_nothing_kept_errors(workspace, stories_path):
    cfg = _cfg(workspace, stories_path, [_src("a", _jsonl(workspace / "a.jsonl", [{"text": GOOD, "s": 0}]),
                                              min_score_field="s", min_score=3)])
    with pytest.raises(ValueError, match="kept no documents"):
        prepare_data(cfg)
