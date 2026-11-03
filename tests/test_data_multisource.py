"""Multi-source data preparation (M3): several sources in order, byte caps, score filter."""
import json

import pytest
import yaml

from conftest import tiny_config_dict
from llm.config import ConfigError, load_config, load_config_dict
from llm.data.prepare import data_dir, prepare_data
from llm.manifest import read_manifest


def _jsonl(path, docs):
    path.write_text("".join(json.dumps(d) + "\n" for d in docs))
    return str(path)


def _cfg(workspace, stories_path, sources, **data):
    d = tiny_config_dict(stories_path)
    d["data"].pop("source")
    d["data"]["sources"] = sources
    d["data"].update(data)
    d["data"]["root"] = str(workspace / "data")
    return load_config_dict(d)


def _src(name, path, **kw):
    return {"name": name, "type": "local_jsonl", "path": path, "license": "test-fixture", **kw}


A = [{"text": f"alpha document number {i} with enough characters", "score": i} for i in range(10)]
B = [{"text": f"beta document number {i} with enough characters", "score": i} for i in range(6)]


def test_two_sources_counts_in_report(workspace, stories_path):
    cfg = _cfg(workspace, stories_path, [_src("a", _jsonl(workspace / "a.jsonl", A)),
                                         _src("b", _jsonl(workspace / "b.jsonl", B))])
    out = prepare_data(cfg)
    per = read_manifest(out)["extra"]["sources"]
    assert [s["name"] for s in per] == ["a", "b"]
    assert per[0]["received"] == 10 and per[1]["received"] == 6
    assert read_manifest(out)["extra"]["clean"]["kept"] == 16
    report = (out / "data_report.md").read_text()
    assert "| a |" in report and "| b |" in report


def test_max_bytes_stops_source(workspace, stories_path):
    one = len(A[0]["text"].encode())
    cfg = _cfg(workspace, stories_path, [_src("a", _jsonl(workspace / "a.jsonl", A), max_bytes=3 * one)])
    per = read_manifest(prepare_data(cfg))["extra"]["sources"]
    assert per[0]["received"] == 3


def test_score_filter(workspace, stories_path):
    cfg = _cfg(workspace, stories_path, [_src("a", _jsonl(workspace / "a.jsonl", A),
                                              min_score_field="score", min_score=7)])
    out = prepare_data(cfg)
    per = read_manifest(out)["extra"]["sources"][0]
    assert per["received"] == 10 and per["below_min_score"] == 7 and per["kept"] == 3
    texts = [json.loads(l)["text"] for l in (out / "train.jsonl").read_text().splitlines()]
    assert len(texts) == 3


def test_empty_source_fails(workspace, stories_path):
    cfg = _cfg(workspace, stories_path, [_src("a", _jsonl(workspace / "a.jsonl", A)),
                                         _src("empty", _jsonl(workspace / "e.jsonl", []))])
    with pytest.raises(ValueError, match="empty"):
        prepare_data(cfg)


def test_source_and_sources_conflict(stories_path):
    d = tiny_config_dict(stories_path)
    d["data"]["sources"] = [_src("a", str(stories_path))]
    with pytest.raises(ConfigError, match="source"):
        load_config_dict(d)


def test_sources_need_license(stories_path):
    d = tiny_config_dict(stories_path)
    d["data"].pop("source")
    d["data"]["sources"] = [{"name": "a", "type": "local_jsonl", "path": str(stories_path)}]
    with pytest.raises(ConfigError, match="license"):
        load_config_dict(d)


def test_single_source_configs_unchanged(tiny_config):
    cfg = load_config(tiny_config)
    out = prepare_data(cfg)
    m = read_manifest(out)
    assert m["extra"]["source"]["path"] == cfg.data.source.path
    assert m["extra"]["clean"]["kept"] == 40
    assert yaml.safe_load(tiny_config.read_text())["data"]["source"]
