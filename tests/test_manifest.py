import json

import pytest

from llm.manifest import (ProvenanceError, atomic_write_json, manifest_checksum, read_manifest,
                          sha256_bytes, verify_files, verify_input, write_manifest)


def make_artifact(d, name="a", content=b"hello"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "x.bin").write_bytes(content)
    return write_manifest(d, kind="test", name=name, version=1, files=["x.bin"], inputs={})


def test_manifest_roundtrip(tmp_path):
    m = make_artifact(tmp_path / "a")
    r = read_manifest(tmp_path / "a")
    assert r == m
    assert r["kind"] == "test" and r["version"] == 1
    assert r["files"] == {"x.bin": sha256_bytes(b"hello")}
    verify_files(tmp_path / "a")


def test_verify_files_detects_modification(tmp_path):
    make_artifact(tmp_path / "a")
    (tmp_path / "a" / "x.bin").write_bytes(b"tampered")
    with pytest.raises(ProvenanceError, match="x.bin"):
        verify_files(tmp_path / "a")


def test_verify_files_detects_missing(tmp_path):
    make_artifact(tmp_path / "a")
    (tmp_path / "a" / "x.bin").unlink()
    with pytest.raises(ProvenanceError, match="x.bin"):
        verify_files(tmp_path / "a")


def test_verify_input_mismatch(tmp_path):
    up = tmp_path / "up"
    make_artifact(up)
    down = tmp_path / "down"
    down.mkdir()
    m = write_manifest(down, "test", "d", 1, files=[], inputs={"up": manifest_checksum(up)})
    verify_input(m, "up", up)
    make_artifact(up, name="changed", content=b"other")
    with pytest.raises(ProvenanceError, match="up"):
        verify_input(m, "up", up)


def test_atomic_write_leaves_no_tmp(tmp_path):
    atomic_write_json(tmp_path / "f.json", {"a": 1})
    assert json.loads((tmp_path / "f.json").read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_checksum_ignores_timestamps_and_info(tmp_path):
    import time

    a = tmp_path / "a"
    make_artifact(a)
    first = manifest_checksum(a)
    time.sleep(0.01)
    a.mkdir(exist_ok=True)
    write_manifest(a, "test", "a", 1, files=["x.bin"], inputs={}, info={"seconds": 99})
    assert manifest_checksum(a) == first
    (a / "x.bin").write_bytes(b"different")
    write_manifest(a, "test", "a", 1, files=["x.bin"], inputs={})
    assert manifest_checksum(a) != first
