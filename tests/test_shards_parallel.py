"""Parallel tokenization (M3): workers > 1 must write byte-identical shards."""
import dataclasses

from conftest import run_pipeline
from llm.manifest import read_manifest
from llm.shards import write_shards


def test_parallel_shards_identical(tiny_config):
    cfg, sdir = run_pipeline(tiny_config, ["shards.shard_size_tokens=1000"])
    one = {p.name: p.read_bytes() for p in sdir.glob("*.bin")}
    m1 = read_manifest(sdir)
    cfg3 = dataclasses.replace(cfg, shards=dataclasses.replace(cfg.shards, workers=3))
    write_shards(cfg3)
    three = {p.name: p.read_bytes() for p in sdir.glob("*.bin")}
    assert len(one) > 1 and one == three
    assert read_manifest(sdir)["extra"]["splits"] == m1["extra"]["splits"]
