import json

import numpy as np
import pytest

from conftest import run_pipeline
from llm import cli
from llm.data.prepare import prepare_data
from llm.manifest import ProvenanceError, read_manifest
from llm.shards import (DataPosition, TokenChunks, epoch_order, next_batch, shard_dtype,
                        verify_shard_provenance)
from llm.tokenizer.bpe import BPETokenizer, tokenizer_dir, train_tokenizer


def _all_tokens(shard_dir, split):
    m = read_manifest(shard_dir)
    dt = np.dtype(m["extra"]["dtype"])
    return np.concatenate([np.fromfile(shard_dir / f, dtype=dt) for f in m["extra"]["splits"][split]["files"]])


def test_dtype_rule():
    assert shard_dtype(65536) == np.uint16 and shard_dtype(65537) == np.uint32


def test_write_read_roundtrip(prepared_tokenized, workspace):
    cfg, sdir = prepared_tokenized
    tok = BPETokenizer.load(tokenizer_dir(cfg))
    docs = [json.loads(l)["text"] for l in (workspace / "data/tiny-v1/train.jsonl").read_text().splitlines()]
    ids = _all_tokens(sdir, "train").tolist()
    expected = []
    for d in docs:
        expected += tok.encode(d) + [tok.eot_id]
    assert ids == expected
    assert ids.count(tok.eot_id) == len(docs) == 40


def test_manifest_records_bytes_and_tokens(prepared_tokenized):
    _, sdir = prepared_tokenized
    s = read_manifest(sdir)["extra"]["splits"]
    assert s["train"]["tokens"] == len(_all_tokens(sdir, "train")) and s["train"]["bytes"] > 0
    assert s["val"] == {"tokens": 0, "bytes": 0, "files": []}


def test_multiple_shard_files(tiny_config):
    cfg, sdir = run_pipeline(tiny_config, ["shards.shard_size_tokens=1000"])
    files = read_manifest(sdir)["extra"]["splits"]["train"]["files"]
    assert len(files) > 1
    chunks = TokenChunks(sdir, "train", 64)
    flat = _all_tokens(sdir, "train")
    assert np.array_equal(chunks[3], flat[3 * 64: 3 * 64 + 65])


def test_manifest_mismatch_hard_error(tiny_config):
    from llm.config import load_config

    cfg, _ = run_pipeline(tiny_config)
    verify_shard_provenance(cfg)
    train_tokenizer(load_config(tiny_config, ["tokenizer.vocab_size=301"]))  # same dir, new tokenizer
    with pytest.raises(ProvenanceError, match="tokenizer"):
        verify_shard_provenance(cfg)


def test_tokenize_refuses_stale_tokenizer(tiny_config):
    from llm.config import load_config

    cfg, _ = run_pipeline(tiny_config)
    prepare_data(load_config(tiny_config, ["data.min_chars=380"]))  # data changes after tokenizer trained
    assert cli.main(["prepare-data", str(tiny_config), "--set", "data.min_chars=380"]) == 0
    with pytest.raises(ProvenanceError, match="data"):
        cli.main(["tokenize", str(tiny_config)])


def test_same_seed_same_order():
    assert np.array_equal(epoch_order(100, 7, 0), epoch_order(100, 7, 0))
    assert not np.array_equal(epoch_order(100, 7, 0), epoch_order(100, 7, 1))
    assert sorted(epoch_order(100, 7, 0).tolist()) == list(range(100))


def test_batch_shapes_and_shift(prepared_tokenized):
    _, sdir = prepared_tokenized
    chunks = TokenChunks(sdir, "train", 64)
    x, y, pos = next_batch(chunks, DataPosition(), 4, seed=1)
    assert x.shape == y.shape == (4, 64) and str(x.dtype) == "torch.int64"
    assert (x[:, 1:] == y[:, :-1]).all()
    assert pos == DataPosition(0, 4)


def test_resume_position_identical(prepared_tokenized):
    _, sdir = prepared_tokenized
    chunks = TokenChunks(sdir, "train", 64)
    pos, full = DataPosition(), []
    for _ in range(10):
        x, _, pos = next_batch(chunks, pos, 3, seed=5)
        full.append(x)
    pos, part = DataPosition(), []
    for _ in range(4):
        x, _, pos = next_batch(chunks, pos, 3, seed=5)
        part.append(x)
    resumed = DataPosition(pos.epoch, pos.index)
    for _ in range(6):
        x, _, resumed = next_batch(chunks, resumed, 3, seed=5)
        part.append(x)
    assert all((a == b).all() for a, b in zip(full, part))


def test_epoch_rollover(prepared_tokenized):
    _, sdir = prepared_tokenized
    chunks = TokenChunks(sdir, "train", 64)
    n = len(chunks)
    x, _, pos = next_batch(chunks, DataPosition(0, n - 1), 3, seed=0)
    assert x.shape[0] == 3 and pos == DataPosition(1, 2)


def test_chunks_too_small_raises(prepared_tokenized):
    _, sdir = prepared_tokenized
    with pytest.raises(ValueError, match="val"):
        TokenChunks(sdir, "val", 64)


def test_bytes_per_token(prepared_tokenized):
    _, sdir = prepared_tokenized
    assert TokenChunks(sdir, "train", 64).bytes_per_token > 1.0
