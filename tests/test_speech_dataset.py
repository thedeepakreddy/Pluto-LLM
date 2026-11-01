import json

import numpy as np
import pytest
import torch
from conftest import setup_speech_config

from llm.shards import DataPosition, TokenChunks, shard_dir
from llm.speech.build import build_speech_data, speech_dir
from llm.speech.dataset import SpeechBatch, SpeechShards, collate, next_speech_batch
from llm.speech.fakes import FakeASR, FakeCodec, FakeTTS
from llm.speech.layout import EMPTY, build_aligned, build_text_then_audio, special_ids

K = 8


def _cfg(tiny_config, layout="aligned"):
    # context_length is 64 in the tiny config, so cap chunks at 4.5 s (needs 57 + delay + 2 positions)
    cfg = setup_speech_config(tiny_config, speech={"max_seconds": 4.5}, audio={"layout": layout})
    build_speech_data(cfg, FakeTTS(), FakeCodec(), FakeASR())
    return cfg


@pytest.fixture
def cfg(tiny_config):
    return _cfg(tiny_config)


@pytest.fixture
def sp(cfg):
    return special_ids(cfg.tokenizer.vocab_size)


@pytest.fixture
def ds(cfg):
    return SpeechShards(speech_dir(cfg), "train")


@pytest.fixture
def text_chunks(cfg):
    return TokenChunks(shard_dir(cfg), "train", cfg.model.context_length - 1)


def _same(a: SpeechBatch, b: SpeechBatch) -> bool:
    return all(torch.equal(getattr(a, f), getattr(b, f)) for f in ("text", "codes", "text_mask", "code_mask"))


def _run(ds, cfg, sp, text_chunks, n, pos=None, start_step=0, bs=2, seed=7):
    pos, out = pos or DataPosition(), []
    for step in range(start_step, start_step + n):
        b, pos = next_speech_batch(ds, pos, bs, seed, cfg.audio, sp, step, text_chunks)
        out.append(b)
    return out, pos


def test_shards_roundtrip(cfg, ds):
    d = speech_dir(cfg)
    rows = [json.loads(line) for line in (d / "train_index.jsonl").read_text().splitlines()]
    assert len(ds) == len(rows) > 0
    assert ds.seconds_total == pytest.approx(sum(r["seconds"] for r in rows))
    raw = np.fromfile(d / "train_codes.bin", dtype=np.uint16).reshape(-1, K)
    for i in (0, len(rows) - 1):
        text_ids, token_frames, codes = ds[i]
        r = rows[i]
        assert text_ids == r["text_ids"] and token_frames == r["token_frames"]
        assert codes.shape == (r["frames"], K)
        np.testing.assert_array_equal(codes, raw[r["offset"]: r["offset"] + r["frames"]])
    with pytest.raises(IndexError):
        ds[len(rows)]


def test_collate_padding_masks(sp):
    codes_a = np.arange(3 * K).reshape(3, K) % 2048
    codes_b = np.arange(5 * K).reshape(5, K) % 2048
    a = build_aligned([1, 2], [0, 1], codes_a, sp, 1)
    b = build_aligned([3, 4, 5], [0, 2, 3], codes_b, sp, 1)
    batch = collate([a, b], sp)
    t = max(len(a.text), len(b.text))
    assert len(a.text) < len(b.text)
    assert batch.text.shape == (2, t) and batch.codes.shape == (2, t, K)
    assert batch.text_mask.shape == (2, t) and batch.code_mask.shape == (2, t, K)
    assert batch.text.dtype == torch.long and batch.codes.dtype == torch.long
    assert batch.text_mask.dtype == torch.bool and batch.code_mask.dtype == torch.bool
    n = len(a.text)
    assert (batch.text[0, n:] == sp.pad_text).all() and (batch.codes[0, n:] == EMPTY).all()
    assert not batch.text_mask[0, n:].any() and not batch.code_mask[0, n:].any()
    np.testing.assert_array_equal(batch.text[0, :n].numpy(), a.text)
    np.testing.assert_array_equal(batch.codes[1].numpy(), b.codes)
    np.testing.assert_array_equal(batch.code_mask[0, :n].numpy(), a.code_mask)


def test_batch_to_device(sp):
    ex = build_aligned([1], [0], np.zeros((2, K), dtype=np.int64), sp, 1)
    b = collate([ex], sp).to("cpu")
    assert isinstance(b, SpeechBatch) and b.text.device.type == "cpu"


def test_same_seed_same_batches(cfg, ds, sp, text_chunks):
    a, _ = _run(ds, cfg, sp, text_chunks, 8)
    b, _ = _run(ds, cfg, sp, text_chunks, 8)
    assert all(_same(x, y) for x, y in zip(a, b))
    c, _ = _run(ds, cfg, sp, text_chunks, 8, seed=8)
    assert not all(_same(x, y) for x, y in zip(a, c))


def test_resume_position_identical(cfg, ds, sp, text_chunks):
    straight, _ = _run(ds, cfg, sp, text_chunks, 10)
    first, pos = _run(ds, cfg, sp, text_chunks, 4)
    rest, _ = _run(ds, cfg, sp, text_chunks, 6, pos=pos, start_step=4)
    assert len(straight) == 10
    assert all(_same(x, y) for x, y in zip(straight, first + rest))


def test_resume_across_epoch_boundary(cfg, ds, sp):
    # batch_size that does not divide len(ds) forces epoch rollover; resuming must still match
    bs = max(2, len(ds) // 2 + 1)
    straight, end = _run(ds, cfg, sp, None, 6, bs=bs)
    assert end.epoch >= 1
    first, pos = _run(ds, cfg, sp, None, 3, bs=bs)
    rest, _ = _run(ds, cfg, sp, None, 3, pos=pos, start_step=3, bs=bs)
    assert all(_same(x, y) for x, y in zip(straight, first + rest))


def test_text_every_10th_step(cfg, ds, sp, text_chunks):
    assert cfg.audio.text_every == 10
    pos = DataPosition()
    for step in range(20):
        b, new_pos = next_speech_batch(ds, pos, 2, 7, cfg.audio, sp, step, text_chunks)
        is_text = step in (9, 19)
        assert not b.code_mask.any() if is_text else b.code_mask.any()
        assert (b.codes == EMPTY).all() == is_text
        assert (new_pos == pos) == is_text  # text batches leave the speech position alone
        assert (b.text[:, 0] == sp.bos).all()
        assert b.text.shape[1] <= cfg.model.context_length
        if is_text:
            assert not (b.text == sp.audio).any() and not (b.text == sp.end_audio).any()
        pos = new_pos


def test_no_text_chunks_means_all_speech(cfg, ds, sp):
    for step in (9, 19):
        b, _ = next_speech_batch(ds, DataPosition(), 2, 7, cfg.audio, sp, step, None)
        assert b.code_mask.any()


def test_layout_switch(tiny_config):
    cfg_a = _cfg(tiny_config, "aligned")
    sp = special_ids(cfg_a.tokenizer.vocab_size)
    ds = SpeechShards(speech_dir(cfg_a), "train")
    aligned, _ = next_speech_batch(ds, DataPosition(), 3, 7, cfg_a.audio, sp, 0, None)
    from dataclasses import replace

    audio_b = replace(cfg_a.audio, layout="text_then_audio")
    base, _ = next_speech_batch(ds, DataPosition(), 3, 7, audio_b, sp, 0, None)
    assert (base.text == sp.audio).sum() == 3  # exactly one AUDIO marker per example
    assert not (aligned.text == sp.audio).any()
    text_ids, _, codes = ds[int(np.random.default_rng([7, 0]).permutation(len(ds))[0])]
    ref = build_text_then_audio(text_ids, codes, sp, cfg_a.audio.acoustic_delay)
    np.testing.assert_array_equal(base.text[0, : len(ref.text)].numpy(), ref.text)
