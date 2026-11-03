"""Named reserved special tokens (M3): <think>, <calc>, voice IDs, spares."""
import json

import pytest

from llm.tokenizer.bpe import PLUTO_SPECIALS, BPETokenizer, TokenizerTrainingError

SPECIALS = ["<think>", "</think>", "<calc>", "<spare_0>"]


@pytest.fixture(scope="module")
def tok(stories_texts):
    return BPETokenizer.train(stories_texts, vocab_size=400, specials=SPECIALS)


def test_specials_ids_layout(tok):
    ids = [tok.special_id(n) for n in SPECIALS]
    n_merges = 400 - 256 - len(SPECIALS) - 1
    assert ids == list(range(256 + n_merges, 256 + n_merges + len(SPECIALS)))
    assert tok.eot_id == 399 and tok.specials == dict(zip(SPECIALS, ids))


def test_literal_text_not_special(tok):
    ids = tok.encode("so <think> about it </think>")
    assert not set(ids) & set(tok.specials.values())


def test_decode_special_name(tok):
    assert tok.decode([tok.special_id("<calc>")] + tok.encode(" 2+2")) == "<calc> 2+2"


def test_save_load_roundtrip(tok, tmp_path):
    tok.save(tmp_path)
    again = BPETokenizer.load(tmp_path)
    assert again.specials == tok.specials and again.encode("Once upon a time") == tok.encode("Once upon a time")


def test_old_tokenizer_loads(stories_texts, tmp_path):
    old = BPETokenizer.train(stories_texts, vocab_size=300)
    old.save(tmp_path)
    d = json.loads((tmp_path / "tokenizer.json").read_text())
    d.pop("specials", None)  # files written before M3 have no specials list
    (tmp_path / "tokenizer.json").write_text(json.dumps(d))
    loaded = BPETokenizer.load(tmp_path)
    assert loaded.specials == {} and loaded.vocab_size == 300 and loaded.eot_id == 299


def test_vocab_too_small_rejected(stories_texts):
    with pytest.raises((ValueError, TokenizerTrainingError)):
        BPETokenizer.train(stories_texts, vocab_size=256 + len(SPECIALS), specials=SPECIALS)


def test_pluto_specials_cover_spec():
    for name in ("<think>", "</think>", "<calc>", "<search>", "<result>", "<ask_thinker>",
                 "<voice:emma>", "<voice:pluto_young>", "<now>", "<sure>", "<unsure>"):
        assert name in PLUTO_SPECIALS
    assert len(set(PLUTO_SPECIALS)) == len(PLUTO_SPECIALS)
