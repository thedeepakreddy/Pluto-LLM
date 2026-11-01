import random

import pytest

from llm import cli
from llm.manifest import manifest_checksum, read_manifest
from llm.tokenizer.bpe import BPETokenizer, TokenizerTrainingError


@pytest.fixture(scope="module")
def tok(stories_texts):
    return BPETokenizer.train(stories_texts, vocab_size=300)


@pytest.mark.parametrize("s", ["", " ", "Hello, world!", "  multiple   spaces\n\n", "naïve café ☕ 日本語 🙂",
                               "def f(x):\n    return x**2  # code", "3.14159 1,000,000 2026-11-01",
                               "<|endoftext|>", "It's they'll we've"])
def test_roundtrip(tok, s):
    assert tok.decode(tok.encode(s)) == s


def test_roundtrip_random_unicode(tok):
    rng = random.Random(0)
    alphabet = [chr(c) for c in range(32, 0x2FFF)] + ["🙂", "🚀", "\n", "\t"]
    for _ in range(200):
        s = "".join(rng.choices(alphabet, k=rng.randint(0, 40)))
        assert tok.decode(tok.encode(s)) == s


def test_id_layout(tok, stories_texts):
    assert tok.vocab_size == 300 and tok.eot_id == 299
    ids = tok.encode(" ".join(stories_texts) + " <|endoftext|>")
    assert max(ids) < 299 and min(ids) >= 0
    assert any(i >= 256 for i in ids)  # merges are actually used
    assert tok.decode([tok.eot_id]) == ""


def test_training_deterministic(stories_texts, tmp_path):
    a, b = BPETokenizer.train(stories_texts, 300), BPETokenizer.train(stories_texts, 300)
    a.save(tmp_path / "a"), b.save(tmp_path / "b")
    assert (tmp_path / "a" / "tokenizer.json").read_bytes() == (tmp_path / "b" / "tokenizer.json").read_bytes()


def test_save_load_identical(tok, tmp_path):
    tok.save(tmp_path / "t")
    t2 = BPETokenizer.load(tmp_path / "t")
    for s in ["Once upon a time", "Lily and Tom", "xyz 123", "🙂", ""]:
        assert t2.encode(s) == tok.encode(s)
    assert t2.vocab_size == tok.vocab_size


def test_compression(tok, stories_texts):
    text = " ".join(stories_texts)
    assert len(text.encode()) / len(tok.encode(text)) > 1.5


def test_too_small_corpus_raises():
    with pytest.raises(TokenizerTrainingError):
        BPETokenizer.train(["ab"], vocab_size=300)


def test_cli_train_tokenizer(tiny_config, workspace):
    assert cli.main(["prepare-data", str(tiny_config)]) == 0
    assert cli.main(["train-tokenizer", str(tiny_config)]) == 0
    tdir = workspace / "tokenizers" / "tiny-bpe-v1"
    m = read_manifest(tdir)
    assert m["inputs"]["data"] == manifest_checksum(workspace / "data" / "tiny-v1")
    assert m["extra"]["vocab_size"] == 300
    assert BPETokenizer.load(tdir).vocab_size == 300


def _naive_merges(texts, vocab_size):
    import regex
    from collections import Counter

    from llm.tokenizer.bpe import GPT2_SPLIT_PATTERN, _merge

    chunks = Counter(c for t in texts for c in regex.findall(GPT2_SPLIT_PATTERN, t))
    words = {tuple(c.encode()): f for c, f in chunks.items()}
    merges = []
    for new_id in range(256, vocab_size - 1):
        counts = Counter()
        for w, f in words.items():
            for p in zip(w, w[1:]):
                counts[p] += f
        best = min(counts, key=lambda p: (-counts[p], p))
        merges.append(best)
        new = Counter()
        for w, f in words.items():
            new[tuple(_merge(list(w), best, new_id))] += f
        words = new
    return merges


def test_incremental_training_matches_naive_reference(stories_texts):
    tok = BPETokenizer.train(stories_texts, 320)
    assert tok._merges == _naive_merges(stories_texts, 320)
