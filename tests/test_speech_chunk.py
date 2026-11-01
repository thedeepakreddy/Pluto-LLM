import pytest

from llm.speech.chunk import TimingMismatch, chunk_story, split_sentences, token_start_frames
from llm.speech.fakes import FakeTTS
from llm.speech.interfaces import WordTiming
from llm.tokenizer.bpe import BPETokenizer

FRAME_RATE = 12.5


@pytest.fixture(scope="module")
def tok(stories_texts):
    return BPETokenizer.train(stories_texts, vocab_size=300)


def test_split_sentences():
    out = split_sentences('She said, "Hi!" Then she left. The end.')
    assert out == ['She said, "Hi!"', "Then she left.", "The end."]


def test_split_sentences_edge_cases():
    assert split_sentences("") == []
    assert split_sentences("  No end punctuation here  ") == ["No end punctuation here"]
    assert split_sentences("Wait... what? Yes!\nOk.") == ["Wait...", "what?", "Yes!", "Ok."]


def test_chunk_story_respects_limit():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = chunk_story(text, max_words=6)
    assert chunks == ["One two three. Four five six.", "Seven eight nine. Ten eleven twelve."]
    assert all(len(c.split()) <= 6 for c in chunks)
    assert " ".join(chunks) == text


def test_chunk_story_long_sentence_alone():
    text = "Short one. This sentence is far too long for the limit. Tiny."
    chunks = chunk_story(text, max_words=4)
    assert chunks == ["Short one.", "This sentence is far too long for the limit.", "Tiny."]


def _pieces(tok, ids):
    return [tok._bytes[i].decode("utf-8", errors="replace") for i in ids]


def test_token_start_frames_fake_tts(tok):
    text = "Lily went to the park. She was happy!"
    ids = tok.encode(text)
    words = FakeTTS().synthesize(text).words
    frames = token_start_frames(text, ids, tok, words, FRAME_RATE)
    assert len(frames) == len(ids)
    assert frames[0] == 0
    assert frames == sorted(frames)
    # the token that starts the word "went" (first non-space char 'w' after "Lily") maps to floor(0.4*12.5) = 5
    pieces = _pieces(tok, ids)
    went = [i for i, p in enumerate(pieces) if p.lstrip().startswith("w") and "".join(pieces[:i]).endswith("Lily")]
    assert went, pieces
    assert frames[went[0]] == 5
    assert frames[-1] == 35  # "!" follows the 8th word "happy", which starts at 2.8 s -> frame 35


def test_token_start_frames_punctuation_takes_previous_word(tok):
    text = "Hello, world."
    ids = tok.encode(text)
    words = FakeTTS().synthesize(text).words
    frames = token_start_frames(text, ids, tok, words, FRAME_RATE)
    pieces = _pieces(tok, ids)
    assert "".join(pieces) == text
    by_piece = dict(zip(pieces, frames))
    if "," in by_piece:
        assert by_piece[","] == 0  # after "Hello"
    if "." in by_piece:
        assert by_piece["."] == 5  # after "world"
    assert frames[-1] == 5


def test_token_start_frames_multibyte(tok):
    text = "Café time — yes"
    ids = tok.encode(text)
    words = [WordTiming("café", 0.0, 0.4), WordTiming("time", 0.4, 0.8), WordTiming("yes", 0.8, 1.2)]
    frames = token_start_frames(text, ids, tok, words, FRAME_RATE)
    assert len(frames) == len(ids)
    assert frames == sorted(frames) and frames[0] == 0 and frames[-1] == 10


def test_token_start_frames_mismatch(tok):
    text = "Lily went"
    ids = tok.encode(text)
    words = [WordTiming("lily", 0.0, 0.4), WordTiming("goes", 0.4, 0.8)]
    with pytest.raises(TimingMismatch):
        token_start_frames(text, ids, tok, words, FRAME_RATE)
    with pytest.raises(TimingMismatch):
        token_start_frames(text, ids, tok, words[:1], FRAME_RATE)


def test_token_start_frames_ids_not_matching_text(tok):
    words = [WordTiming("lily", 0.0, 0.4), WordTiming("went", 0.4, 0.8)]
    with pytest.raises(ValueError):
        token_start_frames("Lily went", tok.encode("Something else"), tok, words, FRAME_RATE)
