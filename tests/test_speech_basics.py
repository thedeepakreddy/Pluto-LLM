import numpy as np
import pytest

from llm.config import SpeechConfig
from llm.speech.fakes import FakeASR, FakeCodec, FakeTTS
from llm.speech.interfaces import make_asr, make_codec, make_tts
from llm.speech.wer import char_error_rate, normalize_words, word_error_rate


def _cfg(**kw):
    base = dict(name="s", version=1, tts="fake", codec="fake", asr="fake")
    base.update(kw)
    return SpeechConfig(**base)


def test_normalize_words():
    assert normalize_words("Hello, World! It's  fine.") == ["hello", "world", "it's", "fine"]
    assert normalize_words("") == []


def test_wer_known_values():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0
    assert word_error_rate("the cat sat", "the cat") == pytest.approx(1 / 3)
    assert word_error_rate("the cat sat", "a cat sat on") == pytest.approx(2 / 3)
    assert word_error_rate("Hello, World!", "hello world") == 0.0
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "anything") == 1.0


def test_cer_known_values():
    assert char_error_rate("abc", "abc") == 0.0
    assert char_error_rate("abc", "abd") == pytest.approx(1 / 3)
    assert char_error_rate("abcd", "ab") == pytest.approx(0.5)
    assert char_error_rate("Hello, World!", "hello world") == 0.0
    assert char_error_rate("", "") == 0.0
    assert char_error_rate("", "x") == 1.0


def test_fake_tts_timings():
    syn = FakeTTS().synthesize("Lily went home")
    assert syn.sample_rate == 24000
    assert syn.audio.dtype == np.float32 and syn.audio.ndim == 1
    assert len(syn.audio) == int(1.2 * 24000)
    assert [w.word for w in syn.words] == ["Lily", "went", "home"]
    assert [w.start_s for w in syn.words] == pytest.approx([0.0, 0.4, 0.8])
    assert [w.end_s for w in syn.words] == pytest.approx([0.4, 0.8, 1.2])


def test_fake_tts_is_deterministic_and_word_specific():
    a, b = FakeTTS().synthesize("one two"), FakeTTS().synthesize("one two")
    assert np.array_equal(a.audio, b.audio)
    c = FakeTTS().synthesize("one three")
    assert not np.array_equal(a.audio, c.audio)


def test_fake_codec_shapes():
    codec = FakeCodec()
    assert (codec.sample_rate, codec.frame_rate, codec.codebooks) == (24000, 12.5, 8)
    audio = FakeTTS().synthesize("Lily went home").audio
    codes = codec.encode(audio, 24000)
    assert codes.shape == (15, 8) and codes.dtype == np.int64
    assert codes.min() >= 0 and codes.max() < 2048
    assert len(codec.decode(codes)) == 15 * 1920


def test_fake_codec_encode_formula():
    codes = FakeCodec().encode(np.full(1920, 0.0025, dtype=np.float32), 24000)
    base = int(abs(np.float32(0.0025).astype(np.float64)) * 1e6)
    assert codes[0].tolist() == [(base + k) % 2048 for k in range(8)]


def test_fake_asr_roundtrip():
    tts, asr = FakeTTS(), FakeASR()
    for text in ["Lily went home", "the dog ran far away"]:
        syn = tts.synthesize(text)
        tr = asr.transcribe(syn.audio, syn.sample_rate)
        assert tr.text == text
        assert len(tr.words) == len(text.split())
    unknown = np.random.default_rng(0).standard_normal(24000).astype(np.float32)
    assert asr.transcribe(unknown, 24000).text == ""


def test_fake_asr_does_not_recognise_decoded_audio():
    syn = FakeTTS().synthesize("Lily went home")
    codec = FakeCodec()
    decoded = codec.decode(codec.encode(syn.audio, 24000))
    assert FakeASR().transcribe(decoded, 24000).text == ""


def test_make_fake_components():
    assert isinstance(make_tts(_cfg()), FakeTTS)
    assert isinstance(make_codec(_cfg()), FakeCodec)
    assert isinstance(make_asr(_cfg()), FakeASR)


def test_make_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="bogus"):
        make_tts(_cfg(tts="bogus"))
    with pytest.raises(ValueError, match="bogus"):
        make_codec(_cfg(codec="bogus"))
    with pytest.raises(ValueError, match="bogus"):
        make_asr(_cfg(asr="bogus"))
