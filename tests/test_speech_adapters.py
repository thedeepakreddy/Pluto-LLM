import sys
import types

import numpy as np
import pytest
from conftest import setup_speech_config

from llm import cli
from llm.speech.build import build_speech_data
from llm.speech.fakes import FakeASR, FakeCodec, FakeTTS
from llm.speech.interfaces import Transcript, WordTiming
from llm.manifest import read_manifest


class _Tok:
    def __init__(self, text, s, e):
        self.text, self.start_ts, self.end_ts = text, s, e


class _Res:
    def __init__(self, audio, tokens):
        self.audio, self.tokens = audio, tokens


def _fake_kokoro(monkeypatch, results):
    mod = types.ModuleType("kokoro")
    seen = {}

    class KPipeline:
        def __init__(self, lang_code):
            seen["lang"] = lang_code

        def __call__(self, text, voice):
            seen["voice"] = voice
            return iter(results)

    mod.KPipeline = KPipeline
    mod.__version__ = "0.9.4"
    monkeypatch.setitem(sys.modules, "kokoro", mod)
    return seen


def test_kokoro_adapter_builds_timings(monkeypatch):
    a, b = np.zeros(24000, np.float32), np.zeros(12000, np.float32)
    seen = _fake_kokoro(monkeypatch, [
        _Res(a, [_Tok("Hello", 0.0, 0.4), _Tok(",", 0.4, 0.4), _Tok("world", 0.5, 0.9)]),
        _Res(b, [_Tok("again", 0.1, 0.4)]),
    ])
    from llm.speech.adapters import KokoroTTS

    tts = KokoroTTS(voice="bf_emma")
    syn = tts.synthesize("Hello, world again")
    assert seen == {"lang": "b", "voice": "bf_emma"}
    assert syn.sample_rate == 24000 and len(syn.audio) == 36000 and syn.audio.dtype == np.float32
    assert syn.words == [WordTiming("Hello", 0.0, 0.4), WordTiming("world", 0.5, 0.9),
                         WordTiming("again", 1.1, 1.4)]
    d = tts.describe()
    assert d["voice"] == "bf_emma" and d["kokoro"] == "0.9.4"


def test_kokoro_adapter_missing_timestamps(monkeypatch):
    _fake_kokoro(monkeypatch, [_Res(np.zeros(100, np.float32), None)])
    from llm.speech.adapters import KokoroTTS

    assert KokoroTTS(voice="bf_emma").synthesize("hi").words is None
    _fake_kokoro(monkeypatch, [_Res(np.zeros(100, np.float32), [_Tok("hi", None, None)])])
    assert KokoroTTS(voice="bf_emma").synthesize("hi").words is None


def _fake_transformers(monkeypatch, pipe_out=None):
    torch = pytest.importorskip("torch")
    mod = types.ModuleType("transformers")
    mod.__version__ = "4.99"
    seen = {}

    class Out:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class MimiModel:
        @classmethod
        def from_pretrained(cls, name):
            seen["mimi"] = name
            return cls()

        def to(self, dev):
            return self

        def eval(self):
            return self

        def encode(self, x, num_quantizers):
            seen["q"] = num_quantizers
            frames = x.shape[-1] // 1920
            return Out(audio_codes=torch.ones(1, num_quantizers, frames, dtype=torch.long))

        def decode(self, codes):
            return Out(audio_values=torch.zeros(1, 1, codes.shape[-1] * 1920))

    def pipeline(task, model, return_timestamps):
        seen["asr"] = (task, model, return_timestamps)

        def run(inp):
            seen["inp"] = inp
            return pipe_out

        return run

    mod.MimiModel, mod.pipeline = MimiModel, pipeline
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return seen


def test_mimi_adapter_shapes(monkeypatch):
    seen = _fake_transformers(monkeypatch)
    from llm.speech.adapters import MimiCodec

    c = MimiCodec()
    assert (c.sample_rate, c.frame_rate, c.codebooks) == (24000, 12.5, 8)
    codes = c.encode(np.zeros(1920 * 5, np.float32), 24000)
    assert codes.shape == (5, 8) and codes.dtype == np.int64 and seen["q"] == 8
    audio = c.decode(codes)
    assert audio.shape == (1920 * 5,) and audio.dtype == np.float32
    assert c.describe()["model"] == "kyutai/mimi"
    assert c.encode(np.zeros(32000, np.float32), 16000).shape == (25, 8)  # resampled 16k -> 24k


def test_whisper_adapter_words(monkeypatch):
    out = {"text": " Hello world", "chunks": [{"text": " Hello", "timestamp": (0.0, 0.5)},
                                              {"text": " world", "timestamp": (0.5, None)}]}
    seen = _fake_transformers(monkeypatch, out)
    from llm.speech.adapters import WhisperASR

    asr = WhisperASR(model="openai/whisper-small.en")
    tr = asr.transcribe(np.zeros(24000, np.float32), 24000)
    assert seen["asr"] == ("automatic-speech-recognition", "openai/whisper-small.en", "word")
    assert seen["inp"]["sampling_rate"] == 16000 and len(seen["inp"]["raw"]) == 16000
    assert tr.text == "Hello world"
    assert tr.words[0] == WordTiming("Hello", 0.0, 0.5)
    assert tr.words[1].word == "world" and tr.words[1].end_s >= tr.words[1].start_s
    assert asr.describe()["model"] == "openai/whisper-small.en"


class _DescribedASR(FakeASR):
    def describe(self):
        return {"model": "x"}


def test_builder_whisper_timings_option(tiny_config):
    cfg = setup_speech_config(tiny_config)

    class BadTimingTTS(FakeTTS):  # TTS timings are wrong (would be dropped), Whisper's are right
        def synthesize(self, text):
            s = super().synthesize(text)
            s.words = None
            return s

    d, complete = build_speech_data(cfg, BadTimingTTS(), FakeCodec(), FakeASR(), timings="whisper")
    assert complete
    m = read_manifest(d)
    assert m["extra"]["timings"] == "whisper"
    assert m["extra"]["drops"].get("timing_mismatch", 0) == 0
    assert m["extra"]["splits"]["val"]["chunks"] > 0


def test_builder_default_timings_recorded(tiny_config):
    cfg = setup_speech_config(tiny_config)
    d, _ = build_speech_data(cfg, FakeTTS(), FakeCodec(), _DescribedASR())
    ex = read_manifest(d)["extra"]
    assert ex["timings"] == "tts" and ex["adapters"]["asr"] == {"model": "x"}


def test_builder_rejects_bad_timings(tiny_config):
    cfg = setup_speech_config(tiny_config)
    with pytest.raises(ValueError):
        build_speech_data(cfg, FakeTTS(), FakeCodec(), FakeASR(), timings="nope")


def test_cli_speech_check_with_fakes(tiny_config, capsys):
    setup_speech_config(tiny_config)
    assert cli.main(["speech-check", str(tiny_config), "--n", "3"]) == 0
    out = capsys.readouterr().out
    for key in ("usable timings", "duration error", "round-trip WER", "s/s TTS", "s/s codec", "s/s ASR"):
        assert key in out
    assert "--timings whisper" not in out  # fake timings all present


def test_cli_speech_check_recommends_whisper(tiny_config, capsys, monkeypatch):
    setup_speech_config(tiny_config)

    class NoTimes(FakeTTS):
        def synthesize(self, text):
            s = super().synthesize(text)
            s.words = None
            return s

    import llm.speech.interfaces as itf
    monkeypatch.setattr(itf, "make_tts", lambda cfg: NoTimes())
    assert cli.main(["speech-check", str(tiny_config), "--n", "3"]) == 0
    assert "--timings whisper" in capsys.readouterr().out


def test_cli_build_speech_timings_flag(tiny_config):
    setup_speech_config(tiny_config)
    assert cli.main(["build-speech", str(tiny_config), "--timings", "whisper"]) == 0
