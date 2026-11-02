import subprocess
import sys
import wave

import numpy as np
import pytest
from test_speech_train import _speech_cfg

from llm import cli
from llm.runs import create_run
from llm.train.loop import train


@pytest.fixture
def run(tiny_config):
    cfg = _speech_cfg(tiny_config, {"max_steps": 2, "eval_every": 1000, "warmup_steps": 1,
                                    "micro_batch_size": 4, "tokens_per_step": 256})
    r = create_run(cfg)
    train(cfg, r)
    return r


def test_cli_speak_codes_only(run, tmp_path):
    out = tmp_path / "o.npy"
    assert cli.main(["speak", str(run), "--text", "the cat sat", "--out", str(out),
                     "--codes-only", "--seed", "3"]) == 0
    codes = np.load(out)
    assert codes.ndim == 2 and codes.shape[1] == 8 and codes.shape[0] > 0


def test_cli_speak_empty_text_errors(run, tmp_path, capsys):
    assert cli.main(["speak", str(run), "--text", "  ", "--out", str(tmp_path / "o.wav")]) == 2
    assert "empty" in capsys.readouterr().err


def test_cli_speak_and_continue_wav(run, tmp_path):
    out = tmp_path / "o.wav"
    assert cli.main(["speak", str(run), "--text", "the cat sat", "--out", str(out)]) == 0
    with wave.open(str(out)) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getnframes() > 0
    out2 = tmp_path / "c.wav"
    assert cli.main(["continue-speech", str(run), "--audio", str(out), "--out", str(out2),
                     "--max-new-frames", "6"]) == 0
    with wave.open(str(out2)) as w:
        assert w.getframerate() == 24000 and w.getnframes() > 0


def test_continue_speech_missing_audio(run, tmp_path, capsys):
    assert cli.main(["continue-speech", str(run), "--audio", str(tmp_path / "no.wav"),
                     "--out", str(tmp_path / "c.wav")]) == 2
    assert "error" in capsys.readouterr().err


def test_app_importable_without_gradio():
    code = ("import sys, llm.speech.app as a; "
            "assert 'gradio' not in sys.modules; assert callable(a.build_app)")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_resample_keeps_frequency_and_length():
    from llm.speech.wavio import resample

    t = np.arange(48000) / 48000
    y = resample(np.sin(2 * np.pi * 440 * t).astype(np.float32), 48000, 24000)
    assert len(y) == 24000 and y.dtype == np.float32
    assert int(np.argmax(np.abs(np.fft.rfft(y)))) == 440
    assert np.abs(y).max() == pytest.approx(1.0, abs=0.02)
    x = np.ones(10, dtype=np.float32)
    assert resample(x, 24000, 24000) is not None and len(resample(x, 24000, 24000)) == 10


def test_cli_continue_48k_stereo(run, tmp_path):
    t = np.arange(48000) / 48000
    s = (np.sin(2 * np.pi * 220 * t) * 0.3 * 32767).astype("<i2")
    src = tmp_path / "in.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(np.stack([s, s], axis=1).tobytes())
    out = tmp_path / "c.npy"
    assert cli.main(["continue-speech", str(run), "--audio", str(src), "--out", str(out),
                     "--codes-only", "--max-new-frames", "6"]) == 0
    assert np.load(out).shape[1] == 8


def _fake_gradio(monkeypatch):
    import types

    class Err(Exception):
        pass

    clicks = []

    class Ctx:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def click(self, fn, *a, **k):
            clicks.append(fn)

    g = types.ModuleType("gradio")
    g.Error = Err
    for n in ("Blocks", "Tab", "Textbox", "Slider", "Number", "Audio", "Button"):
        setattr(g, n, Ctx)
    monkeypatch.setitem(sys.modules, "gradio", g)
    return Err, clicks


def test_app_continue_maps_errors(run, monkeypatch):
    err, clicks = _fake_gradio(monkeypatch)
    from llm.speech.app import build_app

    build_app(run)
    _, cont_fn = clicks
    with pytest.raises(err):
        cont_fn(None, 0.8, 50, 1, 6)
    with pytest.raises(err):  # too-short prompt / bad frames -> ValueError from the pipeline
        cont_fn((24000, np.zeros(10, dtype=np.float32)), 0.8, 50, 1, -1)
    sr, _ = cont_fn((48000, np.zeros((4800, 2), dtype=np.int16)), 0.8, 50, 1, 6)
    assert sr == 24000
