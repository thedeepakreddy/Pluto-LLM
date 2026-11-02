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
