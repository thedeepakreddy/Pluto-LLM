import pytest

from conftest import edit_yaml
from llm.config import AudioConfig, ConfigError, SpeechConfig, config_to_dict, load_config

SPEECH = {"name": "speech-pilot", "version": 1}


def _with(tiny_config, **sections):
    def fn(d):
        d.update(sections)

    edit_yaml(tiny_config, fn)
    return tiny_config


def test_m1_config_still_loads(tiny_config):
    cfg = load_config(tiny_config)
    assert cfg.speech is None and cfg.audio is None


def test_m1_config_dict_omits_unset_speech_sections(tiny_config):
    d = config_to_dict(load_config(tiny_config))
    assert "speech" not in d and "audio" not in d


def test_speech_audio_roundtrip_through_dict(tiny_config):
    from llm.config import load_config_dict

    _with(tiny_config, speech=SPEECH, audio={})
    edit_yaml(tiny_config, lambda d: (d["model"].update(context_length=640),
                                      d["train"].update(tokens_per_step=5120)))
    cfg = load_config(tiny_config)
    assert load_config_dict(config_to_dict(cfg)) == cfg


def test_m1_configs_in_repo_still_load():
    from pathlib import Path

    for p in sorted((Path(__file__).parent.parent / "configs").glob("m1-*.yaml")):
        cfg = load_config(p)
        assert cfg.speech is None and cfg.audio is None


def test_audio_defaults(tiny_config):
    # context 64 is too small for the 30 s speech check, so raise it
    _with(tiny_config, audio={}, speech=SPEECH)
    edit_yaml(tiny_config, lambda d: (d["model"].update(context_length=640),
                                      d["train"].update(tokens_per_step=5120)))
    cfg = load_config(tiny_config)
    assert cfg.audio.layout == "aligned"
    assert cfg.audio.acoustic_delay == 1
    assert cfg.audio.code_loss_weights == (3, 1, 1, 1, 1, 1, 1, 1)
    assert isinstance(cfg.audio.code_loss_weights, tuple)
    assert cfg.audio.codebooks == 8 and cfg.audio.codebook_size == 2048
    assert cfg.audio.text_every == 10 and cfg.audio.init_from is None
    assert cfg.speech.voice == "bf_emma" and cfg.speech.max_seconds == 30.0
    assert cfg.speech.root == "speech" and cfg.speech.asr_model == "openai/whisper-small.en"


def test_speech_section_alone_is_valid(tiny_config):
    _with(tiny_config, speech=SPEECH)
    cfg = load_config(tiny_config)
    assert isinstance(cfg.speech, SpeechConfig) and cfg.audio is None


def test_yaml_list_weights_coerce_to_tuple(tiny_config):
    _with(tiny_config, speech={**SPEECH, "max_seconds": 4.0},
          audio={"codebooks": 4, "code_loss_weights": [2, 1, 1, 1]})
    cfg = load_config(tiny_config)
    assert cfg.audio.code_loss_weights == (2.0, 1.0, 1.0, 1.0)
    assert all(isinstance(w, float) for w in cfg.audio.code_loss_weights)
    d = config_to_dict(cfg)
    assert d["audio"]["code_loss_weights"] == [2.0, 1.0, 1.0, 1.0]


def test_loss_weights_must_be_numbers(tiny_config):
    _with(tiny_config, speech={**SPEECH, "max_seconds": 4.0},
          audio={"code_loss_weights": [3, "x", 1, 1, 1, 1, 1, 1]})
    with pytest.raises(ConfigError, match="code_loss_weights"):
        load_config(tiny_config)


def test_loss_weights_length_must_match_codebooks(tiny_config):
    _with(tiny_config, audio={}, speech={**SPEECH, "max_seconds": 4.0})
    with pytest.raises(ConfigError, match="code_loss_weights"):
        load_config(tiny_config, ["audio.codebooks=4"])


def test_depth_heads_divide(tiny_config):
    _with(tiny_config, audio={}, speech={**SPEECH, "max_seconds": 4.0})
    with pytest.raises(ConfigError, match="depth"):
        load_config(tiny_config, ["audio.depth_n_head=5"])


def test_audio_requires_speech(tiny_config):
    _with(tiny_config, audio={})
    with pytest.raises(ConfigError, match="speech"):
        load_config(tiny_config)


def test_speech_context_fits_30s(tiny_config):
    _with(tiny_config, audio={}, speech={**SPEECH, "max_seconds": 30})
    edit_yaml(tiny_config, lambda d: (d["model"].update(context_length=300),
                                      d["train"].update(tokens_per_step=2400)))
    with pytest.raises(ConfigError, match="context_length"):
        load_config(tiny_config)


def test_speech_context_exact_fit(tiny_config):
    # ceil(30*12.5)=375 + delay 1 + 2 = 378 positions
    _with(tiny_config, audio={}, speech={**SPEECH, "max_seconds": 30})
    edit_yaml(tiny_config, lambda d: (d["model"].update(context_length=378),
                                      d["train"].update(tokens_per_step=3024)))
    assert load_config(tiny_config).audio is not None


def test_audio_unknown_key_and_bad_layout(tiny_config):
    _with(tiny_config, audio={"bogus": 1}, speech=SPEECH)
    with pytest.raises(ConfigError, match="audio.bogus"):
        load_config(tiny_config)
    _with(tiny_config, audio={"layout": "interleaved"}, speech=SPEECH)
    with pytest.raises(ConfigError, match="audio.layout"):
        load_config(tiny_config)


def test_speech_fraction_and_ranges(tiny_config):
    _with(tiny_config, speech={**SPEECH, "qa_fraction": 1.5})
    with pytest.raises(ConfigError, match="speech.qa_fraction"):
        load_config(tiny_config)


def test_audio_config_dataclass_defaults():
    a = AudioConfig()
    assert a.text_loss_weight == 1.0 and a.depth_layers == 3 and a.depth_d_model == 192
