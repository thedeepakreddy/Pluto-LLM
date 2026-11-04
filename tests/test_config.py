import pytest
import yaml

from conftest import edit_yaml
from llm.config import ConfigError, config_diff, config_to_dict, grad_accum_steps, load_config


def test_load_roundtrip(tiny_config, workspace):
    cfg = load_config(tiny_config)
    assert cfg.model.n_layer == 2 and cfg.tokenizer.vocab_size == 300
    out = workspace / "again.yaml"
    out.write_text(yaml.safe_dump(config_to_dict(cfg)))
    assert load_config(out) == cfg


def test_defaults_applied(tiny_config):
    cfg = load_config(tiny_config)
    assert cfg.train.betas == (0.9, 0.95)
    assert cfg.train.weight_decay == 0.1 and cfg.train.min_lr_ratio == 0.1
    assert cfg.model.attn_impl == "sdpa" and cfg.data.source.text_field == "text"


def test_unknown_key_rejected(tiny_config):
    edit_yaml(tiny_config, lambda d: d["model"].update(n_layers=3))
    with pytest.raises(ConfigError, match="model.n_layers"):
        load_config(tiny_config)


def test_missing_required_rejected(tiny_config):
    edit_yaml(tiny_config, lambda d: d["model"].pop("d_model"))
    with pytest.raises(ConfigError, match="model.d_model"):
        load_config(tiny_config)


def test_bad_literal_rejected(tiny_config):
    with pytest.raises(ConfigError, match="train.precision"):
        load_config(tiny_config, ["train.precision=fp8"])


def test_override(tiny_config):
    cfg = load_config(tiny_config, ["train.lr=0.01", "model.n_layer=3"])
    assert cfg.train.lr == 0.01 and cfg.model.n_layer == 3


def test_override_unknown_key(tiny_config):
    with pytest.raises(ConfigError, match="train.lrr"):
        load_config(tiny_config, ["train.lrr=0.1"])


def test_tokens_per_step_must_divide(tiny_config):
    with pytest.raises(ConfigError, match="tokens_per_step"):
        load_config(tiny_config, ["train.tokens_per_step=500"])


def test_grad_accum_steps(tiny_config):
    assert grad_accum_steps(load_config(tiny_config, ["train.tokens_per_step=1024"])) == 2


def test_d_model_divisible_by_heads(tiny_config):
    with pytest.raises(ConfigError, match="n_head"):
        load_config(tiny_config, ["model.n_head=3"])


def test_split_ratios_sum_to_one(tiny_config):
    with pytest.raises(ConfigError, match="split"):
        load_config(tiny_config, ["data.split={train: 0.5, val: 0.1, test: 0.1}"])


def test_resume_config_diff_allows_only_session_wall_time_change(tiny_config):
    saved = config_to_dict(load_config(tiny_config))
    resumed = {**saved, "train": {**saved["train"], "max_wall_time_s": 6000}}
    assert config_diff(saved, resumed) == []

    resumed["train"]["max_steps"] += 1
    assert config_diff(saved, resumed) == ["train.max_steps"]
