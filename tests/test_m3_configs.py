"""M3 configs load, have the planned sizes, and share one data/tokenizer/shards recipe."""
from pathlib import Path

import pytest

from llm.config import config_diff, config_to_dict, load_config
from llm.model.gpt import expected_num_params
from llm.tokenizer.bpe import PLUTO_SPECIALS

C = Path(__file__).parent.parent / "configs"
NAMES = ["m3-ab-base", "m3-pilot", "m3-cloud", "m3-extend16k"]


@pytest.mark.parametrize("name", NAMES)
def test_m3_configs_load(name):
    cfg = load_config(C / f"{name}.yaml")
    n = expected_num_params(cfg.model, cfg.tokenizer.vocab_size)
    lo, hi = {"m3-ab-base": (15e6, 30e6), "m3-pilot": (120e6, 180e6),
              "m3-cloud": (330e6, 430e6), "m3-extend16k": (330e6, 430e6)}[name]
    assert lo <= n <= hi
    assert set(PLUTO_SPECIALS) <= set(cfg.tokenizer.special_tokens)
    assert all(s.license != "unknown" for s in cfg.data.sources)


def test_shared_data_recipe():
    base = config_to_dict(load_config(C / "m3-pilot.yaml"))
    for name in NAMES:
        diff = config_diff(base, config_to_dict(load_config(C / f"{name}.yaml")))
        assert not [k for k in diff if k.split(".")[0] in ("data", "tokenizer", "shards")], name


@pytest.mark.parametrize("override", ["model.pos=rope", "model.norm=rmsnorm", "model.mlp=swiglu",
                                      "model.n_kv_head=2"])
def test_ab_variants_differ_in_one_field(override):
    base = config_to_dict(load_config(C / "m3-ab-base.yaml"))
    var = config_to_dict(load_config(C / "m3-ab-base.yaml", [override]))
    assert config_diff(base, var) == [override.split("=")[0]]
