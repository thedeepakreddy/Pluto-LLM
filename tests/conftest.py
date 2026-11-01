import os
from pathlib import Path

import pytest
import yaml

FIXTURES = Path(__file__).parent / "fixtures"


def tiny_config_dict(stories: Path) -> dict:
    return {
        "data": {
            "name": "tiny",
            "version": 1,
            "source": {"type": "local_jsonl", "path": str(stories), "license": "test-fixture"},
            "min_chars": 20,
            "max_chars": 20000,
            "split": {"train": 1.0, "val": 0.0, "test": 0.0},
        },
        "tokenizer": {"name": "tiny-bpe", "version": 1, "vocab_size": 300},
        "shards": {"name": "tiny-shards", "version": 1},
        "model": {"n_layer": 2, "d_model": 64, "n_head": 2, "context_length": 64},
        "train": {
            "seed": 1337,
            "tokens_per_step": 512,
            "micro_batch_size": 8,
            "max_steps": 40,
            "lr": 3e-3,
            "warmup_steps": 5,
            "precision": "fp32",
            "device": "cpu",
            "log_every": 1,
            "eval_every": 20,
            "eval_batches": 2,
            "eval_split": "train",
            "ckpt_every": 10,
            "keep_last": 3,
        },
        "run": {"name": "tiny"},
    }


@pytest.fixture
def stories_path() -> Path:
    return FIXTURES / "stories.jsonl"


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def tiny_config(workspace, stories_path) -> Path:
    p = workspace / "tiny.yaml"
    p.write_text(yaml.safe_dump(tiny_config_dict(stories_path), sort_keys=False))
    return p


def edit_yaml(path: Path, fn) -> None:
    d = yaml.safe_load(path.read_text())
    fn(d)
    path.write_text(yaml.safe_dump(d, sort_keys=False))


@pytest.fixture(scope="session")
def stories_texts() -> list[str]:
    import json

    return [json.loads(l)["text"] for l in (FIXTURES / "stories.jsonl").read_text().splitlines()]
