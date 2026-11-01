"""Typed experiment configuration loaded from a single YAML file.

Every run is defined by one YAML file with the sections ``data``, ``tokenizer``,
``shards``, ``model``, ``train`` and ``run``. Unknown keys, missing required keys
and invalid values are hard errors so typos never silently change an experiment.
"""
from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


class ConfigError(Exception):
    """Raised for any invalid configuration."""


@dataclass(frozen=True)
class SourceConfig:
    type: Literal["hf", "local_jsonl"]
    path: str
    split: str = "train"
    text_field: str = "text"
    max_docs: int | None = None
    license: str = "unknown"
    url: str | None = None


@dataclass(frozen=True)
class DataConfig:
    name: str
    version: int
    source: SourceConfig
    min_chars: int = 20
    max_chars: int = 20000
    split: dict[str, float] = field(
        default_factory=lambda: {"train": 0.98, "val": 0.01, "test": 0.01}
    )
    root: str = "data"


@dataclass(frozen=True)
class TokenizerConfig:
    name: str
    version: int
    vocab_size: int
    train_sample_bytes: int = 100_000_000
    root: str = "tokenizers"


@dataclass(frozen=True)
class ShardConfig:
    name: str
    version: int
    shard_size_tokens: int = 100_000_000
    root: str = "shards"


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    context_length: int
    dropout: float = 0.0
    attn_impl: Literal["sdpa", "manual"] = "sdpa"
    compile: bool = False


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    tokens_per_step: int
    micro_batch_size: int
    max_steps: int
    lr: float
    warmup_steps: int
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    device: str = "auto"
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 20
    eval_split: Literal["train", "val"] = "val"
    ckpt_every: int = 200
    keep_last: int = 3
    max_wall_time_s: float | None = None
    cost_per_hour: float = 0.0
    wandb_project: str | None = None
    hub_repo: str | None = None


@dataclass(frozen=True)
class RunConfig:
    name: str
    root: str = "runs"


@dataclass(frozen=True)
class Config:
    data: DataConfig
    tokenizer: TokenizerConfig
    shards: ShardConfig
    model: ModelConfig
    train: TrainConfig
    run: RunConfig


# ---------------------------------------------------------------- construction

def _coerce(value: Any, tp: Any, key: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{key}: expected a mapping")
        return _build(tp, value, key)
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{key}: {value!r} is not one of {list(args)}")
        return value
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0], key)
    if origin is tuple:
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            raise ConfigError(f"{key}: expected a list of {len(args)} values")
        return tuple(_coerce(v, a, f"{key}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{key}: expected a mapping")
        kt, vt = args
        return {_coerce(k, kt, key): _coerce(v, vt, f"{key}.{k}") for k, v in value.items()}
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key}: expected an integer, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{key}: expected true/false, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{key}: expected a string, got {value!r}")
        return value
    return value


def _build(cls: type, raw: dict, prefix: str) -> Any:
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for k in raw:
        if k not in fields:
            raise ConfigError(f"unknown config key: {prefix}.{k}" if prefix else f"unknown config key: {k}")
    kwargs = {}
    for name, f in fields.items():
        key = f"{prefix}.{name}" if prefix else name
        if name in raw:
            kwargs[name] = _coerce(raw[name], hints[name], key)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"missing required config key: {key}")
    return cls(**kwargs)


def _apply_override(raw: dict, override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override must look like section.key=value: {override!r}")
    dotted, value = override.split("=", 1)
    parts = dotted.strip().split(".")
    node = raw
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override path is not a mapping: {dotted}")
    node[parts[-1]] = yaml.safe_load(value)


def _validate(cfg: Config) -> None:
    m, t = cfg.model, cfg.train
    if m.d_model % m.n_head != 0:
        raise ConfigError(f"model.d_model ({m.d_model}) must be divisible by model.n_head ({m.n_head})")
    if set(cfg.data.split) != {"train", "val", "test"}:
        raise ConfigError("data.split must have exactly the keys train, val, test")
    if any(v < 0 for v in cfg.data.split.values()) or abs(sum(cfg.data.split.values()) - 1.0) > 1e-6:
        raise ConfigError(f"data.split ratios must be >= 0 and sum to 1, got {cfg.data.split}")
    per_micro = t.micro_batch_size * m.context_length
    if t.tokens_per_step % per_micro != 0:
        raise ConfigError(
            f"train.tokens_per_step ({t.tokens_per_step}) must be divisible by "
            f"train.micro_batch_size * model.context_length ({per_micro})"
        )
    if cfg.tokenizer.vocab_size < 258:
        raise ConfigError("tokenizer.vocab_size must be >= 258 (256 bytes + 1 merge + <|endoftext|>)")
    if t.warmup_steps > t.max_steps:
        raise ConfigError("train.warmup_steps must be <= train.max_steps")


def load_config_dict(raw: dict, overrides: typing.Iterable[str] = ()) -> Config:
    raw = yaml.safe_load(yaml.safe_dump(raw))  # deep copy, plain types only
    for o in overrides:
        _apply_override(raw, o)
    cfg = _build(Config, raw, "")
    _validate(cfg)
    return cfg


def load_config(path: str | Path, overrides: typing.Iterable[str] = ()) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return load_config_dict(raw, overrides)


def config_to_dict(cfg: Config) -> dict:
    def conv(x: Any) -> Any:
        if isinstance(x, tuple):
            return [conv(v) for v in x]
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        return x

    return conv(dataclasses.asdict(cfg))


def grad_accum_steps(cfg: Config) -> int:
    return cfg.train.tokens_per_step // (cfg.train.micro_batch_size * cfg.model.context_length)


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """{"a": {"b": 1}} -> {"a.b": 1}"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(flatten_dict(v, key) if isinstance(v, dict) else {key: v})
    return out


def config_diff(a: dict, b: dict) -> list[str]:
    """Dotted keys whose values differ between two config dicts."""
    fa, fb = flatten_dict(a), flatten_dict(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
