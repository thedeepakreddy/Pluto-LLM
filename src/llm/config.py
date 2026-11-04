"""Typed experiment configuration loaded from a single YAML file.

Every run is defined by one YAML file with the sections ``data``, ``tokenizer``,
``shards``, ``model``, ``train`` and ``run``. Unknown keys, missing required keys
and invalid values are hard errors so typos never silently change an experiment.
"""
from __future__ import annotations

import dataclasses
import math
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
    note: str | None = None
    name: str = "main"
    config_name: str | None = None  # Hugging Face dataset config (subset)
    streaming: bool = False
    max_bytes: int | None = None  # stop this source after this many UTF-8 bytes received
    min_score_field: str | None = None  # quality filter, e.g. FineWeb-Edu "int_score"
    min_score: float | None = None
    data_dir: str | None = None  # Hugging Face data_dir (e.g. "data/python" in The Stack)
    prose: bool = True  # False: skip prose heuristics (code)
    chunk_chars: int | None = None  # split longer docs at paragraph breaks (books, long articles)


@dataclass(frozen=True)
class DataConfig:
    name: str
    version: int
    source: SourceConfig | None = None
    sources: tuple[SourceConfig, ...] | None = None  # M3: several sources, taken in order
    quality_filters: bool = False  # M3: Gopher-style prose heuristics (sources path only)
    decontaminate: tuple[str, ...] = ()  # benchmark names (llm.data.benchmarks) to remove 13-gram overlap with
    near_dedup: bool = False  # M3: MinHash near-duplicate removal, global across sources
    workers: int = 1  # >1: per-document analysis in a process pool (output identical to 1)
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
    special_tokens: tuple[str, ...] = ()  # named control tokens (M3: llm.tokenizer.bpe.PLUTO_SPECIALS)
    sample: Literal["head", "spread"] = "head"  # spread: every k-th document across the whole split
    reserved_spare: int = 0  # extra unused tokens <spare_0>.. for future abilities


@dataclass(frozen=True)
class ShardConfig:
    name: str
    version: int
    shard_size_tokens: int = 100_000_000
    root: str = "shards"
    workers: int = 1  # >1: encode documents in a process pool (output identical to 1)


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    context_length: int
    dropout: float = 0.0
    attn_impl: Literal["sdpa", "manual"] = "sdpa"
    compile: bool = False
    # M3 layer switches; the defaults are the M1/M2 architecture.
    pos: Literal["learned", "rope"] = "learned"
    rope_theta: float = 10000.0
    sliding_window: int | None = None  # local attention span (tokens) for windowed layers
    global_every: int = 0  # with sliding_window: every k-th layer keeps full attention (0: none)
    mtp_heads: int = 0  # 1: DeepSeek-style multi-token prediction module (predicts token t+2)
    mtp_weight: float = 0.3  # weight of the auxiliary MTP loss (training only)
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    mlp: Literal["gelu", "swiglu"] = "gelu"
    n_kv_head: int | None = None  # None = n_head (multi-head); fewer = grouped-query attention
    swiglu_hidden: int | None = None  # None = 8/3 * d_model rounded up to a multiple of 64


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
    init_from: str | None = None  # text runs: start from these weights (fresh optimizer, step 0)


@dataclass(frozen=True)
class RunConfig:
    name: str
    root: str = "runs"


@dataclass(frozen=True)
class SpeechConfig:
    """Synthetic-speech dataset recipe (M2). Optional; absent for text-only runs."""

    name: str
    version: int
    voice: str = "bf_emma"
    max_seconds: float = 30.0
    hours: float = 200.0
    val_chunks: int = 100
    test_chunks: int = 200
    qa_fraction: float = 0.02
    qa_max_wer: float = 0.3
    tts: str = "kokoro"
    codec: str = "mimi"
    asr: str = "whisper"
    asr_model: str = "openai/whisper-small.en"
    keep_wavs: int = 20
    root: str = "speech"


@dataclass(frozen=True)
class AudioConfig:
    """Audio-model (brain + depth "voice box") settings (M2). Optional; needs ``speech``."""

    layout: Literal["aligned", "text_then_audio"] = "aligned"
    codebooks: int = 8
    codebook_size: int = 2048
    acoustic_delay: int = 1
    depth_layers: int = 3
    depth_d_model: int = 192
    depth_n_head: int = 8
    code_loss_weights: tuple[float, ...] = (3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    text_loss_weight: float = 1.0
    text_every: int = 10
    init_from: str | None = None


@dataclass(frozen=True)
class IdentityConfig:
    """Creator-identity (PCID) recipe (M2). Optional. Exactly one of path / hf_dataset."""

    path: str | None = None
    hf_dataset: str | None = None
    name: str = "pcid"
    version: int = 1
    exclude_fields: tuple[str, ...] = ("current_location",)
    holdout_fraction: float = 0.2
    text_every: int = 50
    speech_every: int = 20
    root: str = "identity"


@dataclass(frozen=True)
class Config:
    data: DataConfig
    tokenizer: TokenizerConfig
    shards: ShardConfig
    model: ModelConfig
    train: TrainConfig
    run: RunConfig
    speech: SpeechConfig | None = None
    audio: AudioConfig | None = None
    identity: IdentityConfig | None = None


# Mimi codec frame rate (frames/s); 24 kHz audio / 1,920 samples per frame.
MIMI_FRAME_RATE = 12.5


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
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{key}: expected a list of values")
        return tuple(_coerce(v, args[0], f"{key}[{i}]") for i, v in enumerate(value))
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


def _validate_model(m: ModelConfig) -> None:
    if m.d_model % m.n_head != 0:
        raise ConfigError(f"model.d_model ({m.d_model}) must be divisible by model.n_head ({m.n_head})")
    kv = m.n_kv_head or m.n_head
    if kv < 1 or m.n_head % kv != 0:
        raise ConfigError(f"model.n_kv_head ({m.n_kv_head}) must divide model.n_head ({m.n_head})")
    if m.sliding_window is not None and m.sliding_window < 1:
        raise ConfigError("model.sliding_window must be >= 1 (or unset for full attention)")
    if m.global_every < 0:
        raise ConfigError("model.global_every must be >= 0")
    if m.mtp_heads not in (0, 1):
        raise ConfigError("model.mtp_heads must be 0 or 1")
    if m.mtp_heads and m.mtp_weight <= 0:
        raise ConfigError("model.mtp_weight must be > 0")
    if m.pos == "rope" and (m.d_model // m.n_head) % 2 != 0:
        raise ConfigError("RoPE needs an even head dimension (d_model / n_head)")


def _validate_sources(d: DataConfig) -> None:
    if (d.source is None) == (d.sources is None):
        raise ConfigError("set exactly one of data.source and data.sources")
    if d.sources is not None:
        if not d.sources:
            raise ConfigError("data.sources must not be empty")
        names = [s.name for s in d.sources]
        if len(set(names)) != len(names):
            raise ConfigError(f"data.sources names must be unique, got {names}")
        for s in d.sources:
            if s.license == "unknown":
                raise ConfigError(f"data.sources[{s.name}]: license is required")
            if (s.min_score_field is None) != (s.min_score is None):
                raise ConfigError(f"data.sources[{s.name}]: set both min_score_field and min_score")


def _validate(cfg: Config) -> None:
    m, t = cfg.model, cfg.train
    _validate_model(m)
    _validate_sources(cfg.data)
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
    _validate_speech(cfg)
    _validate_identity(cfg)


def _validate_identity(cfg: Config) -> None:
    idc = cfg.identity
    if idc is None:
        return
    if (idc.path is None) == (idc.hf_dataset is None):
        raise ConfigError("identity: exactly one of identity.path or identity.hf_dataset is required")
    if not 0.0 <= idc.holdout_fraction < 1.0:
        raise ConfigError(f"identity.holdout_fraction must be in [0, 1), got {idc.holdout_fraction}")
    if idc.text_every < 1 or idc.speech_every < 1:
        raise ConfigError("identity.text_every and identity.speech_every must be >= 1")


def _validate_speech(cfg: Config) -> None:
    sp, au = cfg.speech, cfg.audio
    if sp is not None:
        if sp.max_seconds <= 0:
            raise ConfigError(f"speech.max_seconds must be > 0, got {sp.max_seconds}")
        if sp.hours <= 0:
            raise ConfigError(f"speech.hours must be > 0, got {sp.hours}")
        for k in ("val_chunks", "test_chunks", "keep_wavs"):
            if getattr(sp, k) < 0:
                raise ConfigError(f"speech.{k} must be >= 0, got {getattr(sp, k)}")
        if not 0.0 <= sp.qa_fraction <= 1.0:
            raise ConfigError(f"speech.qa_fraction must be in [0, 1], got {sp.qa_fraction}")
        if not 0.0 < sp.qa_max_wer <= 1.0:
            raise ConfigError(f"speech.qa_max_wer must be in (0, 1], got {sp.qa_max_wer}")
    if au is None:
        return
    if sp is None:
        raise ConfigError("the audio section requires a speech section (audio.* needs speech.*)")
    if au.codebooks < 1 or au.codebook_size < 1:
        raise ConfigError("audio.codebooks and audio.codebook_size must be >= 1")
    if au.acoustic_delay < 0:
        raise ConfigError(f"audio.acoustic_delay must be >= 0, got {au.acoustic_delay}")
    if au.depth_layers < 1:
        raise ConfigError(f"audio.depth_layers must be >= 1, got {au.depth_layers}")
    if au.depth_n_head < 1 or au.depth_d_model % au.depth_n_head != 0:
        raise ConfigError(
            f"audio.depth_d_model ({au.depth_d_model}) must be divisible by "
            f"audio.depth_n_head ({au.depth_n_head})"
        )
    if len(au.code_loss_weights) != au.codebooks:
        raise ConfigError(
            f"audio.code_loss_weights has {len(au.code_loss_weights)} entries but "
            f"audio.codebooks is {au.codebooks}"
        )
    if any(w < 0 for w in au.code_loss_weights) or au.text_loss_weight < 0:
        raise ConfigError("audio.code_loss_weights and audio.text_loss_weight must be >= 0")
    if au.text_every < 1:
        raise ConfigError(f"audio.text_every must be >= 1, got {au.text_every}")
    # one position per codec frame, plus the acoustic delay, plus BOS and END_AUDIO
    need = math.ceil(sp.max_seconds * MIMI_FRAME_RATE) + au.acoustic_delay + 2
    if need > cfg.model.context_length:
        raise ConfigError(
            f"model.context_length ({cfg.model.context_length}) is too small for "
            f"speech.max_seconds={sp.max_seconds}: needs {need} positions "
            f"(ceil({sp.max_seconds} * {MIMI_FRAME_RATE}) + audio.acoustic_delay "
            f"{au.acoustic_delay} + 2)"
        )


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

    d = conv(dataclasses.asdict(cfg))
    # Optional M2 sections are omitted when unset so M1 config dumps stay byte-identical.
    for k in ("speech", "audio", "identity"):
        if d.get(k) is None:
            d.pop(k, None)
    return d


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
