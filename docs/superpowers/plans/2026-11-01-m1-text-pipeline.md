# Milestone 1 Text Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A from-scratch, config-driven, fully tested text LM pipeline (data → BPE tokenizer → token shards → GPT → training with exact resume → generation → evaluation report) that trains a ~6M TinyStories model on a free GPU.

**Architecture:** A `src/llm` Python package exposed through one `llm` CLI. Each stage writes versioned artifacts with a `manifest.json`; downstream stages verify upstream checksums. One YAML file per experiment holds all sections (`data`, `tokenizer`, `shards`, `model`, `train`, `run`); each command reads the sections it needs.

**Tech Stack:** Python ≥3.10, PyTorch ≥2.2, numpy, pyyaml, regex, datasets (download only), matplotlib, pytest; optional wandb, huggingface_hub.

**Spec:** `docs/superpowers/specs/2026-11-01-m1-text-pipeline-design.md`

## Global Constraints

- Python `>=3.10`; `torch>=2.2`. No HF Transformers, Lightning, Hydra, or third-party tokenizer libraries in `src/llm`.
- Everything outside `tests/` reads its settings from config; no hardcoded vocab size, context length or model dims.
- Every artifact directory (`data/…`, `tokenizers/…`, `shards/…`) has a `manifest.json`; checksum mismatch raises `ProvenanceError` (never a warning).
- Shard dtype: `uint16` if `vocab_size <= 65536`, else `uint32`, recorded in the manifest.
- AdamW β=(0.9, 0.95), weight decay 0.1 on params with `ndim >= 2` only, grad clip 1.0, linear warmup then cosine to `min_lr_ratio = 0.1` of peak.
- Precision `auto`: bf16 if `torch.cuda.is_bf16_supported()`, else fp16 + GradScaler on CUDA, else fp32 on CPU.
- Init normal(0, 0.02); residual output projections (`attn.proj`, `mlp.proj`) normal(0, 0.02/√(2·n_layer)); lm_head weight-tied to token embedding.
- Writes of checkpoints and manifests are atomic (temp file + `os.replace`).
- Frozen prompt set version string `tinystories-v1`; samplers: greedy (temperature 0) and temperature 0.8 / top-k 50 / seed 1234; max 128 new tokens.
- Fast tests (`pytest -m "not slow"`) under ~1 min on CPU; slow tests marked `@pytest.mark.slow`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` only. **Never include session links (claude.ai/code URLs) in commits, PRs, or any file in the repo.**
- Checkpoints are loaded with `torch.load(..., weights_only=True)` (no arbitrary unpickling). Everything saved must therefore be tensors, dicts, lists, tuples or Python primitives; the NumPy RNG state's array is stored as a `torch.Tensor` and converted back on restore.

## Review Focus

1. **Truncated/corrupt latest checkpoint** (session killed during write or upload) → `llm resume` falls back to the newest *valid* checkpoint and logs a warning. Test: `test_load_latest_skips_corrupt_checkpoint` (Task 9).
2. **Config edited between run and resume** (e.g. user changes `train.lr` in `runs/<id>/config.yaml`) → resume refuses with a message naming the differing keys. Test: `test_resume_refuses_changed_config` (Task 11).
3. **`tokens_per_step` not divisible by `micro_batch_size × context_length`** → config load fails with a clear message, before any data/GPU work. Test: `test_tokens_per_step_must_divide` (Task 1).
4. **Evaluation split too small** (e.g. 20-doc overfit data where the val split is empty or < `context_length + 1` tokens) → clear error naming the split and token count at train start; overfit config evaluates on `train` via `train.eval_split`. Test: `test_chunks_too_small_raises` (Task 5).
5. **Prompt longer than the model's context** in `llm generate` → crop to the last `context_length` tokens instead of crashing. Test: `test_generate_crops_long_prompt` (Task 7).

---

## File Structure

```
pyproject.toml                       # package, deps, `llm` entry point, pytest markers
.gitignore                           # data/ tokenizers/ shards/ runs/ __pycache__ .venv
.github/workflows/ci.yml             # fast + slow tests on CPU
configs/m1-overfit.yaml
configs/m1-tinystories.yaml
src/llm/__init__.py
src/llm/config.py                    # dataclasses, load_config, overrides, validation
src/llm/manifest.py                  # sha256 helpers, write/read/verify manifests, atomic write
src/llm/data/__init__.py
src/llm/data/sources.py              # iter_source_docs
src/llm/data/clean.py                # normalize_text, clean_docs
src/llm/data/split.py                # assign_split
src/llm/data/prepare.py              # prepare_data (orchestrates + data_report.md)
src/llm/tokenizer/__init__.py
src/llm/tokenizer/bpe.py             # BPETokenizer
src/llm/shards.py                    # write_shards, TokenChunks, DataPosition, BatchIterator
src/llm/model/__init__.py
src/llm/model/gpt.py                 # GPT, manual_attention, expected_num_params
src/llm/generate.py                  # filter_logits, generate
src/llm/eval/__init__.py
src/llm/eval/metrics.py              # evaluate, bits_per_byte, repetition_rate
src/llm/eval/prompts.py              # PROMPT_SET_VERSION, PROMPTS, SAMPLERS, run_prompt_eval
src/llm/runs.py                      # create_run, git_info, hardware_info, append_jsonl
src/llm/train/__init__.py
src/llm/train/checkpoint.py          # TrainState, save/load/latest, RNG capture
src/llm/train/loop.py                # train(), lr_at, resolve_precision, make_optimizer
src/llm/report.py                    # build_report, compare_runs
src/llm/cli.py                       # argparse subcommands
tests/conftest.py                    # fixtures: tiny corpus path, tmp workspace, tiny config
tests/fixtures/stories.jsonl         # ~40 short hand-written stories, {"text": ...}
tests/test_*.py                      # one file per module, named in tasks
notebooks/m1_kaggle.ipynb            # GPU smoke test + m1-tinystories run with interrupt/resume
README.md
```

---

### Task 1: Project scaffold, config system, CI

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`, `src/llm/__init__.py`, `src/llm/config.py`, `src/llm/cli.py` (skeleton with subcommand registry), `tests/conftest.py`, `tests/fixtures/stories.jsonl`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Config` dataclass with fields `data: DataConfig, tokenizer: TokenizerConfig, shards: ShardConfig, model: ModelConfig, train: TrainConfig, run: RunConfig`.
  - `DataConfig(name: str, version: int, source: SourceConfig, min_chars: int = 20, max_chars: int = 20000, split: dict[str, float] = {"train": .98, "val": .01, "test": .01}, root: str = "data")`
  - `SourceConfig(type: Literal["hf", "local_jsonl"], path: str, split: str = "train", text_field: str = "text", max_docs: int | None = None)`
  - `TokenizerConfig(name: str, version: int, vocab_size: int, train_sample_bytes: int = 100_000_000, root: str = "tokenizers")`
  - `ShardConfig(name: str, version: int, shard_size_tokens: int = 100_000_000, root: str = "shards")`
  - `ModelConfig(n_layer: int, d_model: int, n_head: int, context_length: int, dropout: float = 0.0, attn_impl: Literal["sdpa", "manual"] = "sdpa", compile: bool = False)` — vocab size is **not** here; it is `cfg.tokenizer.vocab_size`.
  - `TrainConfig(seed: int, tokens_per_step: int, micro_batch_size: int, max_steps: int, lr: float, warmup_steps: int, min_lr_ratio: float = 0.1, weight_decay: float = 0.1, betas: tuple[float, float] = (0.9, 0.95), grad_clip: float = 1.0, precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto", device: str = "auto", log_every: int = 10, eval_every: int = 200, eval_batches: int = 20, eval_split: Literal["train", "val"] = "val", ckpt_every: int = 200, keep_last: int = 3, max_wall_time_s: float | None = None, cost_per_hour: float = 0.0, wandb_project: str | None = None, hub_repo: str | None = None)`
  - `RunConfig(name: str, root: str = "runs")`
  - `load_config(path: str | Path, overrides: list[str] = ()) -> Config` — overrides are `"section.key=value"` parsed with `yaml.safe_load` for the value.
  - `config_to_dict(cfg: Config) -> dict` (round-trips through `load_config`).
  - `ConfigError(Exception)`.
  - `grad_accum_steps(cfg: Config) -> int`.
  - `cli.main(argv: list[str] | None = None) -> int`; subcommands registered by later tasks.
  - pytest fixtures in `conftest.py`: `stories_path` (path to fixture), `workspace` (tmp dir, chdir), `tiny_config(workspace) -> Path` writing a YAML with `local_jsonl` source = fixture, `vocab_size: 300`, `n_layer: 2, d_model: 64, n_head: 2, context_length: 64`, `split: {train: 1.0, val: 0.0, test: 0.0}`, `eval_split: train`, `precision: fp32`, `device: cpu`, `tokens_per_step: 512, micro_batch_size: 8`.

- [ ] **Step 1: Write the fixture corpus.** `tests/fixtures/stories.jsonl`: 40 lines, each `{"text": "<3–6 simple sentences>"}` in TinyStories style, ~300–600 characters each, all distinct, plus 2 exact duplicates of earlier lines and 1 line with `"text": "Hi."` (too short) so cleaning tests have something to drop.

- [ ] **Step 2: Write failing tests** in `tests/test_config.py`:

```python
def test_load_roundtrip(tiny_config):
    cfg = load_config(tiny_config)
    assert cfg.model.n_layer == 2 and cfg.tokenizer.vocab_size == 300
    assert load_config_from_dict(config_to_dict(cfg)) == cfg   # helper may be private; or write YAML and reload

def test_unknown_key_rejected(workspace, tiny_config):
    # add model.n_layers (typo) → ConfigError mentioning "n_layers"

def test_missing_required_rejected(...):  # drop model.d_model → ConfigError mentioning "d_model"

def test_override(tiny_config):
    cfg = load_config(tiny_config, ["train.lr=0.01", "model.n_layer=3"])
    assert cfg.train.lr == 0.01 and cfg.model.n_layer == 3

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
```

- [ ] **Step 3: Run:** `pip install -e ".[dev]" && pytest tests/test_config.py -v` → FAIL (import errors).

- [ ] **Step 4: Implement** `pyproject.toml` (`[project.scripts] llm = "llm.cli:main"`, extras `dev = [pytest]`, `hub = [huggingface_hub]`, `wandb = [wandb]`; pytest markers `slow`), `.gitignore`, `config.py` (recursive dataclass construction from dicts; unknown/missing keys → `ConfigError` with dotted key path; validation: `d_model % n_head == 0`, split ratios sum to 1 within 1e-6, `tokens_per_step % (micro_batch_size * context_length) == 0`), and `cli.py` skeleton (argparse with subparsers; `main` returns exit code). CI workflow: ubuntu, Python 3.11, `pip install torch --index-url https://download.pytorch.org/whl/cpu`, `pip install -e ".[dev]"`, then `pytest -m "not slow"` and `pytest -m slow` as separate steps.

- [ ] **Step 5: Run:** `pytest tests/test_config.py -v` → PASS.

- [ ] **Step 6: Commit** `feat: project scaffold, typed config with overrides and validation, CI`.

---

### Task 2: Manifests and provenance

**Files:** Create `src/llm/manifest.py`, `tests/test_manifest.py`

**Interfaces:**
- Produces:
  - `ProvenanceError(Exception)`
  - `sha256_bytes(b: bytes) -> str`, `sha256_file(path: Path) -> str`
  - `atomic_write_bytes(path: Path, data: bytes) -> None`, `atomic_write_json(path: Path, obj: dict) -> None` (write `path.with_suffix(path.suffix + ".tmp")`, fsync, `os.replace`)
  - `write_manifest(dir: Path, kind: str, name: str, version: int, files: list[str], inputs: dict[str, str], extra: dict | None = None) -> dict` — stores `kind, name, version, created_at, git_sha, files: {relpath: sha256}, inputs: {label: upstream_manifest_checksum}, extra`, and returns it; also writes the manifest's own checksum as `manifest_checksum(dir)`.
  - `read_manifest(dir: Path) -> dict`
  - `manifest_checksum(dir: Path) -> str` (sha256 of the manifest file bytes)
  - `verify_files(dir: Path) -> None` — recomputes file hashes; mismatch/missing → `ProvenanceError` naming the file.
  - `verify_input(manifest: dict, label: str, upstream_dir: Path) -> None` — `manifest["inputs"][label] != manifest_checksum(upstream_dir)` → `ProvenanceError` naming both dirs.
  - `git_sha() -> tuple[str | None, bool]` (sha, dirty); `None` outside a repo.

- [ ] **Step 1: Write failing tests:** `test_manifest_roundtrip`; `test_verify_files_detects_modification` (edit a listed file → `ProvenanceError` matching the filename); `test_verify_files_detects_missing`; `test_verify_input_mismatch` (rewrite upstream manifest → `ProvenanceError`); `test_atomic_write_leaves_no_tmp` (no `*.tmp` remains after success).
- [ ] **Step 2: Run** `pytest tests/test_manifest.py -v` → FAIL.
- [ ] **Step 3: Implement** `manifest.py` as specified (`git_sha` via `subprocess.run(["git", "rev-parse", "HEAD"])` and `git status --porcelain`; swallow errors → `(None, False)`).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: artifact manifests with checksums and provenance checks`.

---

### Task 3: Data preparation (sources, cleaning, split, report) + `llm prepare-data`

**Files:** Create `src/llm/data/{__init__,sources,clean,split,prepare}.py`, `tests/test_data.py`; Modify `src/llm/cli.py`

**Interfaces:**
- Consumes: `Config`, `DataConfig`, `write_manifest`, `atomic_write_json`.
- Produces:
  - `iter_source_docs(src: SourceConfig) -> Iterator[str]` — `local_jsonl`: read lines, yield `obj[text_field]`; `hf`: `datasets.load_dataset(path, split=split, streaming=True)`; both stop at `max_docs`.
  - `normalize_text(s: str) -> str` — `unicodedata.normalize("NFC")`, convert `\r\n`/`\r` to `\n`, strip.
  - `CleanStats` dataclass: `received: int`, `kept: int`, `dropped: dict[str, int]` (reasons: `"invalid_utf8"`, `"too_short"`, `"too_long"`, `"exact_duplicate"`).
  - `clean_docs(docs: Iterable[str], min_chars: int, max_chars: int) -> tuple[list[str], CleanStats]` — invalid UTF-8 = `s.encode("utf-8")` raises (lone surrogates); dedup by sha256 of normalized text; invariant `received == kept + sum(dropped.values())`.
  - `assign_split(text: str, ratios: dict[str, float]) -> str` — `h = int(sha256(text.encode()).hexdigest()[:16], 16) % 1000`; iterate splits in order `train, val, test` with cumulative thresholds `round(cum_ratio * 1000)`.
  - `prepare_data(cfg: Config) -> Path` — writes `<root>/<name>-v<version>/{train,val,test}.jsonl` (`{"text": ...}` lines; empty split → empty file), `data_report.md`, `manifest.json` (kind `"data"`, `extra` = source spec, license string from source config if provided else `"unknown"`, clean stats, split doc counts and UTF-8 byte counts).
  - CLI: `llm prepare-data <config> [--set k=v ...]`.
- Add to `SourceConfig`: `license: str = "unknown"`, `url: str | None = None` (configs/m1-* set `license: CDLA-Sharing-1.0`, `url: https://huggingface.co/datasets/roneneldan/TinyStories`).

- [ ] **Step 1: Write failing tests:**

```python
def test_clean_counts_sum(stories_path): # stats.received == stats.kept + sum(stats.dropped.values())
                                         # dropped["exact_duplicate"] == 2, dropped["too_short"] == 1
def test_clean_drops_lone_surrogate():   # clean_docs(["ok text long enough...", "bad \ud800 text ..."], 5, 1000) drops 1 as "invalid_utf8"
def test_normalize_nfc_and_newlines():   # normalize_text("é\r\nx ") == "é\nx"
def test_split_deterministic_and_disjoint(): # 1000 random strings: same result twice; each in exactly one split;
                                             # ratios {train:.8,val:.1,test:.1} → train count within 700..900
def test_split_all_train():              # ratios {train:1.0,val:0,test:0} → always "train"
def test_prepare_data_outputs(tiny_config): # files exist; manifest verify_files passes;
                                            # report contains "exact_duplicate" and the license string
def test_cli_prepare_data(tiny_config):  # cli.main(["prepare-data", str(tiny_config)]) == 0
```

- [ ] **Step 2: Run** `pytest tests/test_data.py -v` → FAIL.
- [ ] **Step 3: Implement** the modules and CLI subcommand. The data report is Markdown with a table: received, each drop reason, kept, then per-split docs/bytes, source, URL, license, download date.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: data preparation with counted cleaning, deterministic split, data report`.

---

### Task 4: Byte-level BPE tokenizer + `llm train-tokenizer`

**Files:** Create `src/llm/tokenizer/{__init__,bpe}.py`, `tests/test_tokenizer.py`; Modify `src/llm/cli.py`

**Interfaces:**
- Consumes: `TokenizerConfig`, data dir from Task 3, manifest helpers.
- Produces:
  - `GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""` (compiled with `regex`)
  - `class BPETokenizer`:
    - ID layout: `0..255` raw bytes; `256 .. vocab_size-2` merges in creation order; `vocab_size-1` = `<|endoftext|>`. So `num_merges = vocab_size - 257`.
    - `@classmethod train(cls, texts: Iterable[str], vocab_size: int) -> BPETokenizer`
    - `encode(self, text: str) -> list[int]` (never emits the special id; literal `"<|endoftext|>"` in text is encoded as ordinary bytes)
    - `decode(self, ids: list[int]) -> str` (`errors="replace"`; eot id decodes to `""`)
    - `eot_id: int` property, `vocab_size: int` property
    - `save(self, dir: Path) -> None` (writes `tokenizer.json` with `{"vocab_size", "merges": [[a, b], ...], "pattern"}`), `@classmethod load(cls, dir: Path) -> BPETokenizer`
  - `TokenizerTrainingError(Exception)` — raised when no pair occurs ≥2 times before reaching `vocab_size`.
  - `train_tokenizer(cfg: Config) -> Path` — reads `train.jsonl` of `cfg.data` up to `train_sample_bytes`, trains, saves to `<root>/<name>-v<version>/`, writes manifest (kind `"tokenizer"`, `inputs={"data": manifest_checksum(data_dir)}`, `extra` = vocab size, sample bytes, training seconds, bytes/token on the sample).
  - CLI: `llm train-tokenizer <config> [--set ...]`.

Training algorithm (not determined by the tests, so it is fixed here): pre-split each text with the pattern; count unique chunks → `Counter[bytes-tuple]`; repeat `num_merges` times: count adjacent pairs weighted by chunk frequency, pick the max count with **ties broken by the smallest `(a, b)` pair**, assign the next id, replace the pair in every chunk. Encoding: per chunk, repeatedly merge the adjacent pair with the lowest merge rank until none applies; memoise chunk→ids in a dict.

- [ ] **Step 1: Write failing tests:**

```python
@pytest.fixture
def tok(stories_texts): return BPETokenizer.train(stories_texts, vocab_size=300)

@pytest.mark.parametrize("s", ["", " ", "Hello, world!", "  multiple   spaces\n\n", "naïve café ☕ 日本語 🙂",
                               "def f(x):\n    return x**2  # code", "3.14159 1,000,000 2026-11-01", "<|endoftext|>"])
def test_roundtrip(tok, s): assert tok.decode(tok.encode(s)) == s

def test_roundtrip_random_unicode(tok):  # 200 random strings from chr(32..0x2FFF) + emoji, seed 0

def test_id_layout(tok):  # tok.vocab_size == 300; tok.eot_id == 299; max(tok.encode(any text)) < 299

def test_training_deterministic(stories_texts):  # two trainings → identical tokenizer.json bytes

def test_save_load_identical(tok, tmp_path):  # load(save) encodes 5 strings identically

def test_compression(tok, stories_texts):  # bytes/token on fixture text > 1.5

def test_too_small_corpus_raises():
    with pytest.raises(TokenizerTrainingError): BPETokenizer.train(["ab"], vocab_size=300)

def test_cli_train_tokenizer(tiny_config):  # prepare-data then train-tokenizer exit 0; manifest inputs.data matches
```

- [ ] **Step 2: Run** `pytest tests/test_tokenizer.py -v` → FAIL.
- [ ] **Step 3: Implement** `bpe.py`, `train_tokenizer`, CLI. Add `regex` to dependencies.
- [ ] **Step 4: Run** → PASS (whole file under ~15 s; if training on the fixture is slower, the pair-count loop is wrong — profile before moving on).
- [ ] **Step 5: Commit** `feat: from-scratch byte-level BPE tokenizer`.

---

### Task 5: Token shards, chunk reader, deterministic batching + `llm tokenize`

**Files:** Create `src/llm/shards.py`, `tests/test_shards.py`; Modify `src/llm/cli.py`

**Interfaces:**
- Consumes: `BPETokenizer.load`, data split files, manifest helpers.
- Produces:
  - `shard_dtype(vocab_size: int) -> np.dtype` (`uint16` if `<= 65536` else `uint32`)
  - `write_shards(cfg: Config) -> Path` — for each split: encode each doc, append `eot_id` **after** each doc, write `<split>_<k:03d>.bin` files of at most `shard_size_tokens` tokens (`ndarray.tofile`); manifest kind `"shards"`, `inputs={"tokenizer": ..., "data": ...}`, `extra={"dtype", "vocab_size", "splits": {split: {"tokens": int, "bytes": int, "files": [...]}}}` where `bytes` = total UTF-8 bytes of that split's doc texts. Empty splits get `tokens: 0, files: []`.
  - `class TokenChunks(shard_dir: Path, split: str, seq_len: int)` — verifies manifest files; memmaps and concatenates the split's files; `__len__ = (n_tokens - 1) // seq_len`; `__getitem__(i) -> np.ndarray` of length `seq_len + 1` = `tokens[i*seq_len : i*seq_len + seq_len + 1]`; raises `ValueError` mentioning split name and token count if `len == 0`. Attribute `bytes_per_token: float` from the manifest.
  - `@dataclass(frozen=True) DataPosition(epoch: int = 0, index: int = 0)`
  - `epoch_order(n: int, seed: int, epoch: int) -> np.ndarray` = `np.random.default_rng([seed, epoch]).permutation(n)`
  - `next_batch(chunks: TokenChunks, pos: DataPosition, batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, DataPosition]` — takes the next `batch_size` chunk indices from the epoch order, rolling into the next epoch when exhausted; returns `x = batch[:, :-1]`, `y = batch[:, 1:]` as int64 tensors.
  - CLI: `llm tokenize <config>` (verifies the tokenizer manifest's `inputs.data` equals the current data manifest checksum).

- [ ] **Step 1: Write failing tests:**

```python
def test_dtype_rule(): assert shard_dtype(65536) == np.uint16 and shard_dtype(65537) == np.uint32
def test_write_read_roundtrip(prepared_tokenized):   # concatenated tokens decode back to docs joined by eot
def test_eot_between_docs(...):                      # count of eot_id == number of train docs
def test_multiple_shard_files(tiny_config):          # --set shards.shard_size_tokens=1000 → >1 train file; reader identical
def test_manifest_mismatch_hard_error(...):          # retrain tokenizer with different vocab after tokenize, then
                                                     # constructing the stage check raises ProvenanceError
def test_same_seed_same_order(): epoch_order(100, 7, 0) equal twice; differs for epoch 1
def test_resume_position_identical(chunks):          # sequence of 10 batches from start == 4 batches, then 6 more from returned pos
def test_epoch_rollover(chunks):                     # batch spanning end of epoch returns pos.epoch == 1
def test_chunks_too_small_raises(...):               # TokenChunks on an empty "val" split → ValueError matching "val"
```

- [ ] **Step 2: Run** `pytest tests/test_shards.py -v` → FAIL.
- [ ] **Step 3: Implement** `shards.py` and the CLI subcommand. Add a `prepared_tokenized` fixture to `conftest.py` that runs prepare-data → train-tokenizer → tokenize on `tiny_config` and returns `(cfg, shard_dir)`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: versioned token shards with deterministic, resumable batching`.

---

### Task 6: GPT model

**Files:** Create `src/llm/model/{__init__,gpt}.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `ModelConfig`, `vocab_size: int`.
- Produces:
  - `class GPT(nn.Module)`: `__init__(self, cfg: ModelConfig, vocab_size: int)`; `forward(self, idx: LongTensor[B,T], targets: LongTensor[B,T] | None = None) -> tuple[Tensor[B,T,V], Tensor | None]` (loss = mean cross-entropy over all positions; `T > context_length` → `ValueError`); `num_params(self, include_embeddings: bool = True) -> int` (non-embedding excludes `tok_emb` and `pos_emb`; tied lm_head counted once).
  - Submodule names (used by init and optimizer grouping): `tok_emb`, `pos_emb`, `blocks[i].ln1`, `blocks[i].attn.qkv`, `blocks[i].attn.proj`, `blocks[i].ln2`, `blocks[i].mlp.fc`, `blocks[i].mlp.proj`, `ln_f`, `lm_head` (no bias; `lm_head.weight is tok_emb.weight`). All Linear layers except `lm_head` have bias.
  - `manual_attention(q, k, v) -> Tensor` (shapes `[B, H, T, Dh]`; causal mask via `torch.triu(..., diagonal=1)` filled with `-inf`).
  - `expected_num_params(cfg: ModelConfig, vocab_size: int) -> int` = `V*d + T*d + L*(12*d*d + 13*d) + 2*d`.

- [ ] **Step 1: Write failing tests:**

```python
CFG = ModelConfig(n_layer=2, d_model=64, n_head=2, context_length=64)
def test_output_shape(): logits, loss = GPT(CFG, 300)(torch.randint(0, 300, (3, 17))); assert logits.shape == (3, 17, 300) and loss is None
def test_param_count(): assert GPT(CFG, 300).num_params() == expected_num_params(CFG, 300)
def test_weight_tying(): m = GPT(CFG, 300); assert m.lm_head.weight is m.tok_emb.weight
def test_causal_mask():  # x2 = x.clone(); x2[:, 10] = (x[:, 10] + 1) % 300 → logits[:, :10] identical (atol 1e-6),
                         # logits[:, 10:] differ; run for attn_impl "sdpa" and "manual"
def test_sdpa_matches_manual():  # same weights (load_state_dict), eval mode → logits allclose atol 1e-5
def test_initial_loss_near_ln_vocab():  # torch.manual_seed(0); random targets → abs(loss - math.log(300)) < 0.3
def test_too_long_sequence_raises():  # T = 65 → ValueError
def test_m1_tinystories_size():  # expected_num_params(ModelConfig(6, 256, 8, 256), 4096) between 5.5e6 and 6.5e6
```

- [ ] **Step 2: Run** `pytest tests/test_model.py -v` → FAIL.
- [ ] **Step 3: Implement** `gpt.py` (pre-LN blocks, GELU (tanh approximation) MLP `d → 4d → d`, init per Global Constraints, `attn_impl` switch, dropout on attention output and MLP output when `dropout > 0`).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: GPT-2-style decoder-only transformer with reference attention`.

---

### Task 7: Generation + `llm generate`

**Files:** Create `src/llm/generate.py`, `tests/test_generate.py`; Modify `src/llm/cli.py`

**Interfaces:**
- Consumes: `GPT`, `BPETokenizer`.
- Produces:
  - `filter_logits(logits: Tensor[V], top_k: int | None, top_p: float | None) -> Tensor[V]` — disallowed entries set to `-inf`; top-p keeps the smallest set of highest-probability tokens whose cumulative probability ≥ `top_p` (always at least one).
  - `apply_repetition_penalty(logits: Tensor[V], prev_ids: list[int], penalty: float) -> Tensor[V]` — for ids in `prev_ids`: positive logits divided by `penalty`, negative multiplied.
  - `@torch.no_grad() generate(model: GPT, prompt_ids: list[int], max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None, repetition_penalty: float = 1.0, stop_ids: Iterable[int] = (), seed: int | None = None) -> list[int]` — returns **only new tokens**; context cropped to the last `context_length` tokens each step; `temperature == 0` → argmax; sampling uses a `torch.Generator` seeded with `seed`; a stop id ends generation and is **not** included.
  - CLI: `llm generate <checkpoint_path_or_run_dir> --prompt TEXT [--max-new-tokens 128 --temperature 0.8 --top-k 50 --top-p --repetition-penalty --seed]` (implemented once Task 9 provides checkpoint loading; until then the subcommand is registered in Task 9 Step 3).

- [ ] **Step 1: Write failing tests:**

```python
def test_same_seed_same_output(model): generate(m, [1,2,3], 20, temperature=1.0, seed=5) twice → equal
def test_greedy_is_argmax(model):  # temperature=0 output == manual argmax loop for 5 tokens
def test_top_k_only_allowed():     # logits = arange(10.); filter_logits(..., top_k=3, None) finite only at 7,8,9
def test_top_p_minimal_set():      # probs [0.5, 0.3, 0.15, 0.05] (as log); top_p=0.8 → finite at indices 0,1 only
def test_stops_at_stop_token(model):  # force eot: set lm_head bias? Instead: monkeypatch model.forward to return logits
                                      # peaked at id 7 → stop_ids=[7] → returns []
def test_max_length(model): len(generate(m, [1], 13, temperature=1.0, seed=0)) == 13
def test_generate_crops_long_prompt(model): generate(m, list(range(100)) (context 64), 5, temperature=0) → 5 tokens, no error
def test_repetition_penalty(): logits [2., -2.], prev [0, 1], penalty 2 → [1., -4.]
```

- [ ] **Step 2: Run** `pytest tests/test_generate.py -v` → FAIL.
- [ ] **Step 3: Implement** `generate.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: sampling with temperature, top-k, top-p, repetition penalty, stop tokens`.

---

### Task 8: Evaluation metrics and frozen prompts

**Files:** Create `src/llm/eval/{__init__,metrics,prompts}.py`, `tests/test_eval.py`

**Interfaces:**
- Consumes: `GPT`, `TokenChunks`, `epoch_order`, `generate`, `BPETokenizer`.
- Produces:
  - `bits_per_byte(mean_loss_nats: float, bytes_per_token: float) -> float` = `mean_loss_nats / (math.log(2) * bytes_per_token)`.
  - `repetition_rate(ids: list[int], n: int = 4) -> float` = `1 - unique_ngrams / total_ngrams` (0.0 if fewer than `n` tokens).
  - `@torch.no_grad() evaluate(model: GPT, chunks: TokenChunks, num_batches: int, batch_size: int, device: torch.device, autocast_ctx) -> dict` → `{"val_loss", "bpb", "ppl"}`; uses the **first** `num_batches * batch_size` chunks in fixed order `epoch_order(len, seed=0, epoch=0)` (capped at `len(chunks)`), so every evaluation sees the same data. Restores `model.train()` afterwards.
  - `PROMPT_SET_VERSION = "tinystories-v1"`; `PROMPTS: list[str]` with exactly 10 TinyStories-style openings (e.g. `"Once upon a time, there was a little dog named"`, `"Lily wanted to go to the park, but"`, …); `SAMPLERS = [{"name": "greedy", "temperature": 0.0}, {"name": "t0.8_k50", "temperature": 0.8, "top_k": 50, "seed": 1234}]`; `MAX_NEW_TOKENS = 128`.
  - `run_prompt_eval(model: GPT, tok: BPETokenizer, step: int, out_path: Path) -> dict` — writes one JSON line per (prompt, sampler) with `step, prompt_set, prompt, sampler, completion, repetition`, and returns `{"repetition_mean": float}`.

- [ ] **Step 1: Write failing tests:** `test_bpb_formula` (`bits_per_byte(math.log(2) * 4, 4.0) == 1.0`); `test_repetition_rate` (`[1,2,3,4]*3` → `1 - 4/9`; `[1,2,3]` → `0.0`); `test_evaluate_deterministic` (two calls equal); `test_evaluate_restores_train_mode`; `test_prompt_eval_writes_all` (20 lines; every line has `prompt_set == "tinystories-v1"`); `test_prompts_frozen` (`len(PROMPTS) == 10` and sha256 of `"\n".join(PROMPTS)` equals a constant committed in the test — editing prompts without bumping the version fails CI).
- [ ] **Step 2: Run** `pytest tests/test_eval.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: val loss, bits-per-byte, repetition metrics and frozen prompt eval`.

---

### Task 9: Run directories and checkpointing

**Files:** Create `src/llm/runs.py`, `src/llm/train/{__init__,checkpoint}.py`, `tests/test_checkpoint.py`; Modify `src/llm/cli.py` (register `generate`)

**Interfaces:**
- Consumes: `Config`, `config_to_dict`, `git_sha`, `atomic_write_*`, `GPT`.
- Produces:
  - `create_run(cfg: Config) -> Path` — `run_id = f"{cfg.run.name}-{UTC %Y%m%d-%H%M%S}-{sha[:7] or 'nogit'}"` (append `-2`, `-3` if the dir exists); writes `config.yaml` (resolved config), `run.json` (`run_id, created_at, git_sha, git_dirty, hardware`), `limitations.md` (template text starting `"<!-- TODO: describe known limitations -->"`), dirs `checkpoints/`, `samples/`.
  - `hardware_info() -> dict` (python, torch, cuda available, device name, CPU count).
  - `append_jsonl(path: Path, record: dict) -> None`.
  - `@dataclass TrainState(step: int = 0, tokens_seen: int = 0, epoch: int = 0, index: int = 0, wall_time_s: float = 0.0, cost_usd: float = 0.0, best_val_loss: float = math.inf)`
  - `get_rng_state() -> dict`, `set_rng_state(state: dict) -> None` (Python `random`, NumPy, torch CPU, all CUDA devices if available). NumPy's state array is stored as a `torch.Tensor` so the payload loads under `weights_only=True`.
  - `save_checkpoint(run_dir: Path, model, optimizer, scaler: torch.amp.GradScaler | None, state: TrainState, provenance: dict, keep_last: int, is_best: bool) -> Path` — file `checkpoints/step_{step:08d}.pt` containing `model, optimizer, scaler, train_state (asdict), rng, config (dict), provenance, git_sha, git_dirty`; written atomically; then `checkpoints/latest.json = {"path": "step_XXXXXXXX.pt", "step": n}` atomically; if `is_best`, copy to `best.pt` atomically; delete older `step_*.pt` beyond `keep_last` (never `best.pt`).
  - `load_checkpoint(path: Path, model, optimizer=None, scaler=None, restore_rng: bool = True) -> tuple[TrainState, dict]` — returns the state and the raw payload (for config/provenance checks). Uses `torch.load(path, map_location="cpu", weights_only=True)`.
  - `find_latest_valid(run_dir: Path) -> Path | None` — try the `latest.json` target, then remaining `step_*.pt` newest first; a file that fails `torch.load` is skipped with a `warnings.warn` naming it.
  - `load_model_for_inference(path_or_run_dir: Path) -> tuple[GPT, BPETokenizer, dict]` — accepts a `.pt` or a run dir (uses `find_latest_valid`); rebuilds `GPT` from the checkpoint's config; loads the tokenizer from the config's tokenizer dir.
  - CLI `llm generate` (spec in Task 7) using `load_model_for_inference`.
  - Optional Hub upload: if `cfg.train.hub_repo`, after each save call `huggingface_hub.upload_file` for the checkpoint and `latest.json`; any exception → `warnings.warn`, training continues.

- [ ] **Step 1: Write failing tests:**

```python
def test_create_run_files(tiny_config):  # config.yaml reloads to equal cfg; run.json has git_sha key; limitations.md exists
def test_save_load_identical_outputs(tmp_path):  # model+AdamW after 2 steps; save; fresh model/opt load; logits equal exactly,
                                                 # optimizer state tensors equal
def test_rng_restored(tmp_path):  # save; draw torch.rand(3), np.random.rand(3), random.random(); load; draws repeat exactly
def test_keep_last_and_best(tmp_path):  # save steps 1..5 keep_last=2, best at step 2 → step_4, step_5, best.pt remain
def test_latest_json_points_to_newest(tmp_path)
def test_load_latest_skips_corrupt_checkpoint(tmp_path):  # truncate newest file to 10 bytes → find_latest_valid returns
                                                          # previous step, with a warning (pytest.warns)
def test_no_tmp_files_left(tmp_path)
```

- [ ] **Step 2: Run** `pytest tests/test_checkpoint.py -v` → FAIL.
- [ ] **Step 3: Implement** `runs.py`, `checkpoint.py`, and the `generate` CLI subcommand.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: run directories and atomic full-state checkpoints`.

---

### Task 10: Training loop core + `llm train`

**Files:** Create `src/llm/train/loop.py`, `tests/test_train.py`; Modify `src/llm/cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `lr_at(step: int, cfg: TrainConfig) -> float` — `step < warmup_steps`: `lr * (step + 1) / warmup_steps`; then cosine from `lr` to `lr * min_lr_ratio` over `[warmup_steps, max_steps]`; `step >= max_steps` → `lr * min_lr_ratio`.
  - `resolve_device(setting: str) -> torch.device` (`auto` → cuda if available else cpu).
  - `resolve_precision(setting: str, device: torch.device) -> Literal["bf16", "fp16", "fp32"]` (rules in Global Constraints; explicit `bf16`/`fp16` on CPU → `ConfigError`).
  - `make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW` — two param groups: `ndim >= 2` with `weight_decay`, others with 0; `fused=True` only on CUDA.
  - `train(cfg: Config, run_dir: Path, resume: bool = False, stop_at_step: int | None = None) -> tuple[TrainState, str]` — returns the final state and a stop reason in `{"max_steps", "stop_at_step", "wall_time", "signal", "nan"}`. Per optimizer step: `grad_accum_steps(cfg)` micro-batches via `next_batch(..., batch_size=micro_batch_size, seed=cfg.train.seed)`, loss divided by accum steps, autocast per precision, GradScaler for fp16, `clip_grad_norm_(…, grad_clip)`, set LR via `lr_at`, `optimizer.step()`, `zero_grad(set_to_none=True)`. Every `log_every` steps append to `metrics.jsonl`: `step, tokens_seen, loss, lr, grad_norm, tokens_per_s, peak_mem_mb, elapsed_s, cost_usd`. Every `eval_every` steps and at the end: `evaluate` on `cfg.train.eval_split` → append `{"step", "type": "eval", "val_loss", "bpb", "ppl", "repetition_mean"}`; `run_prompt_eval` to `samples/step_<n>.jsonl`. Checkpoint every `ckpt_every` steps and at every stop. `cost_usd += elapsed_hours * cost_per_hour`. Seeds `random`, `numpy`, `torch` from `cfg.train.seed` at fresh start. Optional W&B mirror (`wandb.init(project=…, id=run_id, resume="allow")`, `wandb.log` the same records); import errors → warning.
  - CLI: `llm train <config> [--set ...] [--stop-at-step N]` → prints the run dir and the stop reason; exit 0.

Before starting, `train` verifies the provenance chain: shard manifest files (`verify_files`), shard `inputs.tokenizer` against the tokenizer dir, and the tokenizer's vocab size against `cfg.tokenizer.vocab_size`. It constructs `TokenChunks` for `train` and `cfg.train.eval_split` (Task 5 raises clearly if too small).

- [ ] **Step 1: Write failing tests:**

```python
def test_lr_schedule():  # TrainConfig(lr=1.0, warmup_steps=10, max_steps=110, ...): lr_at(0)=0.1, lr_at(9)=1.0,
                         # lr_at(10)=1.0, lr_at(60)≈0.55, lr_at(110)=0.1, lr_at(500)=0.1
def test_resolve_precision_cpu(): resolve_precision("auto", torch.device("cpu")) == "fp32"; "fp16" on cpu → ConfigError
def test_weight_decay_groups(): biases/LayerNorm params in the wd=0 group, all ndim>=2 in the wd=0.1 group, union == all params
def test_overfit(prepared_tokenized):  # cfg with max_steps=1500, lr=3e-3, warmup 20, eval_every 500 →
                                       # final train loss (last metrics.jsonl "loss") < 0.1; stop reason "max_steps"
def test_two_runs_identical(prepared_tokenized):  # max_steps 30, two fresh runs → identical "loss" sequences
def test_metrics_and_samples_written(...):  # metrics.jsonl has eval records; samples/step_*.jsonl exist
def test_cli_train(tiny_config)  # full CLI chain; exit 0
```

If `test_overfit` does not reach < 0.1, tune `lr`/`max_steps` in the test config (CPU time budget ≤ 60 s); do **not** relax the threshold.

- [ ] **Step 2: Run** `pytest tests/test_train.py -v` → FAIL.
- [ ] **Step 3: Implement** `loop.py` and CLI `train`. Mark `test_overfit` `@pytest.mark.slow`.
- [ ] **Step 4: Run** `pytest tests/test_train.py -v` (including slow) → PASS.
- [ ] **Step 5: Commit** `feat: training loop with mixed precision, accumulation, schedule, eval and logging`.

---

### Task 11: Exact resume, wall-time/signal stop, NaN guard + `llm resume`

**Files:** Modify `src/llm/train/loop.py`, `src/llm/cli.py`; Create `tests/test_resume.py`

**Interfaces:**
- Produces:
  - `train(..., resume=True)` path: `find_latest_valid(run_dir)`; load model/optimizer/scaler/RNG/`TrainState`; compare `config_to_dict(cfg)` with the checkpoint's config → any difference raises `ConfigError` listing the differing dotted keys; verify the checkpoint's provenance checksums against the current manifests → `ProvenanceError`; continue from `DataPosition(state.epoch, state.index)`; append `{"type": "resume", "step", "at"}` to `resume_log.jsonl`.
  - Wall-time stop: before each step, if `max_wall_time_s` and `state.wall_time_s + session_elapsed >= max_wall_time_s` → checkpoint, return `"wall_time"`.
  - Signals: `SIGTERM`/`SIGINT` handlers set a flag; the loop checks it after each optimizer step → checkpoint, return `"signal"`; the previous handlers are restored on exit.
  - NaN guard: non-finite loss in any micro-batch → do not step, do not checkpoint, append `{"type": "error", "reason": "non-finite loss", "step"}` to metrics, return `"nan"` (the last good checkpoint remains `latest`).
  - `stop_at_step=N`: after completing step N, checkpoint and return `"stop_at_step"`.
  - CLI: `llm resume <run_dir> [--stop-at-step N]` (loads `run_dir/config.yaml`).

- [ ] **Step 1: Write failing tests:**

```python
def test_resume_bitwise_equal(prepared_tokenized):
    # run A: fresh, max_steps=40 → losses_A, final model state_dict
    # run B: fresh with stop_at_step=17, then train(cfg, runB, resume=True) → losses_B
    # assert losses_A == losses_B (exact float equality, CPU) and all state_dict tensors torch.equal
def test_resume_log_written(...)
def test_resume_refuses_changed_config(...):  # stop at 5; edit run_dir/config.yaml train.lr → ConfigError match "train.lr"
def test_resume_refuses_changed_tokenizer(...):  # stop at 5; retrain tokenizer + retokenize → ProvenanceError
def test_wall_time_stop(...):  # max_wall_time_s=0.001 → reason "wall_time", a checkpoint exists
def test_signal_stop(...):     # monkeypatch so the handler flag is set after step 3 (call the installed handler) →
                               # reason "signal", latest checkpoint step == 3
def test_nan_guard(...):       # monkeypatch GPT.forward to return loss=nan from step 6 with ckpt_every=5 →
                               # reason "nan"; latest checkpoint step == 5; metrics has an "error" record
def test_fp16_path_runs(...):  # if torch.cuda.is_available(): precision fp16, 5 steps, finite losses; else pytest.skip
def test_cli_resume(...)
```

- [ ] **Step 2: Run** `pytest tests/test_resume.py -v` → FAIL.
- [ ] **Step 3: Implement.** For bitwise equality on CPU, set `torch.use_deterministic_algorithms(True)` when `device.type == "cpu"`, and make sure every source of randomness (dropout, data order) is either in the checkpointed RNG state or derived from `(seed, epoch)`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: exact resume, wall-time and signal stops, NaN guard`.

---

### Task 12: Report and compare

**Files:** Create `src/llm/report.py`, `tests/test_report.py`; Modify `src/llm/cli.py`

**Interfaces:**
- Consumes: run dir layout (Tasks 9–11), manifests (data, tokenizer, shards).
- Produces:
  - `build_report(run_dir: Path) -> Path` — writes `report.md` + `charts/{loss,bpb,lr,grad_norm,throughput}.png` (matplotlib `Agg`), sections in this order: Run card, Data card, Tokenizer card, Charts, Results, Sample progression (each prompt × `greedy` sampler at the first, middle and last `samples/step_*.jsonl`), Resume log, Known limitations (contents of `limitations.md`). If `limitations.md` still starts with the TODO template marker, emit `warnings.warn("limitations.md not filled in")` and include a visible "⚠ Not yet written" line.
  - `compare_runs(run_a: Path, run_b: Path, out_dir: Path) -> Path` — `comparison.md` with a table of final/best val loss, BPB, ppl, repetition, tokens, wall time, cost for both runs plus the config keys that differ; overlaid loss and BPB charts.
  - CLI: `llm report <run_dir>`, `llm compare <run_a> <run_b> [--out DIR]` (default `runs/compare-<a>-vs-<b>`).

- [ ] **Step 1: Write failing tests:** `test_report_sections` (train 30 steps; every section heading present; 5 PNGs exist); `test_report_warns_unfilled_limitations` (`pytest.warns`); `test_report_includes_resume_log` (stop + resume run → report mentions the resume step); `test_compare_lists_config_diff` (two runs differing only in `train.lr` → `comparison.md` contains `train.lr`).
- [ ] **Step 2: Run** `pytest tests/test_report.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: run report and run comparison`.

---

### Task 13: End-to-end test, real configs, README, Kaggle notebook

**Files:** Create `tests/test_e2e.py`, `configs/m1-overfit.yaml`, `configs/m1-tinystories.yaml`, `README.md`, `notebooks/m1_kaggle.ipynb`

**Interfaces:**
- Consumes: the CLI only.

Config values:

| key | `m1-overfit` | `m1-tinystories` |
|---|---|---|
| data.source | hf `roneneldan/TinyStories`, `max_docs: 20` | hf `roneneldan/TinyStories`, all |
| data.split | train 1.0 / val 0 / test 0 | 0.98 / 0.01 / 0.01 |
| tokenizer.vocab_size | 512 | 4096 |
| model (L, d, H, T) | 2, 64, 2, 64 | 6, 256, 8, 256 |
| train.tokens_per_step / micro_batch_size | 512 / 8 | 65536 / 64 |
| train.max_steps / lr / warmup_steps | 1500 / 3e-3 / 20 | 3000 / 1e-3 / 200 |
| train.eval_split / eval_every / ckpt_every | train / 500 / 500 | val / 250 / 250 |
| train.precision / device | fp32 / auto | auto / auto |
| train.max_wall_time_s | null | 41400 (11.5 h) |
| run.name | m1-overfit | m1-tinystories |

- [ ] **Step 1: Write the failing slow test** `test_full_pipeline_cli(workspace, tiny_config)`: via `cli.main` run `prepare-data`, `train-tokenizer`, `tokenize`, `train --stop-at-step 20 --set train.max_steps=40`, `resume <run_dir>`, `generate <run_dir> --prompt "Once upon a time" --max-new-tokens 20 --seed 1`, `report <run_dir>` — all return 0; `report.md` exists and mentions the resume; generated text is non-empty. Also `test_configs_load`: both real configs pass `load_config`.
- [ ] **Step 2: Run** `pytest tests/test_e2e.py -v -m slow` → FAIL until configs exist, then PASS once written (the code is already complete; failures here are integration bugs to fix in the owning module with a regression test there).
- [ ] **Step 3: Write `README.md`**: project goal and roadmap (link `docs/01-decisions.md`), install, the exact command sequence for `m1-overfit` and `m1-tinystories`, how to interrupt/resume, where reports land, test commands, CI badge, off-the-shelf disclosure (none in M1), data license.
- [ ] **Step 4: Write `notebooks/m1_kaggle.ipynb`** (plain nbformat-4 JSON, written by hand): cells to (1) clone the repo branch and `pip install -e .`, (2) print GPU info and `resolve_precision("auto", cuda)`, (3) GPU smoke test: `llm train configs/m1-overfit.yaml --set train.device=cuda --set train.precision=auto --set train.max_steps=50`, (4) prepare/tokenize TinyStories, (5) `llm train configs/m1-tinystories.yaml --set train.cost_per_hour=0 --stop-at-step 500` (deliberate interruption), (6) `llm resume runs/<id>`, (7) `llm report runs/<id>` and display `report.md`, (8) optional Hub upload of the run dir using a Kaggle secret `HF_TOKEN`.
- [ ] **Step 5: Run the whole suite:** `pytest -v` → all PASS; `pytest -m "not slow" --durations=5` → total under ~60 s.
- [ ] **Step 6: Commit** `feat: end-to-end test, M1 configs, README, Kaggle notebook`, then push the branch.

---

## Self-review notes

- Spec §1 success criteria → Task 13 (CI green, overfit, notebook interrupt/resume run, README). The GPU run itself is executed by the user on Kaggle from the notebook; its report is committed afterwards.
- Spec §2 layout → File Structure. One addition: `src/llm/manifest.py` and `src/llm/data/prepare.py` (the provenance rule and data orchestration needed a home); `report.py` is listed in the spec.
- Spec §3 data → Task 3; §4 tokenizer → Task 4; shards/forward-compat dtype → Task 5; §5 model → Task 6; §6 training/checkpoint/resume → Tasks 9–11; §7 tests → each task; §8 eval/report → Tasks 8 and 12; §9 cost logging → Task 10 (`cost_per_hour`).
- Interface names cross-checked: `TokenChunks`, `DataPosition`, `next_batch`, `epoch_order`, `TrainState`, `find_latest_valid`, `load_model_for_inference`, `grad_accum_steps`, `ConfigError`, `ProvenanceError` are each defined once and used with the same signatures.
