# M3 Text Brain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stage 1 of the Pluto architecture: a modern-layer text brain (RoPE, RMSNorm, SwiGLU, GQA) with reserved control tokens, trained on a depth-focused multi-source mix on Modal H100s, with small A/B tests per layer and a measured size decision.

**Architecture:** Extend the existing `GPT` with config switches whose defaults reproduce M1/M2 exactly; extend the tokenizer with named reserved special tokens; extend data preparation to many weighted (optionally streamed) sources; add `train.init_from` for curriculum stages and 4k→16k context extension; add a throughput benchmark, a multiple-choice text eval, and a Modal training app.

**Tech Stack:** Python 3.12, plain PyTorch (SDPA/FlashAttention), `datasets` (streaming), `huggingface_hub`, Modal.

**Spec:** `docs/superpowers/specs/2026-11-03-pluto-architecture-design.md` (§3 model, §5 reserved tokens, §6 Stage 1, §7 budget, §8 measurement).

## Global Constraints

- Defaults must keep every existing config, checkpoint and test working unchanged (M1/M2 reproducibility).
- From-scratch rule: no Hugging Face Transformers/Lightning/tokenizer libraries in the model, tokenizer or training loop; `datasets` only downloads text.
- Commit trailer only: `Co-Authored-By: thedeepakreddy <thedeepakreddy1@gmail.com>`; never Claude session links.
- Never debug on H100s: every code path runs on CPU tests and a free/cheap GPU before an H100 run.
- Budget: M3 uses free compute only (Modal free hours, Kaggle, Colab, cloud trial credits); `$0` out of pocket.
- Reserved control tokens exist in the M3 vocabulary from the start (spec §3): `<think> </think> <speak> <revise> <search> <calc> <code> <docs> <result> </result> <ask_thinker> <action> <remember> <recall> <now> <user> <pluto> <sure> <unsure> <calm> <excited> <whisper> <slow> <fast> <voice:emma> <voice:pluto_young>`, plus 64 spare tokens `<spare_0>..<spare_63>`.
- Default vocabulary 32,768 (including specials); context trained at 4,096, extended to 16,384.
- Cloud target size ~400M (Option B); pilot Option A ~150M. Final size is confirmed with the user after the H100 benchmark (Task 7).
- Every dataset's licence is recorded in its data report.

## Review Focus

1. Old M1/M2 checkpoints and configs load and give identical outputs with the new code (defaults = old architecture).
2. Text containing a literal string like `<think>` must NOT encode to the special token (only explicit API inserts specials).
3. A streamed source that ends early or errors mid-download must not silently produce a short dataset: the report records docs per source and preparation fails if a source yields 0 docs.
4. KV-cached generation with RoPE + GQA must match the uncached forward pass (same logits) at every position.
5. `train.init_from` with a different `context_length` (4k→16k) must load all weights (RoPE has no position table) and refuse incompatible shapes with a clear error.

---

### Task 1: Modern layer switches in `GPT`

**Files:**
- Modify: `src/llm/config.py` (`ModelConfig`), `src/llm/model/gpt.py`
- Test: `tests/test_model_modern.py`

**Interfaces:**
- Produces: `ModelConfig` new fields with defaults `pos: Literal["learned","rope"]="learned"`, `rope_theta: float=10000.0`, `norm: Literal["layernorm","rmsnorm"]="layernorm"`, `mlp: Literal["gelu","swiglu"]="gelu"`, `n_kv_head: int|None=None` (None = n_head), `swiglu_hidden: int|None=None` (None = round_up(8/3·d, 64)). `GPT.forward`/`forward_cached` signatures unchanged. `expected_num_params(cfg, vocab_size)` covers all variants.

- [ ] **Step 1: Write failing tests** in `tests/test_model_modern.py`:
  - `test_defaults_identical_to_m1`: GPT with default fields, seed 0, equals output of a GPT built from the same config before the change (compare with a stored small tensor checksum computed from current `main` in this step, saved as a constant in the test).
  - `test_variants_param_count[...]` parametrized over (rope, rmsnorm, swiglu, n_kv_head=2): `model.num_params() == expected_num_params(cfg, V)`.
  - `test_cached_matches_uncached[...]` same params: logits from `forward_cached` token by token equal `forward` logits (atol 1e-5, fp32, CPU).
  - `test_rope_has_no_position_table`: a `pos="rope"` model has no `pos_emb` parameter, and inputs longer than `context_length` still raise the existing `ValueError`.
  - `test_gqa_requires_divisor`: `n_head=6, n_kv_head=4` raises `ConfigError`.
- [ ] **Step 2: Run** `pytest tests/test_model_modern.py -q` → FAIL.
- [ ] **Step 3: Implement** `RMSNorm(d, eps=1e-6)`, `apply_rope(x, cos, sin)`, rotary cache per model, SwiGLU MLP (`w1,w3` gate/up, `proj` down), GQA attention (k/v projected to `n_kv_head`, repeated for SDPA), all selected by config; config validation in `config.py`.
- [ ] **Step 4: Run** `pytest tests/test_model_modern.py tests/test_model.py -q` → PASS (old model tests unchanged).
- [ ] **Step 5: Commit** `feat(model): RoPE, RMSNorm, SwiGLU, GQA switches (defaults = M1 architecture)`.

### Task 2: Reserved special tokens in the tokenizer

**Files:**
- Modify: `src/llm/tokenizer/bpe.py`, `src/llm/config.py` (`TokenizerConfig`)
- Test: `tests/test_tokenizer_specials.py`

**Interfaces:**
- Produces: `TokenizerConfig.special_tokens: tuple[str,...] = ()` and `reserved_spare: int = 0`; `BPETokenizer.train(texts, vocab_size, special_tokens=(), ...)`; `BPETokenizer.special_id(name: str) -> int`; `BPETokenizer.specials -> dict[str,int]`; layout `[256 bytes][merges][specials in order][spares][<|endoftext|>]` with `eot_id == vocab_size - 1` as today; `encode()` never emits a special; `decode()` renders specials as their names. `save/load` round-trips specials; loading an old tokenizer (no specials) still works.
- Constant `PLUTO_SPECIALS` in `bpe.py` = the Global Constraints list; config `configs/m3-*.yaml` set `special_tokens` to it and `reserved_spare: 64`.

- [ ] **Step 1: Tests:** `test_specials_ids_layout` (ids contiguous before eot, merges count = vocab − 256 − specials − spares − 1); `test_literal_text_not_special` (`encode("<think>")` contains no special id); `test_decode_special_name`; `test_old_tokenizer_loads` (load fixture saved without specials → `specials == {}`); `test_vocab_too_small_rejected`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** in `bpe.py`; `train_tokenizer(cfg)` passes `cfg.tokenizer.special_tokens` + spares; manifest records the special list.
- [ ] **Step 4: Run** `pytest tests/test_tokenizer_specials.py tests/test_tokenizer.py -q` → PASS.
- [ ] **Step 5: Commit** `feat(tokenizer): named reserved special tokens and spares`.

### Task 3: Fast parallel tokenization of shards

**Files:**
- Modify: `src/llm/shards.py`, `src/llm/config.py` (`ShardConfig.workers: int = 1`)
- Test: `tests/test_shards_parallel.py`

**Interfaces:**
- Produces: `write_shards(cfg)` unchanged signature; with `shards.workers > 1` documents are encoded in a `multiprocessing` pool in fixed-size ordered batches, output byte-identical to `workers=1`. Logs tokens/s.

- [ ] **Step 1: Test** `test_parallel_shards_identical`: tiny fixture, `workers=1` vs `workers=3` → identical `.bin` files and manifest checksums.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** with `Pool.imap` (ordered, chunks of 256 docs); each worker loads the tokenizer once (initializer).
- [ ] **Step 4: Run** test + `tests/test_shards.py` → PASS. Measure on 100 MB of FineWeb-Edu on CPU and record tokens/s in the commit message.
- [ ] **Step 5: Commit** `perf(shards): parallel ordered tokenization`.

### Task 4: Multi-source, weighted, streamed data preparation

**Files:**
- Modify: `src/llm/config.py` (`DataConfig`, `SourceConfig`), `src/llm/data/sources.py`, `src/llm/data/prepare.py`
- Test: `tests/test_data_multisource.py`

**Interfaces:**
- Produces: `DataConfig.sources: list[SourceConfig] | None = None` (when set, replaces `source`; both set → `ConfigError`); `SourceConfig` adds `name: str`, `config_name: str | None`, `streaming: bool = False`, `max_bytes: int | None`, `min_score_field: str | None`, `min_score: float | None`, `license` (required non-"unknown" when `sources` is used). `prepare_data(cfg)` takes each source in order up to its `max_docs`/`max_bytes`, applies the score filter, then the existing clean/dedup/split; the data report has a per-source table (docs received/kept/bytes, licence, url).

- [ ] **Step 1: Tests** (local_jsonl fixtures standing in for HF): `test_two_sources_counts_in_report`; `test_max_bytes_stops_source`; `test_score_filter` (docs below `min_score` dropped and counted); `test_empty_source_fails` (Review Focus 3); `test_source_and_sources_conflict`; `test_single_source_configs_unchanged` (M1 config prepares identical data).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**; HF sources use `datasets.load_dataset(path, config_name, split=..., streaming=streaming)`; dedup stays global across sources.
- [ ] **Step 4: Run** test + `tests/test_prepare.py` → PASS.
- [ ] **Step 5: Commit** `feat(data): multi-source weighted streamed preparation with per-source report`.

### Task 5: `train.init_from` for text runs (curriculum stages and context extension)

**Files:**
- Modify: `src/llm/config.py` (`TrainConfig.init_from: str | None = None`), `src/llm/train/loop.py`
- Test: `tests/test_init_from_text.py`

**Interfaces:**
- Produces: a fresh text run with `train.init_from=<checkpoint .pt>` loads model weights only (fresh optimizer, step 0); shapes must match except nothing position-dependent exists for `pos="rope"`; a learned-position model with a different `context_length` raises `ValueError` naming the parameter. Provenance (`run.json`) records the init checkpoint path and checksum. `resume` ignores `init_from`.

- [ ] **Step 1: Tests:** `test_init_from_loads_weights` (outputs equal at step 0); `test_init_from_rope_context_extension` (4k→8k tiny model loads); `test_init_from_learned_pos_mismatch_errors` (Review Focus 5); `test_resume_ignores_init_from`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS (+ `tests/test_train.py`).
- [ ] **Step 5: Commit** `feat(train): init_from for text runs`.

### Task 6: Multiple-choice text evaluation (HellaSwag, ARC-Easy, PIQA)

**Files:**
- Create: `src/llm/eval/multiple_choice.py`
- Modify: `src/llm/cli.py` (`llm eval-text <run> --task hellaswag|arc_easy|piqa [--n N]`)
- Test: `tests/test_multiple_choice.py`

**Interfaces:**
- Produces: `score_choices(model, tok, context: str, choices: list[str]) -> list[float]` (sum log-prob of each choice's tokens, length-normalised per token); `run_mc_eval(run_dir, task, n=None) -> Path` writing `<run>/text_eval_<task>.json` with accuracy, n, and a contamination check (fraction of eval contexts found verbatim in the training data's dedup hash set).

- [ ] **Step 1: Tests** with a tiny fake task list: `test_score_prefers_memorised_choice` (overfit tiny model on one sentence → that ending scores highest); `test_accuracy_file_written`; `test_contamination_flag` (an eval context present in training data is counted).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** loaders via `datasets` (validation splits). **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(eval): multiple-choice text eval with contamination check`.

### Task 7: Throughput benchmark and size decision

**Files:**
- Create: `src/llm/train/bench.py`; Modify: `src/llm/cli.py` (`llm bench <config> [--steps 20] [--tokens 10e9]`)
- Test: `tests/test_bench.py`

**Interfaces:**
- Produces: `bench(cfg, steps=20) -> dict` with `tokens_per_s`, `step_s`, `peak_mem_gb`, `params`, `mfu` (from 6·N·tokens/s ÷ device peak, peak table for H100/A100/L4/A10G/T4, None if unknown), and `hours_for(tokens)`. CLI prints a table and the estimated H100 hours for the configured token budget. Uses synthetic token batches (no data needed), runs warmup steps first, excludes them.

- [ ] **Step 1: Test** `test_bench_cpu_tiny` (returns positive tokens_per_s, params == expected_num_params).
- [ ] **Step 2–4:** run → FAIL, implement, run → PASS.
- [ ] **Step 5: Commit** `feat(bench): throughput benchmark with MFU and time estimate`.
- [ ] **Step 6 (user gate):** after Task 9's benchmark on the H100, present measured tokens/s and hours for Option A and B to the user and get the size decision before the full run.

### Task 8: M3 configs and small layer A/B tests

**Files:**
- Create: `configs/m3-ab-base.yaml` (~20M: 6L×384, 6 heads, ctx 1024, learned/LN/GELU, 32k tokenizer, ~300M tokens of the M3 mix), `configs/m3-pilot.yaml` (Option A: 12L×768, 12 heads, 4 kv heads, rope/rmsnorm/swiglu, ctx 4096), `configs/m3-cloud.yaml` (Option B: 24L×1024, 16 heads, 4 kv heads, ctx 4096), `configs/m3-extend16k.yaml` (same as chosen size, ctx 16384, `rope_theta` 500000, `init_from` the 4k run's best.pt, short budget)
- Create: `docs/experiments/m3-layer-ab.md` (results table, filled from runs)
- Test: `tests/test_m3_configs.py`

**Interfaces:**
- Consumes: Tasks 1–5 fields. Data sections (shared by all M3 configs) list the sources with licences: FineWeb-Edu `sample-10BT` (min int_score 3), Wikipedia (latest `20231101.en` or newer), OpenStax-style open textbooks if available as a dataset, StackExchange (CC BY-SA), FineMath, permissively licensed code, Gutenberg. A/B variants are `--set` overrides of `m3-ab-base` (one change each: `model.pos=rope`, `model.norm=rmsnorm`, `model.mlp=swiglu`, `model.n_kv_head=2`, then all four).

- [ ] **Step 1: Test** `test_m3_configs_load` (all load; param counts in expected ranges: pilot 120–180M, cloud 330–430M); `test_ab_variants_differ_in_one_field`.
- [ ] **Step 2–4:** run → FAIL, write configs, run → PASS.
- [ ] **Step 5:** run the 6 A/B trainings on a free/cheap GPU (Kaggle/Colab/Modal L4), same seed and tokens; fill `docs/experiments/m3-layer-ab.md` with val loss and bits/byte; keep only changes that help.
- [ ] **Step 6: Commit** `feat(configs): M3 configs and layer A/B results`.

### Task 9: Modal training app

**Files:**
- Create: `modal_app/m3_train.py`
- Test: `tests/test_modal_m3.py` (imports the app module with `modal` installed; checks step names and that functions are defined; skipped if `modal` missing)

**Interfaces:**
- Produces: `modal run modal_app/m3_train.py --steps prepare,tokenizer,shards` (CPU, large disk Volume `pluto-m3`), `--steps bench --config configs/m3-pilot.yaml` (GPU from `PLUTO_GPU`, default H100, ~10 min), `--steps train --config ...` and `--steps resume --run <id>` (H100, timeout 24 h, Hub sync to `<user>/pluto-runs`, `max_wall_time_s` just under the timeout so it stops cleanly), `--steps summary` (CPU: prints metrics and eval files). Secrets: `pluto-hf`. Code cloned from GitHub at run time; the Volume persists data/tokenizer/shards between calls.

- [ ] **Step 1: Test** module import + `ALL_STEPS` content.
- [ ] **Step 2–4:** run → FAIL, implement following `modal_app/m2_finish.py` patterns, run → PASS.
- [ ] **Step 5 (cheap rehearsal, not H100):** `PLUTO_GPU=L4 modal run modal_app/m3_train.py --steps bench --config configs/m3-ab-base.yaml` and a 200-step `train` of the A/B base; confirm Hub sync and resume after `modal app stop`.
- [ ] **Step 6: Commit** `feat(modal): M3 training app`.

### Task 10: The M3 run and report

**Files:**
- Modify: `docs/papers/pluto-m1-m2-technical-report.md` (new M3 section) or create `docs/papers/pluto-m3-report.md`; `README.md` milestone row; `docs/01-decisions.md` (size decision, token budget)

- [ ] **Step 1:** H100 benchmark for Option A and B (Task 7 Step 6 user gate).
- [ ] **Step 2:** full Stage 1 run of the chosen size at 4k as a curriculum (spec §6): first a foundations run (open textbooks + FineWeb-Edu int_score ≥ 4, ~20% of the token budget), then the full mix with `train.init_from` = the foundations run's best.pt; resume across sessions as needed; then `m3-extend16k`.
- [ ] **Step 3:** evaluate: bits/byte on held-out data, `eval-text` hellaswag/arc_easy/piqa (contamination reported), fixed prompt samples.
- [ ] **Step 4:** write results, compute and cost, limitations; update README and decisions.
- [ ] **Step 5: Commit** `results(m3): text brain`.
