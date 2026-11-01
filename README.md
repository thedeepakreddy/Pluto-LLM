# Pluto: an LLM from scratch → Mini-Live

[![tests](https://github.com/thedeepakreddy/Pluto-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/thedeepakreddy/Pluto-LLM/actions/workflows/ci.yml)

**Pluto** is the first small model in a planned family. This repo is a reproducible
language-model training system built from scratch in plain PyTorch: our own
byte-level BPE tokenizer, our own GPT, our own training loop with exact resume, and an
evaluation/report pipeline. It is the foundation for **Mini-Live**, a small native speech + text
model that listens and talks back in real time, and longer term a model that can see, talk and text.

- Decisions and roadmap: [`docs/01-decisions.md`](docs/01-decisions.md)
- Milestone 1 design: [`docs/superpowers/specs/2026-11-01-m1-text-pipeline-design.md`](docs/superpowers/specs/2026-11-01-m1-text-pipeline-design.md)

| Milestone | What | Status |
|---|---|---|
| M1 | Text pipeline end-to-end | ✅ done: [first run report](results/m1/m1-tinystories-20261101-205014-d0a9bc2/report.md) |
| M2 | First voice model (~25M) on self-made TinyStories-Speech | next |
| M3 | ~100M text "brain" on FineWeb-Edu | planned |
| M4 | ~100M speech + text model | planned |
| M5 | Real-time spoken dialogue (Mini-Live) | planned |
| M6 | Autonomous research agent | planned |

## Milestone 1 result (free Kaggle T4)

A ~6M-parameter model trained from random weights on 197M tokens of TinyStories in **18 minutes**
on a free T4 (~199k tokens/s), with a deliberate interruption at step 500 and an exact resume.

| metric | value |
|---|---:|
| validation loss | 1.738 |
| bits per byte | 0.640 |
| perplexity (tokenizer-specific) | 5.69 |
| 4-gram repetition in samples | 0.146 |

Data finding: **15.1% of TinyStories (320,241 of 2,119,719 stories) are exact duplicates**, removed
before splitting. Sample at step 3,000: *"Once upon a time, there was a little dog named Spot. Spot
loved to play with his ball and run around in the park. One day, Spot saw a big, scary dog…"*
Full report: [`results/m1/…/report.md`](results/m1/m1-tinystories-20261101-205014-d0a9bc2/report.md).

## What is built from scratch

Everything except PyTorch itself: data cleaning and splitting, the BPE tokenizer (training and
encoding), token sharding, the Transformer, sampling, the training loop (mixed precision,
gradient accumulation, schedule, checkpointing, exact resume), metrics and reports.
No Hugging Face Transformers, Lightning or tokenizer libraries. (`datasets` is used only to
download raw text.)

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Run Milestone 1

Every run is one YAML config; any value can be overridden with `--set section.key=value`.

```bash
CFG=configs/m1-tinystories.yaml
llm prepare-data $CFG        # download, clean (every drop counted), split → data/…/data_report.md
llm train-tokenizer $CFG     # byte-level BPE, vocab 4096
llm tokenize $CFG            # token shards + manifest
llm train $CFG               # prints the run directory
llm resume runs/<run-id>     # continue exactly from the latest valid checkpoint
llm generate runs/<run-id> --prompt "Once upon a time" --seed 1
llm report runs/<run-id>     # report.md + charts
llm compare runs/<a> runs/<b>
llm pull-run <hub_repo> <run-id>  # restore a run after a dead session
```

`configs/m1-overfit.yaml` is the sanity check: a ~0.1M model that must memorise 20 stories.
On a free Kaggle GPU, use [`notebooks/m1_kaggle.ipynb`](notebooks/m1_kaggle.ipynb), which runs
the whole pipeline with a deliberate interruption and resume.

### Interruptions and free-tier time limits

Training stops cleanly and saves a checkpoint on `--stop-at-step N`, on SIGTERM/SIGINT, or
when `train.max_wall_time_s` is reached (11.5 h by default, before Kaggle's 12 h limit).
`llm resume` continues **bit-for-bit identically** on CPU (tested; on GPU, some kernels are
non-deterministic, so resumed losses match closely rather than exactly) and refuses to resume if
the config, data or tokenizer changed since the checkpoint. Rebuilding identical data, tokenizer
and shards in a new session is accepted.

If the whole session dies (Kaggle/Colab), set `train.hub_repo` and an `HF_TOKEN`: the run
directory and checkpoints are synced to a private Hugging Face repo, and
`llm pull-run <hub_repo> <run_id>` restores it before `llm resume`.

## Reproducibility guarantees

- Every artifact (data, tokenizer, shards) has a `manifest.json` with file checksums and the
  checksums of what it was built from; any mismatch is a hard error.
- Data order is a permutation fixed by `(seed, epoch)`; the data position is part of every checkpoint.
- Checkpoints hold model, optimizer, loss scaler, RNG states, data position, config, provenance
  and git SHA, are written atomically, and load with `torch.load(weights_only=True)`.
- Evaluation uses a frozen, versioned prompt set (`tinystories-v1`) and fixed sampling settings.
- Bits-per-byte is reported so models with different tokenizers can be compared.

## Tests

```bash
pytest -m "not slow"   # fast unit tests
pytest -m slow         # overfit + full CLI pipeline
```

## Off-the-shelf components and data

- M1 uses no pretrained models.
- Data: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (CDLA-Sharing-1.0),
  synthetic stories generated by GPT-3.5/GPT-4; this is stated in every data report.
