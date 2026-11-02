# From-Scratch LLM Project Plan (handoff from Codex)

This is the original planning handoff written in Codex, preserved verbatim-in-spirit as the
project's starting brief. Decisions made after the handoff live in `docs/superpowers/specs/`.

## Status
Planning only. No model, code, infrastructure, or paid GPU resources have been created yet.

## Project intent
Reproducible, from-scratch decoder-only LM training system, scaling
10M → 50M → 100M → 250M → 500M → 1B → 5B parameters. Portfolio project for AI engineer/researcher
roles; the value is the engineering + research discipline (data, training, eval, scaling, cost),
not frontier capability claims.

## Core philosophy
- Reproducibility over impressive-looking results
- Measured improvements over blindly using the newest technique
- Data quality over quantity
- Evaluation before optimization
- Checkpoint recovery must work before long runs
- Every major change has a baseline comparison
- Claims match measured evidence
- Failures and limitations documented openly

## First milestone (tiny end-to-end pipeline)
1. Load a tiny verified dataset
2. Train or load a tokenizer
3. Tokenize and pack the data
4. Build a tiny Transformer
5. Overfit a small sample
6. Save and reload a checkpoint
7. Resume training
8. Generate text from the checkpoint
9. Produce a small evaluation report

Then scale to 10M, then 50M.

## Planned systems
Data (download/clean/dedup/PII/split/version/report), tokenizer (train/eval/shard),
model (config/build/param count/shape checks/checkpoint), training (AMP, grad accum,
validation, logging, resume), evaluation (loss/ppl/fixed prompts/repetition/compare),
experiment tracking (IDs, configs, versions, hardware, cost), inference (sampling, demo).

## Pre-training tests
Shape, causal-mask, tiny-overfit, save/load equivalence, determinism.

## Checkpoint contents
Model weights, optimizer, scheduler, step, dataset position, RNG states, full config,
dataset version, tokenizer version, code version (git SHA).

## Initial 50M sketch (to be verified experimentally)
8 layers, d_model 512, 8 heads, ctx 2048, vocab 32000, AdamW, bf16/fp16, warmup + cosine.

## Compute strategy
Colab/Kaggle for development and small runs; rented hourly GPUs for long runs once the
pipeline is stable. Never rely on notebook storage for checkpoints.

| Target | Approx GPU time | Rough budget |
|---|---:|---:|
| 50M | 10–100 h | free – few hundred $ |
| 500M | 200–1,000 h | hundreds – few thousand $ |
| 1B | 500–1,500+ h | ~$1,000–3,000+ |

## Modern techniques (add one at a time, each with hypothesis + baseline + ablation)
FlashAttention, RMSNorm, RoPE, SwiGLU, activation checkpointing, DDP/FSDP, better optimizers,
longer context, data mixtures, quantized inference.

## Open decisions at handoff
1. Initial domain  2. Framework  3. Tokenizer type  4. Dataset sources/licenses
5. Context length  6. Language(s)  7. Storage for data/checkpoints
8. Experiment tracking  9. Eval tasks + fixed prompts  10. Paid GPU budget ceiling
