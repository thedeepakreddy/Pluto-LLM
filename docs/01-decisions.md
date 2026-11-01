# Decision log

Decisions agreed during design brainstorming (2026-11-01). The full Milestone 1 design spec
will live in `docs/superpowers/specs/`.

## Goal
A portfolio project for AI engineer/researcher roles: an LLM built from scratch that grows into
**"Mini-Live"**, a small native speech+text model that listens and talks back in real time
(in the style of Gemini Live / ChatGPT voice, at ~100M scale), with text as an option.
Value shown: from-scratch engineering, research discipline, honest measured claims, live demo.

## Roadmap
| Milestone | What | Compute |
|---|---|---|
| M1 | Tiny text pipeline end-to-end: data → tokenizer → model → overfit → checkpoint → resume → generate → report | Free / CPU |
| M2 | First voice model (~20–30M) on self-made "TinyStories-Speech" (Kokoro TTS → Mimi codec tokens), Gradio mic demo | Free (Kaggle/Colab) |
| M3 | ~100M text "brain" on ~2B FineWeb-Edu tokens; small-scale efficiency ablations | Free + small paid |
| M4 | ~100M speech+text model initialised from the text brain | Paid |
| M5 | Spoken dialogue tuning + real-time streaming app + interruptions (full-duplex = stretch) | Paid |
| M6 | Autonomous research agent that proposes/runs/analyses experiments on this system | Free/API |

## Settled choices
- **Framework:** plain PyTorch 2.x; model, training loop, data loader, sampling written by us.
- **Code structure:** modular Python package + single CLI, every run defined by a config file.
- **Tokenizer:** our own byte-level BPE (trainer + encoder), vocab size in config.
- **Data:** TinyStories (CDLA-Sharing-1.0) for M1/M2; FineWeb-Edu `sample-10BT` (ODC-By 1.0) for M3;
  LibriSpeech/LibriLight + self-generated TTS speech for M4+.
- **Eval:** bits-per-byte on held-out data, HellaSwag (contamination-checked), fixed prompt set,
  repetition score; GPT-2-tokenizer reference run as yardstick.
- **Context length:** 256–512 (M1), 1024 (M3).
- **Off-the-shelf parts (disclosed):** Mimi audio codec; Kokoro TTS for data generation;
  ASR only if needed for evaluation.
- **Storage:** Hugging Face Hub (+ Google Drive mount on Colab). Never notebook-only checkpoints.
- **Tracking:** local JSON/CSV per run is the source of truth; W&B public dashboards as a mirror.
- **Budget:** free tier first; **$100 hard cap** on paid GPUs (target $40–80), every dollar logged.
  Rules: no debugging on paid GPUs, ablations at 10–25M on free tier, auto-shutdown, resumable runs.
- **Free-tier GPUs (T4/P100) lack bf16:** fp16 + loss scaling must be supported and tested.

## Long-term vision (added 2026-11-01)
This project is the foundation for our own frontier-class multimodal LLM that can **see, talk,
and text**. Vision is a future milestone after Mini-Live (M5). Architectural consequence today:
modality-agnostic token streams and extensible vocab/embeddings; nothing vision-specific is built
until its milestone.

## M2 decisions (in progress, 2026-11-01)
- **Task format: A.** Each training example is a story's text tokens followed by the same story's
  audio tokens, in one sequence for one model. Teaches text→speech and spoken-story continuation;
  the same text+audio design leads to Mini-Live.
- **Repo/model name:** the project is now **Pluto** (repo `thedeepakreddy/Pluto-LLM`).
- **Creator identity dataset (PCID v1.1):** used later for instruction tuning + evaluation
  (and retrieval), not pre-training. Contains personal details: kept out of the public repo
  until the creator decides which fields are public.
- **Audio token layout: two levels (Moshi-style).** Mimi codec (12.5 frames/s × 8 codebooks).
  The main Transformer ("brain") steps once per frame; a small depth Transformer ("voice box")
  predicts the 8 codebook tokens of each frame. 8× shorter sequences than flattening; real-time
  friendly; the same architecture Mini-Live builds on. First data: stories ≤ 30 s (~375 frames),
  longer ones chunked.
