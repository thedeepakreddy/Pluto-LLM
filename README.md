# Pluto: an LLM from scratch → Mini-Live

[![tests](https://github.com/thedeepakreddy/Pluto-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/thedeepakreddy/Pluto-LLM/actions/workflows/ci.yml)

**Pluto** is the first small model in a planned family. This repo is a reproducible
language-model training system built from scratch in plain PyTorch: our own
byte-level BPE tokenizer, our own GPT, our own training loop with exact resume, and an
evaluation/report pipeline. It is the foundation for **Mini-Live**, a small native speech + text
model that listens and talks back in real time, and longer term a model that can see, talk and text.

- Decisions and roadmap: [`docs/01-decisions.md`](docs/01-decisions.md)
- Plan after M2 (M3–M6): [`docs/02-roadmap-m3-m6.md`](docs/02-roadmap-m3-m6.md)
- **Current status and next steps:** [`docs/03-status-and-next-steps.md`](docs/03-status-and-next-steps.md)
- Technical report (M1–M2, in progress): [`docs/papers/pluto-m1-m2-technical-report.md`](docs/papers/pluto-m1-m2-technical-report.md)
- Milestone 1 design: [`docs/superpowers/specs/2026-11-01-m1-text-pipeline-design.md`](docs/superpowers/specs/2026-11-01-m1-text-pipeline-design.md)

| Milestone | What | Status |
|---|---|---|
| M1 | Text pipeline end-to-end | ✅ done: [first run report](results/m1/m1-tinystories-20261101-205014-d0a9bc2/report.md) |
| M2 | First voice model (~30M) on self-made TinyStories-Speech | ✅ done: **18.8% WER** (text-then-audio), [results](docs/papers/pluto-m1-m2-technical-report.md#38-evaluation-and-results-stage-4) |
| M3 | Text "brain" (~400M cloud size): depth-focused data, modern layers, 16k context, reserved control tokens | code ready; data preparation on Modal |
| M4 | Brain + voice: multi-voice (voice IDs), real voices, fixed word timing | planned |
| M5 | Mini-Live: full-duplex talk, thinks while listening and while speaking | planned |
| M6 | Tools (web search, scientific calculator, code, documents), memory, research agent | planned |
| Later | Laptop/phone sizes by distillation, ~500M, vision | planned |

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

## Milestone 2: first voice

M2 teaches Pluto to talk: a ~30M-parameter speech+text model (an 8-layer "brain", d 384, context
640, plus a 3-layer depth "voice box" that predicts the 8 Mimi codebooks of each 80 ms frame),
trained on **TinyStories-Speech**, a dataset we synthesize ourselves by reading TinyStories aloud.
**Result:** on 200 held-out passages the text-then-audio model reaches **18.8% WER** with correct
timing (codec ceiling 1.3%, untrained floor 689%); the aligned layout rarely advances to the next word (192% WER;
91% with decoding fixes), a training fix is planned for M4. Details: [technical report §3.8](docs/papers/pluto-m1-m2-technical-report.md#38-evaluation-and-results-stage-4).

### Off-the-shelf components (disclosed)

- **Kokoro-82M** (TTS, voice `bf_emma`): only used to *create* the training audio.
- **Kyutai Mimi** (neural audio codec, 24 kHz, 12.5 frames/s, 8 codebooks × 2048): turns audio into
  the tokens Pluto predicts and back into audio. Audio quality is bounded by Mimi.
- **Whisper small.en** (ASR): filters bad synthetic clips (QA) and scores Pluto's speech (WER/CER).

The language model itself is trained from scratch; it reuses M1's tokenizer, data and shards.

### Stages

Configs: `configs/m2-stage0.yaml` (tiny smoke budget), `m2-warmstart.yaml`,
`m2-speech-aligned.yaml`, `m2-speech-baseline.yaml`. On Kaggle use
[`notebooks/m2_kaggle.ipynb`](notebooks/m2_kaggle.ipynb), which also syncs runs and the speech
dataset to a private Hugging Face repo so every stage survives a dead session.

```bash
llm prepare-data configs/m2-warmstart.yaml   # + train-tokenizer, tokenize (identical to M1)
llm speech-check configs/m2-stage0.yaml --n 20                 # Stage 0: are the tools sane?
llm build-speech configs/m2-speech-aligned.yaml --max-hours-session 9    # Stage 1, re-run until complete
llm train configs/m2-warmstart.yaml                            # Stage 2: text-only warm start
cp runs/<warm-run>/checkpoints/best.pt checkpoints/m2-warmstart-best.pt   # = audio.init_from
llm train configs/m2-speech-aligned.yaml                       # Stage 3a
llm train configs/m2-speech-baseline.yaml                      # Stage 3b
llm eval-speech runs/<run> --floor                             # Stage 4: Whisper WER/CER, duration, RTF
llm speak runs/<run> --text "Once upon a time" --out out.wav
llm continue-speech runs/<run> --audio prompt.wav --out cont.wav
llm demo runs/<run>                                            # Gradio demo
```

### The comparison: aligned vs text-then-audio

`m2-speech-aligned` interleaves each word's text tokens with its audio frames (text one frame ahead);
`m2-speech-baseline` puts all the text first, then all the audio. The two configs differ **only** in
`audio.layout` and `run.name` (a test enforces this): same data, warm start, steps, learning rate and
seed. They are compared on the same 200 held-out test chunks by Whisper-transcribing the generated
speech (WER/CER against the text), plus duration ratio vs the reference audio and real-time factor;
`--floor` also scores an untrained model, and the codec+ASR ceiling (Mimi round-trip of real audio)
bounds the best achievable WER. One run per layout and one seed, so differences are indicative only.

### Creator identity (private)

The committed configs contain **no** identity section. To teach Pluto who made it, put your
creator-identity file in a **private** Hugging Face dataset and pass
`--set identity.hf_dataset=<your-username>/<dataset>` to `prepare-identity`, `build-speech` and
`train` (in the notebook, set `IDENTITY_DATASET` in its marked cell). It is never committed and only
uploaded to your private Hub repo. Every stage works with it left empty.

## Where Pluto is going

Full design: [`docs/superpowers/specs/2026-11-03-pluto-architecture-design.md`](docs/superpowers/specs/2026-11-03-pluto-architecture-design.md)
· roadmap and decisions: [`docs/02-roadmap-m3-m6.md`](docs/02-roadmap-m3-m6.md). Everything below is
**planned, not built**; each capability gets its own measured milestone before it is claimed.

**Goal:** a small model that **thinks well and speaks with ultra-low latency**: first spoken word in
~300–500 ms, reasoning only when needed, honest about what it knows.

**How it works (multi-stream with "thought bursts"):** every 80 ms Pluto hears your audio, writes its
own audio and the word it is saying, and between steps can write a burst of **silent thought tokens**.
So it can **listen, think and talk at the same time** (full-duplex):

- keeps talking relevantly while it reasons, so you never wait in silence;
- if you ask "are you thinking?" it says so honestly and carries on;
- **thinks while you are still talking**, so hard answers come faster;
- **answers early and corrects itself gracefully** if a check finds a mistake;
- thinks only as much as each question needs;
- thoughts can be shown on screen ("show thoughts"), and you can steer them by voice.

**Planned model:** RoPE, RMSNorm, SwiGLU, grouped-query attention, FlashAttention, KV cache with a
rolling window, 16k context; one vocabulary with reserved control tokens (thinking, tools, voices,
style, time) from M3 on. English only, for the best quality per parameter.

**Planned sizes (one family, same code):** cloud ~400M (the deep thinker), laptop ~150M (offline on a
MacBook), phone ~30–60M; smaller sizes are distilled from the cloud model and quantised.

**Planned tools and abilities:**

| Ability | What it does |
|---|---|
| Advanced scientific calculator | exact symbolic algebra, calculus, equations, units, constants, matrices, statistics (sandboxed) |
| Internet search | up-to-date answers from the web while thinking, with sources |
| Code runner | runs small programs in a sandbox |
| Your documents | answers from your own notes and files |
| Deep-thinker handoff | small Pluto asks a bigger model for hard questions |
| Memory | remembers you between chats (opt-in, deletable) |
| Multiple voices | `--voice emma`, `--voice pluto_young`, emotion and speed control |
| Self-checking and confidence | verifies answers, says how sure it is (measured) |
| Parallel thoughts | tries several lines of reasoning, keeps the best (cloud) |
| Skill library and "dreaming" | practises on self-made problems checked by tools; reuses verified methods |
| Receipts | every claim can show where it came from |
| Vision | an image/camera stream (after M6) |

**Training plan:** depth-focused text (FineWeb-Edu, open textbooks, latest Wikipedia, expert Q&A,
maths, code, public-domain books, ~10–20B tokens, foundations first) → reasoning (`<think>` format,
tool-verified problems) → tool use → voice (multi-voice) → full-duplex dialogue → distillation.
Licences are checked and disclosed for every source.

**Safety rules:** tools run sandboxed; actions need your confirmation; memory and learning from
conversations are opt-in and deletable; web and document content is treated as information, never
as instructions.

**Budget:** everything up to the ~400M model uses free compute only (Kaggle, Colab, Modal free hours,
cloud trial credits); the $100 cap is reserved for growing Pluto to ~500M.

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

- M1 uses no pretrained models. M2 uses Kokoro-82M, Mimi and Whisper as described above (none of
  them is part of Pluto's trained weights).
- Data: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (CDLA-Sharing-1.0),
  synthetic stories generated by GPT-3.5/GPT-4; this is stated in every data report.
