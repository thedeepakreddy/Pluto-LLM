# Pluto: A From-Scratch Speech + Text Language Model on Free-Tier GPUs

**Technical report, Milestones 1–2 (work in progress)**
thedeepakreddy · November 2026 · Repository: `thedeepakreddy/Pluto-LLM`

> **Status.** Milestones 1 and 2 are trained and evaluated. M2's headline: the text-then-audio model
> (~30M parameters) speaks held-out passages at **18.8% WER** with correct timing (duration ratio
> 1.00), against a 1.3% codec ceiling and a 689% untrained floor. The aligned layout underperforms
> because its decoder stalls on the next-word choice (§3.8); a decoding fix is being tested.

---

## Abstract

We describe **Pluto**, a family of language models built from scratch in plain PyTorch, as the first
step toward a small multimodal model that can read, listen and speak. Milestone 1 (M1) is a fully
reproducible text pipeline: our own data cleaning, byte-level BPE tokenizer, GPT, and training loop
with exact checkpoint resume. A 6M-parameter model trained on TinyStories reaches a validation loss
of 1.738 (0.640 bits per byte) in 18 minutes on a free Kaggle T4. Milestone 2 (M2) extends the same
codebase to speech: we synthesize **TinyStories-Speech**, a 200-hour, 43,461-chunk corpus of the
stories read aloud and encoded as neural-codec tokens, and train a ~30M-parameter speech+text model
(an 8-layer "brain" plus a 3-layer depth "voice box") under two sequence layouts, *aligned*
(text interleaved with its audio frames) and *text-then-audio*, in a controlled comparison scored
by Whisper word error rate. The whole project runs on free-tier GPUs under a self-imposed $100 cap;
spend to date is $0.

---

## 1. Motivation and constraints

The long-term goal is a model that can **see, talk and text**. Instead of fine-tuning an existing
model, Pluto is trained from random weights so every component is understood, tested and
reproducible. Project principles (from `CLAUDE.md`):

- **Reproducibility over impressive results.** Claims must match measured evidence.
- **One change at a time:** hypothesis, baseline, controlled experiment, conclusion.
- **Checkpoint/resume before long runs;** never debug on paid GPUs.
- **Budget:** free tiers first; hard cap of $100 on paid GPUs, cost logged per run.
- **Honest disclosure** of off-the-shelf components and open documentation of failures.
- **Modality-agnostic core:** new modalities enter as new tokenizers/encoders and vocabulary
  extensions, not rewrites.

Roadmap: M1 text pipeline → **M2 first voice (~30M)** → M3 ~100M text brain (FineWeb-Edu) →
M4 ~100M speech+text → M5 real-time spoken dialogue ("Mini-Live") → M6 research agent → vision.

---

## 2. Milestone 1: the text pipeline (complete)

### 2.1 System

Everything except PyTorch is written in the repository: data download and cleaning (every dropped
document is counted by reason), deterministic splitting, byte-level BPE training and encoding,
token sharding with checksummed manifests, the Transformer, sampling with a KV cache, the training
loop (mixed precision, gradient accumulation, cosine schedule, wall-time stops, atomic checkpoints),
metrics, charts and reports.

Reproducibility guarantees:
- Every artifact (data, tokenizer, shards) carries a `manifest.json` with file checksums and the
  checksums of its inputs; any mismatch is a hard error.
- Data order is a permutation fixed by `(seed, epoch)`; the data position is saved in every checkpoint.
- Checkpoints hold model, optimizer, loss scaler, RNG states, data position, config, provenance and
  git SHA, are written atomically and loaded with `weights_only=True`.
- `llm resume` is **bit-for-bit identical on CPU** (tested) and refuses to resume if the config,
  data or tokenizer changed. On GPU some kernels are non-deterministic, so resumed losses match
  closely rather than exactly.
- Runs sync to a private Hugging Face repo, so a killed free-tier session loses at most one
  checkpoint interval.

### 2.2 Data finding

Of 2,119,719 TinyStories documents, **320,241 (15.1%) are exact duplicates**, removed before
splitting (230 more dropped as too short). Kept: 1,799,248 documents (train 1,763,298 /
val 17,947 / test 18,003).

### 2.3 Setup and results

| item | value |
|---|---|
| Model | 6 layers, d_model 256, 8 heads, context 256, vocab 4,096 (~6M parameters) |
| Tokenizer | byte-level BPE, 4,096 tokens, 3.93 bytes/token, trained in 13.97 s |
| Training | 3,000 steps × 65,536 tokens = 196.6M tokens, lr 1e-3, warmup 200, seed 1337 |
| Hardware | 1× Tesla T4 (Kaggle free tier), ~199k tokens/s |
| Wall time / cost | 1,061 s (≈18 min) / $0 |

| metric | value |
|---|---:|
| validation loss | **1.738** |
| bits per byte | **0.640** |
| perplexity (tokenizer-specific) | 5.69 |
| 4-gram repetition in samples | 0.146 |

The run included a deliberate interruption at step 500 and an exact resume. A second independent
run reproduced the same validation loss and bits per byte. Sample at step 3,000, prompt
*"Once upon a time, there was a little dog named"*: *"Spot. Spot loved to play with his ball and run
around in the park. One day, Spot saw a big, scary dog…"* At step 250 the same prompt produced
repetitive loops ("I'm sorry, Lily. I'm sorry…"), which disappear by step 1,750.

Full report: `results/m1/m1-tinystories-20261101-205014-d0a9bc2/report.md`.

---

## 3. Milestone 2: first voice (in progress)

### 3.1 Research question

Does interleaving each word's text tokens with its audio frames (**aligned**, "inner monologue"
style) make a small model's generated speech more intelligible than putting all text first and
then all audio (**text-then-audio baseline**), at equal size, data, steps, learning rate and seed?

### 3.2 Off-the-shelf components (disclosed)

| component | role | part of Pluto's weights? |
|---|---|---|
| Kokoro-82M (TTS, voice `bf_emma`) | only to *create* training audio | no |
| Kyutai Mimi (24 kHz, 12.5 frames/s, 8 codebooks × 2,048) | audio ↔ tokens | no (frozen codec) |
| Whisper small.en | QA filter on synthetic clips; WER/CER scoring | no |

The language model is trained from scratch and reuses M1's data, tokenizer and shards. Audio
quality is upper-bounded by Mimi; intelligibility scores depend on Whisper.

### 3.3 TinyStories-Speech: a self-built dataset

Pipeline: TinyStories splits from M1 → whole-sentence chunks of at most 30 s (≤ 375 frames) →
Kokoro synthesis at 24 kHz with word timings → Whisper QA on a 2% sample (chunks above 30% WER are
rejected) → Mimi codes → shards with a manifest. Generation is resumable: a progress file records
finished chunk ids, so a session that hits its time limit continues where it stopped.

**Measured result:**

| split | chunks | audio |
|---|---:|---:|
| test | 200 | 0.918 h |
| val | 100 | 0.462 h |
| train | 43,461 | 200.000 h |

Build throughput on Kaggle (2× T4, of which the TTS/codec pipeline used one): roughly 18–20 hours of
audio per hour of wall time (85.5 h of audio after 4 h 41 min of session time). The build took one
long session plus a short continuation and was synced to the private Hub repo after each session.

### 3.4 Model

- **Brain:** a GPT with 8 layers, d_model 384, 6 heads, context 640, initialised from a text-only
  warm start (§3.6). Each 80 ms frame is one position in the sequence.
- **Voice box (depth Transformer):** 3 layers, d_model 192, 8 heads. Given the brain's hidden
  state for a frame, it predicts the frame's 8 Mimi codebooks autoregressively.
- **Size:** ~29.6M parameters (reported exactly by the model).
- **Acoustic delay:** codebooks 2–8 of frame *k* are paired with codebook 1 of frame *k+1*
  (one-frame delay, as in Moshi), giving the voice box more context.
- **Loss:** codebook 1 (the most semantic) weighted 3×, codebooks 2–8 and text 1×.
- **Stopping:** a learnable stop-audio class (codebook-1 `EMPTY` target at flush positions) lets the
  model end its own speech.

### 3.5 The two layouts

- **Aligned:** each frame carries its audio codes plus the text token being spoken at that moment,
  with text one frame ahead of its audio. Word timings come from Kokoro.
- **Text-then-audio (baseline):** the whole text, then the whole audio.

The two configs differ **only** in `audio.layout` and `run.name`; a unit test enforces this.

### 3.6 Training schedule

| stage | what | budget | status |
|---|---|---|---|
| 0 | `speech-check`: tools sanity check on 20 val chunks | minutes | run |
| 1 | build TinyStories-Speech | 200 h of audio | **done** (43,461 chunks) |
| 2 | text-only warm start of the brain on TinyStories | 2,400 steps × 81,920 tokens (196.6M) | **done** (stopped at max_steps 2,400) |
| 3a | aligned speech+text | 6,000 steps × 40,960 tokens (245.8M), lr 6e-4, warmup 300 | running |
| 3b | text-then-audio baseline | identical | running |
| 4 | Whisper WER/CER, duration ratio, RTF; demo; report | 200 test chunks per model | **done** (§3.8) |

Stages 3a and 3b run **in parallel, one per T4** (`CUDA_VISIBLE_DEVICES=0/1`). Both use the same
GPU type, data, seed and settings, so the comparison stays controlled while the wall time halves.

### 3.7 Training-time observations (measured, Stage 3)

| quantity | aligned | baseline |
|---|---:|---:|
| throughput | ~77k tokens/s | ~72–74k tokens/s |
| GPU utilisation | 97–98% | 98% |
| peak memory (allocator) | ~2.2 GB | ~2.4 GB |
| total loss, step ~230 | 4.88 | 4.34 |
| total loss, step ~1,500 | 2.56–3.05 | 2.81 |
| codebook-1 loss, step ~230 → ~1,500 | 3.09 → 0.69 | 3.00 → 1.15 |
| text loss, step ~1,500 | 0.71–1.18 | 0.87 |

Interpretation, kept deliberately cautious:
- The text loss is low from the start (≈0.8–1.3 nats), so the warm start transferred.
- Codebook-1 loss falls far below the uniform-guess level of ln 2048 ≈ 7.6 nats, so the model is
  learning audio structure. Lower loss does **not** prove intelligible speech; that is what
  Stage 4 measures.
- Losses of the two layouts are **not directly comparable** (different token mixes per sequence);
  the comparison is made only on Whisper WER of generated speech.

### 3.8 Evaluation and results (Stage 4)

**Protocol.** The first 200 held-out test chunks (multi-sentence passages, 73% over 60 words). WER and
CER compare Whisper's transcript of the generated speech with the source text; duration ratio is
generated vs reference audio length; RTF is generation time / audio duration. Reference points: the
**ceiling** (real Kokoro audio → Mimi → Whisper) and the **floor** (an untrained model of the same
architecture, scored on the first 30 chunks because it never stops and is slow to transcribe). Default
sampling: voice box temperature 0.8, top-k 50; text head greedy. One run per layout, one seed.

| system | chunks | WER | CER | duration ratio | RTF (GPU) |
|---|---:|---:|---:|---:|---:|
| ceiling (real audio through the codec) | 200 | **1.3%** | 1.0% | – | – |
| **text-then-audio (baseline)** | 200 | **18.8%** | 14.2% | **1.00** | 0.39 |
| aligned | 200 | 192% | 141% | 2.33 | 0.34 |
| untrained floor (aligned architecture) | 30 | 689% | 621% | 2.38 | 0.34 |

WER above 100% means the transcript has more wrong words than the reference has words: the aligned
model keeps talking after the text and its babble is counted as insertions.

**Finding 1: the text-then-audio model speaks whole passages.** At ~30M parameters it reproduces
81% of the words of unseen multi-sentence passages and ends its speech at the right time (only 1% of
outputs reach the frame limit). Error is flat across length (16% at 1–20 words, 19% at 61+).

**Finding 2: the aligned model fails at decoding, not only at modelling.** Per-item generation
statistics show that with default decoding it places only a median **23%** of the text tokens and
reaches the frame limit in **98%** of outputs. In the aligned layout the text head chooses, at every
frame, between the next token and "wait" (PAD), and speech may only stop after every token is placed;
so a stalled text head means drifting audio that never stops. Its read-along is still real: before
drifting it reads the opening sentence(s) correctly, sometimes a whole passage (best: 4.7% WER on a
64-word passage).

Decoding experiments (aligned, first 50 chunks):

| decoding | WER | prefix WER | text placed (median) | outputs at frame limit |
|---|---:|---:|---:|---:|
| default (voice T=0.8, text greedy) | 203% | 72% | 23% | 98% |
| voice T=0.5 | 201% | 66% | 31% | 100% |
| voice T=0 (greedy) | 252% | 67% | 28% | 96% |
| voice T=0.8 + pacing guard (place next token after 12 waiting frames) | **106%** | 59% | 88% | 64% |
| voice T=0.5 + pacing guard | 109% | 54% | 97% | 54% |

Lower voice-box temperature does not help; forcing the text forward halves WER, confirming the stall
as the main failure, but the aligned model remains far behind the baseline. Working hypothesis under
test: because most training frames are PAD, the next-token probability at the right frame is often
below 50%, so a **greedy** binary choice always picks "wait"; sampling the choice (text temperature
~1) should place tokens at roughly the trained rate. Results will be added here.

**Conclusion so far (indicative: one run per layout, one seed).** For this ~30M model and this data,
text-then-audio clearly beats the aligned "inner monologue" layout under the decoding we used. That
is a statement about the full system (training + decoding), not a general verdict on aligned layouts,
which matter for low-latency streaming (M5) and will be revisited at the M4 scale.

**Creator identity.** Text recall of the held-out identity probes: 100% (both layouts). Spoken recall
(aligned): 0%, consistent with the aligned decoding failure.

### 3.9 Creator identity (private)

Pluto is taught basic facts about its own name and creator from a private creator-identity dataset
(stored as a private Hugging Face dataset, never committed). Statements are mixed into text batches
every 50 steps and spoken into an identity speech split used every 100 steps, with some phrasings
held out to measure recall rather than memorisation. Public exports pass through
`export_public_results`, which redacts identity content and fails closed on any leak. The model is a
continuation model, not an assistant: it can complete "Pluto was created by…", but question
answering and dialogue are M5 goals.

---

## 4. Engineering log: what went wrong and how it was fixed

Free-tier training is mostly an exercise in surviving the environment. Issues hit in M2, in order:

| problem | cause | fix |
|---|---|---|
| `kokoro` would not install | Kaggle runs Python 3.13; Kokoro ≥ 0.9 needs ≤ 3.12, and Kokoro 0.7 pins an old NumPy | the pipeline runs in an isolated **uv Python 3.12 venv**; the notebook calls its `llm` CLI |
| re-running setup nested the repo and prompted interactively | relative clone path, `uv venv` overwrite prompt | absolute work dir; reuse the venv if present |
| Hub sync silently off | `HF_TOKEN` secret not attached to the notebook | notebook prints explicit warnings; secrets documented |
| `prepare-identity` FileNotFoundError | the dataset's `identity.json` sat in a subfolder | loader detects a single nested dataset folder (tested) |
| speech training refused to start | dataset never built (`manifest.json` missing) | by design: training fails closed on an incomplete dataset |
| Kokoro crashed: spaCy `en_core_web_sm` missing | Kokoro's auto-download installed the model into the *system* Python 3.13, not the 3.12 venv | install the model wheel directly into the venv; setup now does this automatically |
| one of two GPUs idle | the TTS build and single training runs use one GPU | the two Stage 3 runs are launched as two processes, one per GPU |
| session time budget | Kaggle 12 h session limit, ~30 GPU-h/week | Hub sync and exact resume; runs stop cleanly on SIGTERM; continuation notebook for Colab |

---

## 5. Compute and cost

| stage | hardware | wall time | cost |
|---|---|---|---|
| M1 run | 1× T4 (Kaggle) | 18 min | $0 |
| M2 speech build | 2× T4 (Kaggle), one used | ~11 h | $0 |
| M2 warm start | 1× T4 (Kaggle) | ~1–1.5 h | $0 |
| M2 Stage 3a + 3b | 2× T4 (Kaggle), parallel | ~1 h (est. from measured throughput) | $0 |
| **Total paid spend** | | | **$0 of the $100 cap** |

Plan: finish M2 evaluation on Kaggle or Google Colab (free); keep Modal's free H100 hours for M3+.

---

## 6. Limitations

- ~30M parameters; synthetic speech from **one** Kokoro voice reading children's stories only.
- Audio quality is bounded by the Mimi codec; WER is measured by Whisper, itself imperfect.
- One run per layout and one seed: differences are indicative, not statistically established.
- GPU training is not bit-exact across resumes or hardware; a session split across machines is
  noted in each run's limitations.
- No dialogue or question answering yet; identity knowledge is completion-only.

---

## 7. Next steps

1. Test sampled text-head decoding for the aligned model (§3.8 hypothesis) and record the result.
2. Fix the aligned decoder/training for M4–M5, where streaming needs the aligned layout.
3. M3: a ~100M text brain on FineWeb-Edu (Modal H100s), then M4 brain + voice at ~100M.

---

## Appendix A: Reproduce

```bash
pip install -e ".[dev,speech,hub]"
llm prepare-data configs/m2-warmstart.yaml && llm train-tokenizer configs/m2-warmstart.yaml && llm tokenize configs/m2-warmstart.yaml
llm speech-check configs/m2-stage0.yaml --n 20
llm build-speech configs/m2-speech-aligned.yaml --max-hours-session 9   # re-run until complete
llm train configs/m2-warmstart.yaml
cp runs/<warm-run>/checkpoints/best.pt checkpoints/m2-warmstart-best.pt
llm train configs/m2-speech-aligned.yaml
llm train configs/m2-speech-baseline.yaml
llm eval-speech runs/<run> --floor
```

Notebooks: `notebooks/m2_kaggle.ipynb` (full pipeline), `notebooks/m2_colab.ipynb` (resume Stage 3
from the Hub and run Stage 4).

## References

- R. Eldan and Y. Li. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* 2023.
- A. Défossez et al. *Moshi: a speech-text foundation model for real-time dialogue* (Mimi codec, inner monologue, acoustic delay). Kyutai, 2024.
- A. Radford et al. *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). 2022.
- hexgrad. *Kokoro-82M* text-to-speech model (Hugging Face: hexgrad/Kokoro-82M).
