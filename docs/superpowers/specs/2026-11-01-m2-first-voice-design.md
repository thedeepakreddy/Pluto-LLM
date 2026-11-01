# Milestone 2 Design: Pluto's First Voice

- **Date:** 2026-11-01
- **Status:** Draft, awaiting review
- **Scope:** M2 only. Builds on M1 (`2026-11-01-m1-text-pipeline-design.md`, completed: first run in
  `results/m1/`).
- **Decisions:** `docs/01-decisions.md` (M2 section).

## 1. Goal and success criteria

Pluto learns to **speak**: given text, it produces speech in one consistent British female voice;
given the start of a spoken story, it continues it in speech. One model learns text and audio
together, using the two-level (brain + voice box) architecture that M4–M5 (Mini-Live) build on.

M2 is done when:
1. All tests pass in CI (CPU; audio models mocked where needed).
2. The pilot (~200 h of speech) trains on free GPUs with exact resume and Hub sync.
3. The main design reaches **WER ≤ 35%** on 200 held-out sentences (pilot), clearly better than
   the untrained floor, with the ceiling (Kokoro → Mimi round trip) reported alongside.
4. The main-vs-baseline comparison is reported honestly (difference vs seed variation).
5. `llm speak`, `llm continue-speech` and a Gradio demo work from a trained checkpoint.

Stretch (Stage 4): ~800 h of speech, WER ≤ 15%.

### Non-goals
Conversation or answering questions, real-time streaming, interruptions, multiple voices,
real human recordings, models above ~30M parameters. These belong to M3–M5.

### Off-the-shelf components (disclosed in README and every report)
- **Kokoro-82M** (Apache-2.0): generates the training speech. Not part of Pluto.
- **Mimi** (Kyutai, CC-BY-4.0): audio codec, 24 kHz, 12.5 frames/s, 8 of its codebooks used,
  2,048 codes each. Frozen; encodes training audio and decodes Pluto's output.
- **Whisper** (one pinned size and version): scoring (WER) and word timing for prompts and, if
  needed, training data.

Everything else is ours: the brain, the voice box, the data pipeline, training and evaluation.

## 2. Speech dataset ("TinyStories-Speech")

```
TinyStories (M1 data splits) → sentence chunks ≤ ~30 s → Kokoro (bf_emma, 24 kHz)
  → word timings → QA sample (Whisper) → Mimi codes → shards + manifest
```

- **Source text:** the M1 cleaned splits (`data/tinystories-v1`), so train/val/test separation
  carries over. Chunks are whole sentences, ≤ ~30 s of speech (≤ 375 frames).
- **Voice:** `bf_emma` (config value; alternatives `bf_isabella`, `bf_alice`, `bf_lily`).
- **Word timings:** each word's start time, from Kokoro's own timestamps; fallback: Whisper word
  timestamps. Stage 0 verifies which works before anything else is built on it.
- **Quality check:** Whisper transcribes a random 2% of clips; clips above a WER threshold are
  dropped and counted by reason in the data report, as in M1.
- **Stored per example:** text token ids (M1 tokenizer), Mimi codes `[frames, 8]` (uint16), and
  the frame index at which each text token starts. Raw audio is not stored, except a small
  random sample of `.wav` files kept for listening.
- **Sizes:** pilot ~200 h (~25k chunks); scale-up ~800 h. ~0.6 GB of codes at 800 h.
- **Resumable generation:** a progress file records finished chunk ids; a killed session resumes.
  Versioned manifests and provenance checks exactly as in M1 (deterministic checksums).

## 3. Model

### Sequence layouts

**Main design: time-aligned "inner monologue".** One position per 80 ms frame. Each position
carries the frame's 8 audio codes plus a text slot: the text token that starts at this frame, or
`<PAD_TEXT>` if no new token starts here. Text that starts in the same frame spills into the next
free slots, and a word's audio is delayed if needed so text never lags its audio. Stream:

```
[BOS] <AUDIO> (t1,f1) (t2,f2) … (tN,fN) <END_AUDIO>      t = text slot, f = 8 codes
```

**Baseline: text then audio.** `[BOS] text tokens <AUDIO> f1 … fN <END_AUDIO>`, the original
plan, kept for the controlled comparison.

**Acoustic delay (both layouts):** codebooks 2–8 of frame k are paired with codebook 1 of frame
k+1 (Moshi-style one-frame delay), giving the voice box more context.

### Brain (main Transformer)
- Reuses the M1 block (`llm.model.gpt.Block`): 8 layers, d_model 384, 6 heads, learned positions,
  context 640 positions.
- Input embedding per position = text-slot embedding + sum of 8 codebook embeddings
  (8 × 2,048 × 384), or only the text embedding for text positions in the baseline.
- Text vocabulary = the M1 tokenizer (4,096) + `<AUDIO>`, `<END_AUDIO>`, `<PAD_TEXT>`, `<BOS>`.
- Outputs: a text head (next text slot), and a hidden state handed to the voice box.

### Voice box (depth Transformer)
- 3 layers, d_model 256, 8 heads, 8 positions; input = projected brain state, then the
  embeddings of codes already chosen in this frame; 8 output heads of 2,048 classes.
- Predicts codebook 1, then 2, …, 8 of the frame autoregressively.

**Total:** ~25–30M trainable parameters (reported exactly by the model, as in M1).

### Loss
`L = w_text · CE(text) + Σ_k w_k · CE(code_k) + CE(<END_AUDIO> decision)`, with `w_1` > `w_2..8`
(defaults in config; e.g. `w_1 = 3`, others 1). Each component is logged separately.

### Fast generation (KV cache)
Brain and voice box both use a key/value cache during generation, so each new frame costs a
constant amount of compute instead of re-running the whole sequence. Generation speed
(frames/s and real-time factor) is measured and reported on CPU and GPU. Required now because
Mini-Live (M5) must run in real time.

## 4. Training

| Stage | What | Est. free GPU time |
|---|---|---|
| 0 | 100-clip check: timings, Mimi round trip, measured throughput | ~0.5 h |
| 1 | Pilot speech data, ~200 h | ~4–6 h (resumable) |
| 2 | Text warm-start: brain trained on TinyStories text only | ~0.5 h |
| 3a | Main design (aligned), initialised from Stage 2 | ~3–5 h |
| 3b | Baseline (text→audio), identical size, data, steps, seed | ~3–5 h |
| 4 | Scale-up to ~800 h, only if 3a meets the pilot threshold | ~10–15 h |

All estimates are replaced by measured values in the reports.

- ~10% text-only batches during speech training, to keep text ability (text loss tracked).
- Fair comparison: 3a and 3b differ only in layout. A second seed is run if the WER gap is
  within plausible seed variation.
- Inherits everything from M1: config files, exact resume, wall-time stops, Hub sync,
  cost logging, NaN guards, reports.
- Every N steps: Pluto speaks the 10 frozen demo sentences, saved as `.wav` in `samples/`.

## 5. Evaluation

- **WER and CER** via Whisper on: 10 frozen demo sentences (versioned, like M1's prompts) and
  200 held-out test sentences (TinyStories-Speech test split), fixed sampling settings and seed.
- **Reference points:** ceiling (Kokoro audio → Mimi → decode → Whisper), Pluto main,
  Pluto baseline, floor (untrained model).
- **Duration ratio** (generated vs reference length) to detect skipped or repeated words.
- **Per-codebook audio loss** (bits), **text loss** vs the warm-start model.
- **Generation speed:** real-time factor with the KV cache.
- **Listening:** the report links the 10 demo `.wav` files at early, middle and final checkpoints.
- **Pass thresholds:** pilot WER ≤ 35%; scale-up target ≤ 15%. The comparison is claimed only if
  the gap exceeds seed variation; otherwise "no clear difference" is reported.

## 6. Demo and deliverables

- `llm speak <run> --text "…" --out out.wav`
- `llm continue-speech <run> --audio prompt.wav --out out.wav` (Whisper transcribes and times the
  prompt so the text stream is aligned).
- Gradio app for Kaggle/Colab: type text → hear Pluto; record a story start → Pluto continues.
- Report additions: WER table (ceiling / main / baseline / floor), comparison result, audio links,
  speed, per-codebook charts.
- Notebook `notebooks/m2_kaggle.ipynb` covering Stages 0–4 with Hub sync and resume.

## 7. Testing

CPU-only CI with the external models replaced by small fakes behind interfaces:
- **Data:** chunking keeps whole sentences and respects the duration limit; timings map to the
  correct frame indices; resumable generation skips finished chunks; manifests and provenance.
- **Layouts:** aligned and baseline sequence builders round-trip (build → split back to text and
  codes); acoustic delay applied and removed exactly; text never lags its audio.
- **Model:** shapes; causal masking across frames; the voice box only sees earlier codes of the
  same frame; parameter count formula; initial loss ≈ ln(vocab) per head.
- **KV cache:** cached generation equals uncached generation exactly (greedy) for both
  brain and voice box.
- **Training:** a tiny speech-text model overfits a few synthetic clips; resume is bitwise on
  CPU; warm-start loads text weights correctly and initialises the new parts.
- **Evaluation:** WER computation on known strings; duration ratio; scoring is deterministic.
- **End-to-end (slow):** synthetic "codec" + fake TTS → data → train → speak → report.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Kokoro timings unavailable or unreliable | Stage 0 checks first; Whisper word timestamps as fallback |
| Pluto babbles (voice box too weak) | Weighted code-1 loss, acoustic delay, warm-start; listen at every checkpoint; scale data before scaling model |
| Skipped/repeated words | Aligned layout; duration-ratio metric |
| Free-tier time per Kaggle session | Resumable data generation and training; Hub sync |
| Text ability forgotten | 10% text-only batches; text loss tracked |
| Overclaiming the comparison | Fixed rule: claim only beyond seed variation |
| Mimi/Kokoro/Whisper licence or availability changes | Pinned versions recorded in manifests; codec behind an interface |
