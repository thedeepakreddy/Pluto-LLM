# Roadmap after M2: M3 → M6 (plan, 2026-11-03)

Status: **plan, not started.** Numbers below are estimates; each is replaced by a measured value
(tokens/s from a short benchmark run) before any paid or limited GPU time is spent.

## Order of work

1. **Finish M2:** Whisper evaluation (Colab), demo audio, reports, results in the paper.
2. **Speed up training code** (bf16 on H100, FlashAttention / SDPA, `torch.compile`), tested on CPU/free GPUs.
3. **Modal setup + 10-minute benchmark** on an H100 to measure real tokens/s.
4. **M3** → **M4** → **M5** in stages (below), each with a design doc, tests and a controlled comparison.
5. **M6** research agent once M5 works.

## Milestones

| Milestone | What | Pluto size |
|---|---|---|
| M1 ✅ | text pipeline, TinyStories | ~6M |
| M2 ✅ (evaluating) | first voice, 200 h TinyStories-Speech | ~30M |
| M3 | smarter text brain on a high-quality text mix | ~100M |
| M4 | M3 brain + voice box, real human voices | ~110–120M |
| M5 | instructions + spoken dialogue, real-time streaming ("Mini-Live") | same as M4 |
| M6 | autonomous research agent that experiments on Pluto (see below) | no new Pluto weights |
| Later | better data mixes, bigger model (~300M–1B), vision | decided after M5 |

## Datasets to consider (approximate public sizes; verify version and license before use)

| Skill | Dataset | Approx. size |
|---|---|---|
| Knowledge | FineWeb-Edu (`sample-10BT`) | 10B tokens (full set ~1.3T) |
| Knowledge | English Wikipedia | ~4–5B tokens |
| Math | OpenWebMath; FineMath | ~15B; ~10–35B tokens |
| Code | The Stack v2; StarCoder data | ~1T; ~250B tokens |
| Books | Project Gutenberg (public domain) | ~60k books, a few B tokens |
| Instructions / chat | SmolTalk (and similar) | ~1M conversations |
| Real voices | LibriSpeech; Mozilla Common Voice (EN) | 960 h; a few thousand h |
| Spoken dialogue | synthetic (generated); licensed corpora | as generated; Fisher ~2,000 h (paid) |
| Identity | private creator dataset (private HF dataset) | a few hundred statements |

Rules: check every license (non-commercial, research-only and model-generated data carry terms) and
disclose all sources; quality over quantity; measure each dataset's effect before/after.

A ~100M model uses roughly **2–10B tokens** well, so we take slices and mix, e.g. ~70% FineWeb-Edu,
~10% Wikipedia + books, ~10% math, ~5% code, ~5% instructions (at the end). The exact mix is an
experiment (a good first job for M6).

## Staged training plan (fits the ~7.5 free Modal H100 hours)

| Stage | Data | H100 time (est.) |
|---|---|---|
| M3 pretrain | mixed text, ~5B tokens | ~2–3 h |
| M4 voice | LibriSpeech + Common Voice + M2 data, ~1,000–2,000 h | ~1–1.5 h |
| M5 dialogue | instructions + synthetic spoken dialogue | ~0.5–1 h |
| Margin | benchmark, retries, evaluation | ~2 h |

Why stages, not one run: learning order matters (knowledge → voice → conversation), a failure costs
one stage instead of the whole quota, and each stage's effect can be measured.

Calibration note: on M2 my estimates were ~3× too pessimistic (Stage 3 estimated 2–4 h, took 58 min).
Revised training total for M3–M5 is likely ~1–3 H100 hours; plan for the table above anyway.

Work that does **not** run on the H100: downloading/tokenizing text (CPU, hours), encoding
1,000–2,000 h of audio into Mimi tokens (free Kaggle/Colab GPUs, likely 10–20+ h over sessions),
evaluation, and engineering. Never debug on the H100.

## M6: research agent (what it is)

A program (a loop around a frontier model used through an API, e.g. Claude) that runs the research
loop on Pluto: propose a hypothesis with a written prediction → run a controlled experiment with the
existing CLI (`--set` overrides or sampling flags; never edits code or a running run) → read
`metrics.jsonl` / `speech_eval.json` → compare to a baseline → write a report → propose the next step.

Guardrails: free tests first; paid/limited GPU only with human approval; one change per experiment;
results only from runs that actually happened. Disclosure: "a research agent powered by <model> that
experiments on Pluto", never "Pluto researches itself".

## Model size: decide during M3 (open decision)

Two candidate designs (parameter counts calculated; the code prints exact counts):

| | Option A | Option B |
|---|---|---|
| Brain | 12 layers × 768, 32k vocab | 24 layers × 1024, 32k vocab |
| M3 total | ~134M (85M non-embedding) | ~368M (302M non-embedding) |
| M4 / M5 total (+ voice box) | ~160M | ~400M |
| M3 tokens | ~5B (~37 tokens/param) | ~7–10B |
| M3 H100 time (estimate) | ~1.5–4 h | ~6–14 h |
| Compute | free Modal hours | needs cloud credits (GCP / Azure) |

**Decision rule (agreed 2026-11-03):** run the Option A pilot and the 10-minute H100 benchmark first,
measure real tokens/s, then **ask the user** whether to continue with A or scale to B, using the
measured time and cost. No size is fixed before that measurement.

## Product goal: ultra-low latency, good quality, reasoning when needed (agreed 2026-11-03)

Design: **fast talker + deep thinker** (do not make one model do everything).

1. **Fast talker (always on):** the small Pluto (Option A size, ~135–160M) listens and answers
   immediately using the aligned streaming layout, KV cache and later quantization. It handles most
   conversation. Quality improves through better data, continued pre-training and distillation from
   larger models; none of this changes its size, so latency stays the same.
2. **Deep thinker (only when needed):** for hard questions (math, planning, multi-step reasoning) Pluto
   decides it needs to think, says a natural filler right away ("let me think about that…"), the
   thinker works in the background, and Pluto speaks the result. Candidates, in order: Pluto in a
   step-by-step "think-first" mode; later a larger Pluto (~400M–1B, the Option B size); during
   development an API model, always disclosed.

| Goal | How | When |
|---|---|---|
| Ultra-low latency | small model, aligned streaming, KV cache, quantization | M3–M5 |
| Output quality | quality data mix + distillation from larger models | M3 onward |
| Reasoning when needed | math/code data, step-by-step reasoning examples, think-first mode, a router that decides when to think | M5+ |
| Measured, not claimed | time to first spoken word (target ~300–500 ms), RTF, WER, quality and reasoning benchmarks | every milestone |

Consequences now: M3 leans to **Option A** (the fast talker), Option B is the future thinker; M3's data
mix includes math/reasoning data. The final size decision is still taken after the H100 benchmark.
Measured so far: M2 aligned (~30M) RTF 0.34 on a T4.
