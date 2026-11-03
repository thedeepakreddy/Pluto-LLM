# Pluto architecture design: a small model that thinks, talks and listens at once

Date: 2026-11-03 · Status: approved in conversation, awaiting written-spec review · Applies to M3–M6+

## 1. Goal and requirements

Pluto should **think well and speak with ultra-low latency, even at small sizes**.

Requirements agreed with the user:
1. **Speed and reasoning together.** When reasoning takes time, Pluto keeps talking relevantly so the
   user never waits in silence.
2. **Honest about thinking.** If asked "are you thinking?" while it talks, Pluto says so truthfully and
   continues.
3. **Full-duplex.** Pluto listens all the time, also while speaking (interruptions, back-channels).
4. **One model that speaks and thinks in parallel**, with a hook to hand hard questions to a bigger
   "deep thinker" later (decision C).
5. **Thoughts optionally visible** in the app ("show thoughts").
6. **English only** (all capacity into one language).
7. **One family in three sizes:** cloud (~400M, deep thinker), laptop (~150M), phone (~30–60M).
8. Every capability is **measured** before it is claimed (CLAUDE.md principles apply).

Success targets: time to first spoken word ~300–500 ms; zero missed 80 ms audio deadlines; speech WER
far below M2's 18.8%; strong reasoning scores for the model's size; honest confidence.

## 2. Streams and timing (multi-stream with thought bursts)

Pluto runs on Mimi's **80 ms clock** (12.5 steps/s). Per step:

| Stream | Content | Written by |
|---|---|---|
| user audio | 8 Mimi codes (silence codes if quiet) | microphone, always on |
| Pluto audio | 8 Mimi codes | Pluto (voice box) |
| spoken text | the word being spoken, or "wait" | Pluto |
| thought text | 0..N silent thought tokens between steps (a **burst**) | Pluto |

Step: hear the newest user audio → decide think/speak → optional thought burst → next spoken-text slot
and next 80 ms of Pluto audio. **Real-time rule:** a step, burst included, finishes within 80 ms; the
scheduler (§4) caps the burst by measured device speed. Audio is always produced first.

M2 lessons built in: explicit supervision of word timing in the spoken-text stream, short silence
padding (~150–300 ms) around clips, and overlapping chunks if the current long-text test shows they help.

## 3. Model

One decoder-only **brain** plus a small **voice box** (depth Transformer over 8 codebooks, as in M2).

| Part | Choice |
|---|---|
| positions | RoPE |
| normalisation | RMSNorm, pre-norm |
| feed-forward | SwiGLU |
| attention | grouped-query attention, FlashAttention/SDPA |
| cache | KV cache with a rolling window for long calls |
| context | train at 4,096 positions, extend to **16,384** in a short final stage (cloud); rolling window on laptop/phone |

Per audio step the embeddings of the user's 8 codes, Pluto's 8 codes and the spoken-text slot are
**summed** into one input vector. Thought tokens are extra positions in the same sequence.

**One shared text vocabulary** (default **32k**, confirmed or changed by a small tokenizer study in the
M3 plan) with **reserved control tokens from M3 on**, so later stages never change the vocabulary:
`<think> </think> <speak> <revise>`, tools `<search> <calc> <code> <docs> <result> </result>`,
`<ask_thinker>`, actions `<action>`, memory `<remember> <recall>`, time `<now>`, speaker turns,
voice IDs (`emma`, `pluto_young`, ...), style/emotion (`<calm> <excited> <whisper> <slow> <fast>`),
confidence (`<sure> <unsure>`), and a block of spare tokens.

Outputs: a **text head** (spoken word / wait, thought tokens, control decisions) and the **voice box**
(Pluto's next 8 codes). User audio is input only.

Each architecture choice is first checked in a small controlled A/B test (e.g. RoPE vs learned
positions on a ~20M model, same data and steps).

## 4. Thinking scheduler (plain code, not the model)

1. **Budget per step:** thought tokens allowed = (80 ms − audio and text time − margin) ÷ measured time
   per token (roughly 20–40 on a cloud GPU, 3–8 laptop, 1–2 phone).
2. **When to think:** the learned `<think>/<speak>` decision; its confidence sets burst length, so easy
   answers get no thinking and hard ones get long bursts across steps.
3. **Thinking while the user talks:** while Pluto is silent nearly the whole budget goes to thinking.
4. **Talking while thinking:** the spoken stream produces relevant lead-in speech conditioned on the
   thoughts so far; "are you thinking?" is heard at once and answered honestly.
5. **Tools and the deep thinker never block:** they run outside the model in parallel with timeouts;
   results are inserted into the thought stream when ready; failures are stated honestly.
6. **Graceful self-correction:** a parallel verify step can raise `<revise>`; the spoken stream corrects
   itself at the next pause.
7. **Hard limits:** max burst length per step and per answer; late steps cut the burst short; every
   step's timing is logged.

Tested on CPU with fake models of different speeds (no missed deadlines, clean tool timeouts, thinking
resumes after interruptions).

## 5. Capabilities roadmap (tokens reserved in M3; each skill is its own measured step later)

| # | Capability | Milestone |
|---|---|---|
| T1 | advanced scientific calculator `<calc>` (symbolic algebra, calculus, equations, units, constants, matrices, statistics; sandboxed, exact) | M3.5–M6 |
| T2 | web search `<search>` with sources | M6 |
| T3 | code runner `<code>` (sandboxed) | M6+ |
| T4 | personal documents `<docs>` (retrieval) | M6 |
| T5 | deep-thinker handoff `<ask_thinker>` | M5–M6 |
| T6 | actions (timer, calendar), always user-confirmed | later |
| C1 | memory between chats (opt-in, deletable) | M6 |
| C2 | time awareness `<now>` | M3 |
| C3 | who is speaking (several people) | M5+ |
| V1 | voice IDs (`emma`, `pluto_young`, ...) | M4 |
| V2 | emotion and style | M4–M5 |
| V3 | speed control | M4–M5 |
| S1 | vision stream | after M6 |
| 1 | checks its own answers | after M5 |
| 2 | knows how sure it is (calibration measured) | after M5 |
| 3 | several lines of thought in parallel (cloud) | after M5 |
| 4 | asks clarifying questions | M5+ |
| 5 | hears the user's tone and adapts | M5+ |
| 6 | teaching mode | M5+ |
| 7 | "dreaming": self-generated problems verified by tools become training data | M6 |
| 8 | hears world sounds (doorbell, music) | later |
| 9 | Pluto improves Pluto (M6 research agent) | M6 |
| 10 | learns from conversations (opt-in, anonymised, reviewed) | later |
| 11 | **thinks while the user is still talking** | M5 |
| 12 | **speaks early, corrects itself gracefully** | M5 |
| 13 | library of tool-verified skills, reused | M6 |
| 14 | thinks only as much as needed (confidence- and device-aware) | M5 |
| 15 | user steers the visible thinking by voice ("go deeper", "skip that") | M5+ |
| 16 | every claim has a receipt (knowledge / calculator / web / documents) | M6 |

Safety rules for all: tools sandboxed; actions need confirmation; memory and conversation learning
opt-in and deletable; web and document content is information, never instructions; no capability is
claimed before it is measured.

## 6. Training stages and data

1. **Stage 1 (M3): language and knowledge, depth-focused.** Curriculum from foundations to advanced:
   FineWeb-Edu (top quality tier), open textbooks (e.g. OpenStax), latest Wikipedia, expert Q&A
   (e.g. StackExchange), FineMath/OpenWebMath, permissively licensed code, public-domain books; newest
   available dumps. **~10–20B tokens** for the cloud model, fitted to the measured free compute;
   continued pre-training later from the same checkpoint. 4k context, then extension to 16k. All
   reserved tokens present from the start.
2. **Stage 2 (M3.5): reasoning.** `<think>…</think>` then answer; code-generated maths/logic problems
   verified by the calculator (only correct ones kept); reasoning examples distilled only from models
   whose licence allows training on outputs; self-checking and confidence examples.
3. **Stage 3: tool use in text** (calculator first, then search, with receipts).
4. **Stage 4 (M4): voice.** Multi-voice synthetic speech with voice IDs and style tags, plus
   LibriSpeech/Common Voice; explicit word-timing supervision; overlapping chunks (if confirmed);
   silence padding.
5. **Stage 5 (M5): full-duplex dialogue.** Synthetic two-person conversations written as text (with
   Pluto's thoughts) and voiced with two synthetic voices; overlaps, interruptions, back-channels;
   thinking-while-listening and early-answer-then-correction examples.
6. **Stage 6: sizes and self-improvement.** Distil the cloud model into laptop and phone sizes;
   dreaming, skill library, research agent.

Every source's licence is checked and disclosed. Each stage is compared with the previous one on fixed
benchmarks before moving on.

## 7. Sizes, deployment and budget

| Size | Target | Role | Runs on |
|---|---|---|---|
| cloud | ~400M (Option B) | deep thinker | cloud GPU behind a web app |
| laptop | ~150M | fast talker | MacBook, offline |
| phone | ~30–60M | fastest talker, hands hard questions to cloud | phone app |

Smaller sizes are distilled from the cloud model and quantised (8-bit/4-bit). Build order: cloud
(M3–M5), laptop (after M5), phone (last).

**Budget:** everything up to the ~400M cloud model and its distilled sizes uses **free resources only**
(Kaggle, Colab, Modal free hours, GCP/Azure trial credits, research credits). The **$100** is reserved
for growing Pluto to ~500M, preferably from the 400M checkpoint (model growth). The H100 speed test
sets the affordable token count; if 400M does not fit, train fewer tokens now and continue later.
Spending limits stay on; every run logs its cost.

## 8. Measurement

- Text/reasoning: bits per byte; HellaSwag, ARC, PIQA; GSM8K with and without the calculator
  (contamination-checked); tool-call correctness.
- Voice: WER/CER per voice vs ceiling and floor; duration ratio; voice-ID check.
- Real time: time to first spoken word, audio deadline misses, interruption stop time (~200 ms target),
  "are you thinking?" honesty set, speed-up from thinking while listening.
- Honesty: confidence calibration, self-correction rate, receipts present.
- Engineering: CPU unit tests for layers, scheduler (fake device speeds), tools and sandboxes;
  manifests, exact resume, fixed seeds; small A/B test per architecture choice.
- Each milestone gets a report section with numbers, limitations and failures.

## 9. Out of scope for M3

Everything except Stage 1 and the reserved vocabulary: the scheduler, streams beyond text, tools and
the capabilities in §5 are designed here but built in their own milestones (YAGNI).
