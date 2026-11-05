# Pluto: status and next steps (living handoff)

Updated 2026-11-05. Read this first when picking the project up (any person or AI assistant).
Then read: `CLAUDE.md` (rules), `docs/superpowers/specs/2026-11-03-pluto-architecture-design.md`
(the approved architecture), `docs/superpowers/plans/2026-11-04-m3-text-brain.md` (the M3 plan),
`docs/01-decisions.md` (decisions log), `docs/papers/pluto-m1-m2-technical-report.md` (results so far).

## LATEST STATE (2026-11-05) — start here

**UPDATE (A/B results and H100 bench in):** see `docs/experiments/m3-layer-ab.md`. Recipe chosen: RoPE + RMSNorm +
SwiGLU + GQA + hybrid sliding window (1024, every 4th layer global); **MTP dropped** (+0.003 bpb, no gain).
Pilot, cloud and extend16k configs updated. H100 pilot baseline: 287,403 tokens/s, 26.4% MFU, 1.82 s/step,
28.5 GB peak memory, about 4.8 h for 5B tokens. Bench sweep variants ranged 4.6–5.3 h; see the dashboard
table for detail. Using current Modal H100 + CPU/memory rates, budget roughly $22 for the measured 4.8 h;
the current $10.92 balance alone will not cover the full run. The 5B training schedule is configured below;
training has not started.

**UPDATE (sessioned 5B pilot configured):** `train.max_steps=9537` × `tokens_per_step=524288` = **5,000,134,656
tokens**. `train.max_wall_time_s=6000` is checked against each `train()` invocation, so it resets on every
resume; cumulative elapsed time remains in checkpoint `TrainState.wall_time_s`. Resume config validation now
ignores only `train.max_wall_time_s`, allowing the per-session cap to change while preserving the original
schedule and all other run settings. The 6,000-second cap targets about 1h40 of loop time per session, leaving
roughly 20 minutes for Modal startup, compilation, final checkpoint and Hub upload; at the
bench rate, the budget is roughly 4.6–4.8 H100-hours, likely three sessions. `cost_per_hour=4.65` estimates
H100 plus the Modal function CPU and memory for run cost accounting. Each wall-time stop saves a full
checkpoint and requires the configured Hub sync to succeed before returning; if sync fails, the checkpoint
remains on the Modal Volume and the session reports a failure rather than claiming a successful handoff. Python
syntax and the config-diff smoke check passed; pytest could not run because it is absent locally, disk was full
during dependency installation, and package-index DNS is unavailable.

**Completed:** all 8 layer A/B trainings on Modal A10G (24M `m3-ab-base`, same seed, 300M tokens each): base,
rope, rmsnorm, swiglu, gqa, window, mtp, all.

**Done since the sections below were written:**
- Data prepared on the `pluto-m3` Volume: **train 5,877,776,453 tokens** (59 shard files), **val 23,940,308
  tokens**; 32k tokenizer with the reserved specials. (The per-reason drop counts in
  `data/m3-mix-v1/data_report.md` have not been reviewed yet: run `--steps summary` and check that drops look
  sane, roughly 10–40%.)
- GPU rehearsal (A10G, `bench --sweep` on m3-ab-base): FlashAttention used everywhere; sliding window runs
  (flex); GQA speeds batch-1 decode by +19–25% (368 → 437/459 tok/s); MFU 18–29% (tiny model); 1k → 16k context
  halves throughput at 24M. Fixed after it: FlexAttention now only inside compiled graphs (`cdbdc00`).
- Fixes: parallel data prep with 8 workers (byte-identical output), `modal deploy` + `--spawn` (jobs survive
  the laptop sleeping), the `ab` launch check (`a9b6b46`).

**Decisions taken by the user:**
- Train the **151M pilot (`configs/m3-pilot.yaml`) first** on the H100 (bench measured ~4.8 h for 5B tokens;
  current Modal rates estimate ~$22 including requested CPU/memory), then decide
  about the 388M cloud model, which should use free/credit compute (Modal startup/academic credits, Kaggle,
  cloud trials). Assistant advised against using a friend's Modal account/credits for this project
  (account-sharing / multi-account credit use likely breaks Modal's terms); a real contributor running his own
  jobs, or credit programs, are the clean routes.
- One model, staged: M3 trains the text brain; M4 continues the **same** brain with speech (`init_from`).
  Before M4, port `SpeechGPT` (src/llm/model/voice.py, still M2 layers) to the modern layers.

**Next actions for the new session (in order):**
1. Run CPU tests for the session cap, resume config comparison, and required handoff upload.
2. After user approval, deploy the H100 Modal app and start the configured 5B-token pilot with `--spawn`.
   Each session stops at 6,000 s; resume the same run ID next month.
3. While it trains: write Task 6 (multiple-choice eval, `src/llm/eval/multiple_choice.py`, `llm eval-text`),
   and add a step that uploads the prepared shards to the private HF repo (portable data for Kaggle/other
   compute).
4. After training: eval (bits/byte, HellaSwag/ARC-Easy/PIQA with contamination check, samples), M3 report
   section, README, cost log; final whole-branch review of the M3 code.
5. As a separate follow-up experiment, implement identity-initialized residual block expansion and compare an
   expanded checkpoint against its original on held-out evaluations; see `docs/01-decisions.md`.

**Practical notes for helping this user:**
- The user runs commands on a Mac (zsh) in `~/Pluto-LLM` with a `modal-env` venv. Give commands one per
  code block, never with `<placeholders>` (zsh treats `<` as a redirect; the user pastes literally).
- `PLUTO_GPU` is fixed at **deploy** time for `--spawn` jobs: redeploy when switching GPU (A10G for cheap tests;
  L4 often queues; H100 for real training). Redeploy after any pull that changes `modal_app/m3_train.py`.
- Never Ctrl+C a running job's terminal; use `--spawn` for anything long.
- Pushes go to `main` (Modal clones `main`); the old `claude/...` branch is still GitHub's default branch until
  the user switches it in Settings.

## Where we are

| Milestone | State |
|---|---|
| M1 text pipeline, M2 speech (~30M) | **done**, reported (§3.8 of the technical report) |
| M3 text brain: code | **done and tested** (393+ tests), on `main` |
| M3 data preparation on Modal | **done** |
| M3 A/B tests, H100 bench | **done**; pilot training not started |

### M2 final results (for context)
Text-then-audio 18.8% WER (ceiling 1.3%, floor 689%); aligned 192%. Aligned decoding fixes: pacing guard +
text temperature 1 → 91%. Long-text mode (user's chunking idea): **independent 20-word parts → 40%** (aligned);
carrying the previous part as context makes both layouts worse (the models were not trained on such
prompts). Overlapping chunks therefore become a *training* change for M4.

## What was built for M3 (all on `main`)

1. **Model** (`src/llm/model/gpt.py`, switches in `ModelConfig`; defaults reproduce M1/M2 exactly):
   RoPE, RMSNorm, SwiGLU, grouped-query attention (GQA), **hybrid sliding-window attention**
   (`sliding_window`, `global_every`; FlexAttention on GPU, bounded KV cache), **multi-token prediction**
   (`mtp_heads=1`, DeepSeek-V3 style, training-only auxiliary loss), `train.init_from` (curriculum stages,
   context extension).
2. **Tokenizer**: 32k BPE with reserved control tokens (`PLUTO_SPECIALS`: `<think>`, `<calc>`, `<search>`,
   voice IDs, `<sure>/<unsure>`, …) + 64 spares; literal text never becomes a control token; trained on a
   sample spread across the whole data mix.
3. **Data** (`src/llm/data/`): multi-source streaming with byte caps and score filters; quality filters
   (Gopher-style heuristics + English stop-word check), normalised exact dedup, MinHash near-dedup (global),
   13-gram **benchmark decontamination** (HellaSwag, ARC, PIQA, GSM8K), long-document chunking; every drop
   counted by reason in `data_report.md`.
4. **Shards**: parallel tokenization (`shards.workers`), byte-identical to single-process.
5. **Benchmark**: `llm bench <config> [--sweep]`: tokens/s, MFU, memory, attention kernel actually used
   (flash/efficient/math, +flex for windows), batch-1 decode speed, hours for the token budget; sweep over
   KV heads, sliding window, 4k/8k/16k context.
6. **Configs**: `configs/m3-ab-base.yaml` (24M, A/B tests), `m3-pilot.yaml` (151M, laptop size, Option A),
   `m3-cloud.yaml` (388M, cloud size, Option B), `m3-extend16k.yaml` (16k context stage).
   Data mix `m3-mix` v1 (~28 GB raw): FineWeb-Edu sample-10BT (int_score ≥ 3), Wikipedia 20231101.en,
   FineMath 4+, Project Gutenberg. **No code yet** (`bigcode/the-stack-smol` stayed gated).
7. **Modal app** `modal_app/m3_train.py`: CPU steps `check, prepare, tokenizer, shards`; GPU steps
   `bench, train, resume, ab`; `summary`. Volume `pluto-m3`, secret `pluto-hf`, Hub repo `<hf-user>/pluto-runs`.

## What happened on 2026-11-03/04 (lessons)

- `--steps check` caught two problems cheaply: PIQA's dataset script (fixed: `baber/piqa`) and the gated
  code dataset (dropped for v1).
- The first `prepare` run was **cancelled after ~30 min by Ctrl+C** in the terminal that started it,
  even with `--detach`. **Rule: after starting a detached job, close that terminal window; never Ctrl+C.**
- The second run died when the laptop lid was closed: `--detach` still needs the Mac awake.
  Fixed with `modal deploy` + `--spawn` (the job runs on the deployed app).
- The first run's traceback revealed a real bug: 32-bit n-gram hashes made decontamination flag ~35% of clean pages.
  Fixed with 64-bit BLAKE2b hashes + a regression test (commit `b832439`). Data must be (re)prepared
  with this fix; the cancelled run produced nothing usable.
- Git: the repo's GitHub default branch is the old `claude/...` branch; set **`main` as default** in
  GitHub Settings → Branches before deleting that branch. Pushes go to `main` (Modal clones `main`).

## Next steps (in order)

Run on the Mac, in the `Pluto-LLM` folder (`git pull origin main` first):

1. **Data** (CPU, ~4–8 h, near-free):
   `modal deploy modal_app/m3_train.py` then
   `modal run modal_app/m3_train.py --steps prepare,tokenizer,shards --spawn` → the job runs on Modal alone;
   the laptop may sleep. (`--detach` still died when the laptop lid was closed.) Re-deploy after pulls.
   Watch: Modal dashboard, or `modal app list` + `modal app logs <ap-id> | tail -20`
   (lines like `[prepare] fineweb-edu: received ...`). If one step fails, rerun only the remaining steps.
2. **Check the data**: `modal run modal_app/m3_train.py --steps summary` → read the per-reason drop counts
   (too strict / too loose?) and the token counts (= the real token budget).
3. **L4 rehearsal** (first real test of FlexAttention):
   `PLUTO_GPU=L4 modal run modal_app/m3_train.py --steps bench --config configs/m3-ab-base.yaml --sweep`
4. **Layer A/B tests** (L4, ~8 runs × ~1 h; rerun to continue, finished variants are skipped):
   `PLUTO_GPU=L4 modal run --detach modal_app/m3_train.py --steps ab` → fill `docs/experiments/m3-layer-ab.md`
   (val loss, bits/byte); keep only changes that help (base, rope, rmsnorm, swiglu, gqa, window, mtp, all).
5. **H100 bench** of both sizes with the chosen layers (`--steps bench --config configs/m3-pilot.yaml --sweep`,
   then `m3-cloud.yaml`).
6. **User gate**: choose size (150M vs 400M) and token budget from the measured hours; update the config's
   `max_steps`; record in `docs/01-decisions.md`.
7. **Train** (H100, `--spawn`, stops cleanly at 6,000 s per session; resume the same run next month).
   The current decision is one full-mix 5B-token pilot; foundations and 16k extension remain later decisions.
8. **Evaluate + report**: bits/byte, HellaSwag/ARC-Easy/PIQA (Task 6, multiple-choice eval, **not written
   yet**), samples; M3 section in the report; README; cost.

## Still to build

- Task 6: `src/llm/eval/multiple_choice.py` + `llm eval-text` (needed before the M3 report, not before training).
- Final whole-branch code review of the M3 work (plan's last step).
- Later: code data (data v2), StackExchange/textbooks; MTP revisits (scale A/B before 388M; post-hoc MTP head for self-speculative decoding in M5) — see docs/experiments/m3-layer-ab.md.

## Budget and rules (unchanged)

Free compute up to ~400M; the $100 is reserved for growing Pluto to ~500M. Never debug on H100s; rehearse
on L4. Log cost per run. Disclose off-the-shelf components (Mimi codec, Kokoro TTS, Whisper for scoring,
FlexAttention/PyTorch kernels). Commit trailer only `Co-Authored-By: thedeepakreddy <thedeepakreddy1@gmail.com>`;
never put AI session links in the repo. Never paste tokens in chat; use Modal secrets / the dashboard.
