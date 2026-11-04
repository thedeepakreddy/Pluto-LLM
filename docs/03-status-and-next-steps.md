# Pluto: status and next steps (living handoff)

Updated 2026-11-04. Read this first when picking the project up (any person or AI assistant).
Then read: `CLAUDE.md` (rules), `docs/superpowers/specs/2026-11-03-pluto-architecture-design.md`
(the approved architecture), `docs/superpowers/plans/2026-11-04-m3-text-brain.md` (the M3 plan),
`docs/01-decisions.md` (decisions log), `docs/papers/pluto-m1-m2-technical-report.md` (results so far).

## Where we are

| Milestone | State |
|---|---|
| M1 text pipeline, M2 speech (~30M) | **done**, reported (§3.8 of the technical report) |
| M3 text brain: code | **done and tested** (393+ tests), on `main` |
| M3 data preparation on Modal | **running / to restart** (see "Next steps") |
| M3 A/B tests, H100 bench, size decision, training | not started |

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
7. **Train** (H100, `--detach`, stops cleanly at 23.5 h; continue with `--steps resume --run runs/<id>`):
   foundations stage (~20% of tokens), then the full mix with `train.init_from`, then `m3-extend16k`.
8. **Evaluate + report**: bits/byte, HellaSwag/ARC-Easy/PIQA (Task 6, multiple-choice eval, **not written
   yet**), samples; M3 section in the report; README; cost.

## Still to build

- Task 6: `src/llm/eval/multiple_choice.py` + `llm eval-text` (needed before the M3 report, not before training).
- Final whole-branch code review of the M3 work (plan's last step).
- Later: code data (data v2), StackExchange/textbooks, MTP speculative decoding (only if the A/B keeps MTP).

## Budget and rules (unchanged)

Free compute up to ~400M; the $100 is reserved for growing Pluto to ~500M. Never debug on H100s; rehearse
on L4. Log cost per run. Disclose off-the-shelf components (Mimi codec, Kokoro TTS, Whisper for scoring,
FlexAttention/PyTorch kernels). Commit trailer only `Co-Authored-By: thedeepakreddy <thedeepakreddy1@gmail.com>`;
never put AI session links in the repo. Never paste tokens in chat; use Modal secrets / the dashboard.
