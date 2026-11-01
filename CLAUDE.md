# Project context for Claude

## North star
This project is the foundation for building **our own frontier-class multimodal LLM**: a model
that can **see (vision), talk (speech in/out, real-time), and text**, like frontier models.
Every milestone is a step on that path. Current near-term target: "Mini-Live" (~100M speech+text,
real-time voice), see `docs/01-decisions.md`.

## Design implications (apply at every stage)
- Core model consumes **modality-agnostic token streams**; new modalities (audio codec tokens,
  image patches/tokens) plug in as new tokenizers/encoders + vocabulary/embedding extensions,
  not rewrites.
- Keep components modular, config-driven, and tested; scale-up paths (DDP/FSDP, longer context,
  bigger vocab) must not be blocked by early shortcuts.
- But build only what the current milestone needs (YAGNI); don't pre-build future modalities.

## Working principles
- Reproducibility over impressive-looking results; claims must match measured evidence.
- One change at a time, each with hypothesis, baseline, controlled experiment, conclusion.
- Checkpoint/resume must work before long runs; never debug on paid GPUs.
- Paid GPU hard cap: $100 (target $40–80); log cost per run.
- Disclose any off-the-shelf components (e.g. Mimi codec, Kokoro TTS) honestly.
- Document failures and limitations openly.
