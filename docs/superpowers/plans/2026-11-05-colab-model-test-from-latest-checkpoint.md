# Colab Model Test from Latest Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a simple Google Colab notebook to test the latest M3 text checkpoint and the existing M2 voice checkpoint, with visible prompt-edit cells.

**Architecture:** The notebook uses two separate checkpoints: M3 for text and M2 aligned for speech, with an optional text-to-speech chain that sends M3 output to M2. A CPU-only Modal step publishes the M3 tokenizer to the private Hub repo; it runs after the active H100 session stops and does not commit the read-only Volume snapshot.

**Tech Stack:** Python 3.12, Jupyter/Google Colab, Pluto `llm` CLI and Python API, `huggingface_hub`, Modal.

**Spec:** `docs/superpowers/specs/2026-11-05-colab-text-voice-test-design.md`

## Global Constraints

- Notebook only downloads checkpoints and performs inference; it does not train, resume, or modify weights.
- Keep `HF_TOKEN` in Colab Secrets; never save credentials or publish generated outputs.
- Do not deploy or invoke the tokenizer publisher until the current H100 session has stopped cleanly.
- Make the M3 text model and M2 voice model distinction clear; they are not one unified model.
- Put prompt text in separately labeled editable cells and instruct the user exactly which variables to change.
- Do not add AI session links to any commit or repository file.

## Review Focus

- A private Hub run ID or missing checkpoint must produce a clear setup error, not a traceback that looks like model failure.
- A missing tokenizer artifact must give the exact publish step needed.
- A publisher-only CPU call must not commit a Volume snapshot, avoiding stale concurrent writes.
- The optional speech chain must use M2 weights and must label that fact in its output.
- Notebook shell commands and prompt strings must preserve quotes and Unicode safely.

---

### Task 1: Publish the M3 tokenizer artifact from Modal

**Files:**
- Create: `src/llm/hub.py`
- Modify: `modal_app/m3_train.py`
- Modify: `tests/test_modal_m3.py`
- Test: `tests/test_hub.py`

**Interfaces:**
- `publish_folder(local_dir: Path, hub_repo: str, path_in_repo: str, api=None) -> None` validates the local folder and manifest, creates the private model repo if needed, then uploads the folder to the requested Hub path.
- Modal CPU step name: `publish-tokenizer`; upload `tokenizer_dir(load_config(configs/m3-pilot.yaml))` to `tokenizers/m3-bpe-v1` in the `pluto-runs` repo.
- A publish-only `cpu_job` call performs no `volume.commit()`.

- [ ] **Step 1: Write failing tests** for upload path/private repo setup and missing tokenizer manifest; add a Modal step exposure assertion.
- [ ] **Step 2: Run tests to verify expected failures** with `pytest tests/test_hub.py tests/test_modal_m3.py -q`.
- [ ] **Step 3: Implement** the upload helper and CPU step; commit the Volume only for steps that mutate data, tokenizer, or shards.
- [ ] **Step 4: Run targeted tests** and `pytest -q -p no:cacheprovider` on CPU when dependencies are available.
- [ ] **Step 5: Commit** the publisher and tests. Do not deploy or run it until the current H100 session is stopped.

### Task 2: Build the ordered Colab inference notebook

**Files:**
- Create: `notebooks/model_test_from_latest_checkpoint.ipynb`
- Modify: `docs/03-status-and-next-steps.md`

**Interfaces:**
- Notebook sections: Goal; choose free GPU and add `HF_TOKEN` Colab Secret; clone/install; set M3 and M2 run IDs; download the latest checkpoint and M3 tokenizer; edit `TEXT_PROMPT`; generate text; edit `VOICE_PROMPT`; generate and play speech; optionally speak the M3 generated text with M2; troubleshooting.
- `TEXT_RUN_ID` defaults to auto-discovery of the newest `m3-pilot-*` run with a Hub `checkpoints/latest.json`; the user may override it. `VOICE_RUN_ID` is an editable variable defaulting to the existing M2 aligned run. The loader selects the latest valid checkpoint within each chosen run.
- Default M2 voice run: `m2-aligned-20261102-213616-72fb023`.
- No Hub or GitHub uploads occur from the notebook.

- [ ] **Step 1: Build notebook JSON** with short markdown and small code cells, keeping each setup/action in execution order.
- [ ] **Step 2: Validate notebook format and Python cell syntax** locally; assert prompt variables and required sections exist.
- [ ] **Step 3: Review all cell outputs/state** to ensure the notebook has no saved credentials, stale output, or hidden run-order dependency.
- [ ] **Step 4: Update** `docs/03-status-and-next-steps.md` with the notebook path, tokenizer publish prerequisite, and validation limits.
- [ ] **Step 5: Commit** the notebook, tests, and handoff update. State that Colab GPU execution remains for the user because it needs their private HF secret and free-runtime assignment.
