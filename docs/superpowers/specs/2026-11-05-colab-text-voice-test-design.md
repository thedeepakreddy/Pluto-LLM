# Colab text and voice test notebook

## Intent

Give Deepak a simple, top-to-bottom Google Colab notebook using a free GPU to try Pluto text generation and speech generation, with clearly labeled prompt variables.

## Approaches

- **Two current checkpoints (chosen):** load the M3 text pilot and the existing M2 aligned speech model separately. Optionally send generated M3 text to M2 `speak`. This tests the current artifacts while making clear they are not yet one unified model.
- **Wait for a unified M4 checkpoint:** simpler model story, but unavailable until the speech integration and training are complete.

## Design

- Create `notebooks/model_test_from_latest_checkpoint.ipynb`; order the cells as runtime check, secrets and setup, model/run IDs, text prompt and generation, voice prompt and audio playback, optional M3-text-to-M2-speech playback, and troubleshooting.
- Put the editable text and speech prompts in separate, conspicuous code cells. Explain exactly which lines to change.
- Store `HF_TOKEN` only in Colab Secrets. Do not write credentials or generated results to GitHub or the private Hub.
- Pull the M3 run checkpoint and the existing M2 aligned run from the private `pluto-runs` repo. The M3 tokenizer is not included in today's run upload, so add a CPU-only Modal step to publish that tokenizer folder to the same private repo.
- Wait until the current H100 session has stopped and committed its Volume before invoking the tokenizer publish step. The active run and its config remain untouched.
- The notebook performs inference only; it does not train, resume, or modify either checkpoint.

## Validation and limits

Validate notebook JSON and Python cell syntax locally. Colab GPU execution requires Deepak's Hugging Face secret and uploaded checkpoints, so provide an ordered run checklist and report that remote execution was not performed locally.

## Approval

Approved by the user on 2026-11-05.
