"""Finish M2 on Modal with an ordinary GPU (default L4); the H100 is kept for M3+.

Runs the same steps as notebooks/m2_finish.ipynb in a Modal container: pull data and runs from the
private Hugging Face repo, baseline evaluation, aligned floor + identity, drift diagnostics, demo
audio and reports. Every finished evaluation phase is uploaded to the Hub by `llm eval-speech`, and
the work folder lives on a Modal Volume, so a stopped run continues where it left off.

One-time setup (on your computer):
    pip install modal && modal setup
    modal secret create pluto-hf HF_TOKEN=<your Hugging Face write token>

Run (``--detach`` keeps it running if you close the terminal):
    modal run --detach modal_app/m2_finish.py
    modal run --detach modal_app/m2_finish.py --steps baseline,floor     # only some steps
    PLUTO_GPU=T4 modal run --detach modal_app/m2_finish.py                # a different GPU
    modal run modal_app/m2_finish.py --steps summary                      # print all results (CPU)

Steps: setup, diagnostics, baseline, floor, demo, report (default: all, in that order).
PLUTO_BASELINE_N (default 50) sets how many test chunks the baseline is scored on.
Logs: printed here and in the Modal dashboard. Cost: shown in the Modal dashboard (an L4 or T4
for ~3 h is a few dollars of credit).
"""

from __future__ import annotations

import os

import modal

REPO_URL = "https://github.com/thedeepakreddy/Pluto-LLM.git"
BRANCH = "main"
GPU = os.environ.get("PLUTO_GPU", "L4")
IDENTITY_DATASET = "thedeepakreddy/pluto-creator-identity"  # private HF dataset ("" = off)
ALIGNED_RUN = "runs/m2-aligned-20261102-213616-72fb023"
BASELINE_RUN = "runs/m2-baseline-20261102-213616-72fb023"
WARM = "configs/m2-warmstart.yaml"
ALIGNED = "configs/m2-speech-aligned.yaml"
# Order: the most informative and cheapest results first. The baseline's babbling outputs are slow
# to transcribe (~2.5 min per chunk on an L4), so it is scored on the first 50 test chunks: the
# same chunks as the first 50 of the aligned 200-chunk result, so the comparison stays matched.
ALL_STEPS = ("setup", "diagnostics", "baseline", "floor", "demo", "report")
BASELINE_N = int(os.environ.get("PLUTO_BASELINE_N", "50"))
SPACY_WHEEL = ("en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/"
               "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
DIAGNOSTICS = [("--temperature 0.8 --max-pad-run 10000", "t0.8_k50_pad10000"),  # stats only
               ("--temperature 0.5", "t0.5_k50"),
               ("--temperature 0", "t0_k50"),
               ("--temperature 0.8 --max-pad-run 12", "t0.8_k50_pad12"),
               ("--temperature 0.5 --max-pad-run 12", "t0.5_k50_pad12")]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "espeak-ng")
    # Dependencies only; the repo itself is cloned at run time so the latest code is always used.
    .pip_install("torch>=2.2", "numpy", "pyyaml", "regex", "datasets", "matplotlib",
                 "huggingface_hub", "kokoro>=0.9", "misaki[en]", "transformers>=4.45",
                 "soundfile", SPACY_WHEEL)
)
app = modal.App("pluto-m2-finish", image=image)
volume = modal.Volume.from_name("pluto-work", create_if_missing=True)
WORK = "/vol/Pluto-LLM"


def _sh(cmd: str, check: bool = True) -> int:
    """Run a shell command in the repo, streaming output (Whisper's noisy warning filtered).
    Only fixed commands built in this file are run (no external input)."""
    import subprocess

    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=WORK, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "Whisper did not predict" not in line:
            print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc


@app.function(gpu=GPU, timeout=6 * 3600, volumes={"/vol": volume},
              secrets=[modal.Secret.from_name("pluto-hf")])
def finish(steps: list[str]) -> None:
    import glob
    import subprocess

    from huggingface_hub import HfApi, snapshot_download

    if not os.path.exists(WORK):
        subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, WORK], check=True)
    else:
        subprocess.run(["git", "-C", WORK, "fetch", "-q", REPO_URL, BRANCH], check=True)
        subprocess.run(["git", "-C", WORK, "reset", "-q", "--hard", "FETCH_HEAD"], check=True)
    _sh("pip install -q --no-deps -e .")
    _sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    hub = HfApi().whoami()["name"] + "/pluto-runs"
    print("Hub:", hub, flush=True)
    id_flags = (f"--set identity.hf_dataset={IDENTITY_DATASET} --set identity.speech_every=100"
                if IDENTITY_DATASET else "")

    try:
        if "setup" in steps:
            speech = "speech/tinystories-speech-v1"
            if not os.path.exists(f"{WORK}/{speech}/manifest.json"):
                snapshot_download(hub, allow_patterns=[speech + "/*"], local_dir=WORK)
            if not glob.glob(f"{WORK}/tokenizers/*/manifest.json"):
                _sh(f"llm prepare-data {WARM}")
                _sh(f"llm train-tokenizer {WARM}")
            if IDENTITY_DATASET:
                _sh(f"llm prepare-identity {ALIGNED} {id_flags}")
            for run in (ALIGNED_RUN, BASELINE_RUN):
                _sh(f"llm pull-run {hub} {os.path.basename(run)}")
            volume.commit()
        if "diagnostics" in steps:
            for flags, tag in DIAGNOSTICS:
                print(f"\n===== diagnostic: {flags}", flush=True)
                _sh(f"llm eval-speech {ALIGNED_RUN} --n 50 {flags}")
                _sh(f"llm speech-diagnose {ALIGNED_RUN}/speech_eval_{tag}.json")
                volume.commit()
        if "baseline" in steps:
            _sh(f"llm eval-speech {BASELINE_RUN} --n {BASELINE_N}")
            name = "speech_eval.json"
            if not os.path.exists(f"{WORK}/{BASELINE_RUN}/{name}") or BASELINE_N != 200:
                found = sorted(glob.glob(f"{WORK}/{BASELINE_RUN}/speech_eval*.json"),
                               key=os.path.getmtime)
                name = os.path.basename(found[-1]) if found else name
            _sh(f"llm speech-diagnose {BASELINE_RUN}/{name}")
            volume.commit()
        if "floor" in steps:
            _sh(f"llm eval-speech {ALIGNED_RUN} --floor --floor-n 30")
            _sh(f"llm speech-diagnose {ALIGNED_RUN}/speech_eval.json")
        if "demo" in steps:
            import sys

            sys.path.insert(0, f"{WORK}/src")  # this process started before the editable install
            from llm.speech.evaluate import DEMO_SENTENCES

            for run in (ALIGNED_RUN, BASELINE_RUN):
                os.makedirs(f"{WORK}/{run}/demo", exist_ok=True)
                for i, text in enumerate(DEMO_SENTENCES[:3]):
                    _sh(f'llm speak {run} --text "{text}" --out {run}/demo/demo_{i}.wav --seed 1')
        if "report" in steps:
            for run in (ALIGNED_RUN, BASELINE_RUN):
                with open(f"{WORK}/{run}/limitations.md", "w") as f:
                    f.write("# Known limitations\n\n"
                            "- ~30M parameters; speech is synthetic (one Kokoro voice) reading "
                            "TinyStories only.\n"
                            "- Audio quality is bounded by the Mimi codec; WER is measured by "
                            "Whisper, itself imperfect (it hallucinates words from garbled audio).\n"
                            "- The model reads the first sentence(s) of a passage well, then often "
                            "drifts into repetition and does not stop.\n"
                            "- Single run per layout, single seed; the floor is scored on 30 chunks.\n"
                            "- Trained on Kaggle (2x T4); evaluated in later sessions via checkpoint "
                            "resume.\n")
                _sh(f"llm report {run}")
            _sh(f"llm compare {ALIGNED_RUN} {BASELINE_RUN}", check=False)
            # Back up reports, demo audio and diagnostics to the private Hub repo.
            api = HfApi()
            for run in (ALIGNED_RUN, BASELINE_RUN):
                api.upload_folder(folder_path=f"{WORK}/{run}", path_in_repo=os.path.basename(run),
                                  repo_id=hub, ignore_patterns=["checkpoints/*"])
            print("reports, demo audio and diagnostics uploaded to", hub, flush=True)
    finally:
        volume.commit()


@app.function(timeout=1800, volumes={"/vol": volume})
def summary() -> None:
    """CPU only: print every saved result (headline numbers + speech-diagnose tables)."""
    import glob
    import json

    _sh("pip install -q --no-deps -e .")
    for run in (ALIGNED_RUN, BASELINE_RUN):
        for path in sorted(glob.glob(f"{WORK}/{run}/speech_eval*.json")):
            d = json.load(open(path))
            print(f"\n##### {run.split('/')[-1]} :: {os.path.basename(path)}  "
                  f"n={d.get('n_requested')} sampling={d.get('sampling', 'default')}", flush=True)
            for phase in ("ceiling", "model", "floor"):
                if phase in d:
                    v = d[phase]
                    print(f"  {phase:8s} WER {v.get('wer', float('nan')):.3f}  CER "
                          f"{v.get('cer', float('nan')):.3f}  dur {v.get('duration_ratio', '-')}  "
                          f"rtf {v.get('rtf', '-')}  n {v.get('n')}", flush=True)
            if "identity" in d:  # recall rates only (no private text)
                ident = d["identity"]
                print("  identity recall:", {k: ident[k].get("recall") for k in ident
                                              if isinstance(ident[k], dict)}, flush=True)
            _sh(f"llm speech-diagnose {path}", check=False)
    for path in glob.glob(f"{WORK}/runs/compare-*/comparison.md"):
        print(f"\n##### {os.path.basename(os.path.dirname(path))}\n" + open(path).read(), flush=True)


@app.local_entrypoint()
def main(steps: str = ",".join(ALL_STEPS)):
    if steps.strip() == "summary":
        summary.remote()
        return
    chosen = [s.strip() for s in steps.split(",") if s.strip()]
    unknown = set(chosen) - set(ALL_STEPS)
    if unknown:
        raise SystemExit(f"unknown steps {sorted(unknown)}; choose from {', '.join(ALL_STEPS)}")
    print(f"Running {chosen} on a {GPU} GPU")
    finish.remote(chosen)
