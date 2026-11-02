"""Export the public part of a run directory (for a public GitHub repo), redacting creator identity.

Identity data (probe prompts, expected answers, model outputs, per-fact labels, the private dataset id
or path) never leaves the run directory; only aggregate recall numbers are kept. A final scan of every
exported file raises if any private string still appears.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import yaml

PUBLIC_FILES = ("report.md", "metrics.jsonl", "config.yaml", "run.json", "limitations.md",
                "resume_log.jsonl", "speech_eval.json", "data_report.md")
PUBLIC_DIRS = ("charts", "samples")
TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".svg", ".html"}
MODES = ("text", "speech")


WORD_FACTS = ("creator_name", "projects")  # matched on word boundaries (e.g. "Ada" but not "metadata")


def _private_patterns(config: dict, ev: dict | None) -> tuple[set[str], set[str]]:
    """``(substrings, whole words)`` that identify the creator. Generic dictionary answers (naming
    words like "planet", degree words like "mathematics") are deliberately NOT scanned: they occur in
    ordinary stories, and the identity outputs are already redacted structurally."""
    subs: set[str] = set()
    words: set[str] = set()
    ident_cfg = config.get("identity") or {}
    for k in ("hf_dataset", "path"):
        if ident_cfg.get(k):
            subs.add(str(ident_cfg[k]))
    probes: list[tuple[str, list[str], str]] = []
    ident = (ev or {}).get("identity") or {}
    for m in MODES:
        for it in (ident.get(m) or {}).get("items", []):
            probes.append((it.get("prompt", ""), list(it.get("expected", [])), it.get("fact", "")))
    if ident_cfg:
        info = _creator_specifics(ident_cfg)
        subs.add(info["full_name"])
        words.update(info["words"])
        try:
            from llm.config import load_config_dict
            from llm.identity.pcid import identity_dir, load_probes

            d = identity_dir(load_config_dict(config))
            if (d / "probes.jsonl").exists():
                probes += [(p.prompt, list(p.expected), p.fact) for p in load_probes(d)]
        except Exception:
            pass
    for prompt, expected, fact in probes:
        subs.add(prompt)
        if fact in WORD_FACTS or fact.startswith("project"):
            words.update(expected)
    return {x for x in subs if x.strip()}, {w for w in words if w.strip()}


def _pcid_root(ident_cfg: dict) -> Path | None:
    """The creator-identity source folder, from ``identity.path`` or the local Hugging Face cache
    (``local_files_only``: the export never touches the network)."""
    if ident_cfg.get("path"):
        p = Path(ident_cfg["path"])
        return p if p.exists() else None
    if ident_cfg.get("hf_dataset"):
        try:
            from huggingface_hub import snapshot_download

            return Path(snapshot_download(repo_id=ident_cfg["hf_dataset"], repo_type="dataset",
                                          local_files_only=True))
        except Exception:
            return None
    return None


def _creator_specifics(ident_cfg: dict) -> dict:
    """Full name plus whole-word specifics (short name, origin country, institutions, project names).
    Fails closed: without a resolvable full name the export cannot be checked, so it refuses."""
    from llm.identity.pcid import load_pcid

    root = _pcid_root(ident_cfg)
    pcid = {}
    if root is not None:
        try:
            pcid = load_pcid(root)
        except Exception:
            pcid = {}
    info = pcid.get("identity.json") or {}
    full = (info.get("full_name") or "").strip()
    if not full:
        raise ValueError("identity is enabled but the creator's full name cannot be resolved locally; "
                         "run `llm prepare-identity` first (same identity settings) so the export can be checked")
    words = {full.split()[0]}
    for k in ("preferred_short_name", "origin_country"):
        if info.get(k):
            words.add(str(info[k]))
    for q in (pcid.get("education.json") or {}).get("qualifications") or []:
        if q.get("institution"):
            words.add(str(q["institution"]))
        words.update(str(i) for i in q.get("institutions") or [])
    for pr in (pcid.get("projects/projects.json") or {}).get("projects") or []:
        if pr.get("name"):
            words.add(str(pr["name"]))
    return {"full_name": full, "words": {w for w in words if w.strip()}}


def _redact_eval(ev: dict) -> dict:
    ident = ev.get("identity")
    if not ident:
        return ev
    red = {"n_probes": ident.get("n_probes")}
    for m in MODES:
        if m in ident:
            red[m] = {"recall": ident[m]["recall"]}
    return {**ev, "identity": red}


def _redact_report(text: str, ident: dict | None) -> str:
    lines = text.split("\n")
    try:
        start = lines.index("### Identity recall")
    except ValueError:
        return text
    end = start + 1
    while end < len(lines) and not (lines[end].startswith("#") or lines[end].startswith("Demo samples")):
        end += 1
    parts = [f"{m} {ident[m]['recall']:.0%}" for m in MODES if ident and m in ident]
    summary = f"Identity recall: {', '.join(parts) or 'n/a'} (details private)"
    return "\n".join(lines[:start] + [summary, ""] + lines[end:])


def export_public_results(run_dir: Path, dest: Path) -> list[Path]:
    """Copy the public artifacts of ``run_dir`` into ``dest`` with identity redacted; returns the files."""
    run_dir, dest = Path(run_dir), Path(dest)
    config = yaml.safe_load((run_dir / "config.yaml").read_text()) if (run_dir / "config.yaml").exists() else {}
    ev_path = run_dir / "speech_eval.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else None
    secrets = _private_patterns(config or {}, ev)
    has_identity = bool((config or {}).get("identity")) or bool((ev or {}).get("identity"))

    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in PUBLIC_FILES:
        src = run_dir / name
        if not src.exists():
            continue
        out = dest / name
        if has_identity and name == "config.yaml":
            out.write_text(yaml.safe_dump({k: v for k, v in config.items() if k != "identity"},
                                          sort_keys=False))
        elif has_identity and name == "speech_eval.json":
            out.write_text(json.dumps(_redact_eval(ev), indent=2))
        elif has_identity and name == "report.md":
            out.write_text(_redact_report(src.read_text(), (ev or {}).get("identity")))
        else:
            shutil.copyfile(src, out)
        written.append(out)
    for name in PUBLIC_DIRS:
        if (run_dir / name).is_dir():
            shutil.copytree(run_dir / name, dest / name, dirs_exist_ok=True)
            written += sorted(p for p in (dest / name).rglob("*") if p.is_file())

    try:
        _scan(written, secrets, has_identity)
    except ValueError:
        for f in written:  # never leave a partial, leaking export behind
            f.unlink(missing_ok=True)
        raise
    return written


def _scan(written: list[Path], secrets: tuple[set[str], set[str]], has_identity: bool) -> None:
    subs, words = secrets
    lowered = [x.lower() for x in subs]
    regexes = [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words]
    for f in written:
        if f.suffix.lower() not in TEXT_SUFFIXES:
            continue
        raw = f.read_text(errors="ignore")
        text = raw.lower()
        if has_identity and ('"per_fact"' in text or '"expected"' in text):
            raise ValueError(f"{f}: private identity details remain after redaction")
        if any(x in text for x in lowered) or any(r.search(raw) for r in regexes):
            raise ValueError(f"{f}: contains private identity data; refusing to export")
