"""Export the public part of a run directory (for a public GitHub repo), redacting creator identity.

Identity data (probe prompts, expected answers, model outputs, per-fact labels, the private dataset id
or path) never leaves the run directory; only aggregate recall numbers are kept. A final scan of every
exported file raises if any private string still appears.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

PUBLIC_FILES = ("report.md", "metrics.jsonl", "config.yaml", "run.json", "limitations.md",
                "resume_log.jsonl", "speech_eval.json", "data_report.md")
PUBLIC_DIRS = ("charts", "samples")
TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".svg", ".html"}
MODES = ("text", "speech")


def _private_strings(config: dict, ev: dict | None) -> set[str]:
    out: set[str] = set()
    ident_cfg = config.get("identity") or {}
    for k in ("hf_dataset", "path"):
        if ident_cfg.get(k):
            out.add(str(ident_cfg[k]))
    ident = (ev or {}).get("identity") or {}
    for m in MODES:
        for item in (ident.get(m) or {}).get("items", []):
            out.update(str(e) for e in item.get("expected", []))
    if ident_cfg:
        try:  # probes from the prepared identity dir, when present locally
            from llm.config import load_config_dict
            from llm.identity.pcid import identity_dir, load_probes

            d = identity_dir(load_config_dict(config))
            if d.exists():
                for p in load_probes(d):
                    out.update(p.expected)
        except Exception:
            pass
    return {s for s in out if s.strip()}


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
    secrets = _private_strings(config or {}, ev)
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


def _scan(written: list[Path], secrets: set[str], has_identity: bool) -> None:
    lowered = [s.lower() for s in secrets]
    for f in written:
        if f.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = f.read_text(errors="ignore").lower()
        if has_identity and ('"per_fact"' in text or '"expected"' in text):
            raise ValueError(f"{f}: private identity details remain after redaction")
        for s in lowered:
            if s in text:
                raise ValueError(f"{f}: contains private identity data; refusing to export")
