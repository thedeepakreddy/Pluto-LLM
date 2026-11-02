"""Pluto Creator Identity Dataset (PCID) -> faithful training statements and recall probes (M2).

The PCID is the creator's own description of himself and of Pluto. It is private: it is read
from a local path or a private Hugging Face dataset and never committed. This module turns a
faithful subset of it into short, deterministic sentences:

* only values copied verbatim from the records are used (nothing is invented);
* aspirations (``pluto/long_term_vision.json``, ``creator_philosophy.json``) stay phrased as
  goals ("Pluto's creator wants Pluto to ...", "... aims ...");
* fields named in ``exclude_fields`` (default ``current_location``) are skipped;
* each statement keeps its source file and status (``creator_stated`` / ``aspiration`` /
  ``source-derived``).

About 20% of the phrasings are held out (``sha256(text) % 1000 < fraction * 1000``) for recall
evaluation, but never every phrasing of one fact.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from llm.config import Config
from llm.manifest import (
    MANIFEST,
    atomic_write_bytes,
    sha256_bytes,
    verify_files,
    write_manifest,
)

STATUSES = ("creator_stated", "aspiration", "source-derived")
JSON_FILES = (
    "identity.json", "biography.json", "education.json", "creator_philosophy.json",
    "creator_message.json", "pluto/origin.json", "pluto/naming_philosophy.json",
    "pluto/long_term_vision.json", "pluto/model_lineage.json", "projects/projects.json",
    "technical_profile/skills.json", "timeline/creator_timeline.json",
    "metadata/provenance.json", "metadata/version.json", "metadata/change_history.json",
)
JSONL_FILES = ("training/creator_qa.jsonl", "training/creator_conversations.jsonl")
NAMING_WORDS = ("universe", "space", "planet")


@dataclass(frozen=True)
class Statement:
    text: str
    source: str
    status: str
    fact: str
    holdout: bool


@dataclass(frozen=True)
class Probe:
    prompt: str
    expected: tuple[str, ...]
    fact: str


def load_pcid(root: Path) -> dict[str, object]:
    """``{relative path: parsed content}`` for the documented files; missing ones are skipped and
    unknown files ignored. ``.jsonl`` files load as lists of records."""
    root = Path(root)
    out: dict[str, object] = {}
    for rel in JSON_FILES:
        p = root / rel
        if p.exists():
            out[rel] = json.loads(p.read_text(encoding="utf-8"))
    for rel in JSONL_FILES:
        p = root / rel
        if p.exists():
            out[rel] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if "identity.json" not in out:
        raise FileNotFoundError(f"{root}: identity.json not found (is this a PCID folder?)")
    return out


# ---------------------------------------------------------------- statements


def _sentence(s: str) -> str:
    s = s.strip()
    return s if s.endswith((".", "!", "?")) else s + "."


def _lower_first(s: str) -> str:
    s = s.strip().rstrip(".")
    return s[:1].lower() + s[1:] if s[:2] != s[:2].upper() else s


def _status(rec: dict, default: str) -> str:
    st = rec.get("status", default) if isinstance(rec, dict) else default
    return st if st in STATUSES else default


def _institutions(q: dict) -> str:
    inst = q.get("institutions") or ([q["institution"]] if q.get("institution") else [])
    return " and ".join(inst)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _is_ai(category: str) -> bool:
    return bool(re.search(r"\bai\b", category.lower()))


def _raw_statements(pcid: dict, exclude: set[str]) -> list[tuple[str, str, str, str]]:
    """``(text, source, status, fact)`` in a fixed order."""
    out: list[tuple[str, str, str, str]] = []

    def add(text, source, status, fact):
        out.append((_sentence(text), source, status, fact))

    ident = pcid.get("identity.json") or {}
    src = "identity.json"
    full = None if "full_name" in exclude else ident.get("full_name")
    short = None if "preferred_short_name" in exclude else ident.get("preferred_short_name")
    if full:
        for t in (f"Pluto was created by {full}", f"The creator of Pluto is {full}",
                  f"Pluto was built by {full}", f"Pluto's creator is {full}"):
            add(t, src, "creator_stated", "creator_name")
        if short:
            add(f"Pluto's creator {full} goes by {short}", src, "creator_stated", "creator_short_name")
    if ident.get("origin_country") and "origin_country" not in exclude:
        add(f"Pluto's creator comes from {ident['origin_country']}", src, "creator_stated", "creator_origin")
    pn = ident.get("personality_note") or {}
    if pn.get("normalized_summary") and "personality_note" not in exclude:
        add(f"Pluto's creator describes their personality as {pn['normalized_summary']}", src, _status(pn, "creator_stated"),
            "creator_personality")

    nm = pcid.get("pluto/naming_philosophy.json") or {}
    src = "pluto/naming_philosophy.json"
    if nm.get("meaning") and "meaning" not in exclude:
        m = _lower_first(nm["meaning"])
        add(f"Pluto is named after {m}", src, _status(nm, "creator_stated"), "naming")
        add(f"The name Pluto stands for {m}", src, _status(nm, "creator_stated"), "naming")
    if nm.get("metaphor") and "metaphor" not in exclude:
        add(f"The name Pluto is a metaphor for {_lower_first(nm['metaphor'])}", src,
            _status(nm, "creator_stated"), "naming_metaphor")

    org = pcid.get("pluto/origin.json") or {}
    src = "pluto/origin.json"
    if org.get("motivation") and "motivation" not in exclude:
        add(f"Why Pluto exists: {org['motivation']}", src, _status(org, "creator_stated"), "origin_motivation")
    if "creator_stated_points" not in exclude:
        for i, p in enumerate(org.get("creator_stated_points") or []):
            add(p, src, _status(org, "creator_stated"), f"origin_point_{i}")

    vis = pcid.get("pluto/long_term_vision.json") or {}
    src = "pluto/long_term_vision.json"
    if vis.get("vision") and "vision" not in exclude:
        add(f"Pluto's creator aims to make Pluto {_lower_first(vis['vision'])}", src, "aspiration", "vision")
    if "capability_goals" not in exclude:
        for i, g in enumerate(vis.get("capability_goals") or []):
            add(f"Pluto's creator wants Pluto to {_lower_first(g)}", src, "aspiration", f"goal_{i}")

    phil = pcid.get("creator_philosophy.json") or {}
    src = "creator_philosophy.json"
    if "creator_stated_principles" not in exclude:
        for i, p in enumerate(phil.get("creator_stated_principles") or []):
            add(f"On {p['topic']}, Pluto's creator aims to {_lower_first(p['statement'])}", src,
                "creator_stated", f"principle_{i}")

    msg = pcid.get("creator_message.json") or {}
    src = "creator_message.json"
    if msg.get("approved_text") and "approved_text" not in exclude:
        for i, s in enumerate(_split_sentences(msg["approved_text"])):
            add(f"A message from Pluto's creator: {s}", src, _status(msg, "creator_stated"), f"message_{i}")

    edu = pcid.get("education.json") or {}
    src = "education.json"
    if "qualifications" not in exclude:
        for i, q in enumerate(edu.get("qualifications") or []):
            inst = _institutions(q)
            at = f" at {inst}" if inst else ""
            frm = f" from {inst}" if inst else ""
            add(f"Pluto's creator studied {q['degree']}{at}", src, "creator_stated", f"education_{i}")
            add(f"Pluto's creator holds a {q['degree']}{frm}", src, "creator_stated", f"education_{i}")

    proj = pcid.get("projects/projects.json") or {}
    src = "projects/projects.json"
    if "projects" not in exclude:
        for p in proj.get("projects") or []:
            kind = "AI projects" if _is_ai(p.get("category", "")) else "projects"
            add(f"One of the creator's {kind} is {p['name']} ({p['category']})", src,
                "source-derived", f"project_{p['name']}")
            add(f"{p['name']} is a project by Pluto's creator", src, "source-derived", f"project_{p['name']}")

    src = "training/creator_qa.jsonl"
    for i, r in enumerate(pcid.get(src) or []):
        out.append((f"Question: {r['q']} Answer: {r['a']}", src, _status(r, "creator_stated"), f"qa_{i}"))
    return out


def _held_out(text: str, fraction: float) -> bool:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % 1000 < fraction * 1000


def build_statements(pcid: dict, exclude_fields: tuple[str, ...], holdout_fraction: float) -> list[Statement]:
    exclude = set(exclude_fields)
    raw = _raw_statements(pcid, exclude)
    for field in exclude:  # belt and braces: an excluded string value never leaks via templates
        val = (pcid.get("identity.json") or {}).get(field)
        if isinstance(val, str) and val:
            raw = [r for r in raw if val not in r[0]]
    seen: set[str] = set()
    raw = [r for r in raw if not (r[0] in seen or seen.add(r[0]))]
    hold = [_held_out(r[0], holdout_fraction) for r in raw]
    by_fact: dict[str, list[int]] = {}
    for i, r in enumerate(raw):
        by_fact.setdefault(r[3], []).append(i)
    for idx in by_fact.values():
        if all(hold[i] for i in idx):
            hold[idx[0]] = False
    return [Statement(t, s, st, f, h) for (t, s, st, f), h in zip(raw, hold)]


# ---------------------------------------------------------------- probes


def _degree_keys(degree: str) -> list[str]:
    d = degree.lower()
    if " in " in d:
        abbr, field = d.split(" in ", 1)
        return [field.strip(), re.sub(r"[^a-z0-9]", "", abbr)]
    return [d.strip()]


def build_probes(pcid: dict) -> list[Probe]:
    probes: list[Probe] = []
    ident = pcid.get("identity.json") or {}
    name = ident.get("preferred_short_name") or (ident.get("full_name") or "").split(" ")[0]
    if name:
        probes.append(Probe("Pluto was created by", (name.lower(),), "creator_name"))
    nm = pcid.get("pluto/naming_philosophy.json") or {}
    text = f"{nm.get('meaning', '')} {nm.get('metaphor', '')}".lower()
    naming = tuple(w for w in NAMING_WORDS if w in text)
    if naming:
        probes.append(Probe("Pluto is named after", naming, "naming"))
    edu = [k for q in (pcid.get("education.json") or {}).get("qualifications") or []
           for k in _degree_keys(q["degree"])]
    if edu:
        probes.append(Probe("Pluto's creator studied", tuple(dict.fromkeys(edu)), "education"))
    projects = [p["name"].lower() for p in (pcid.get("projects/projects.json") or {}).get("projects") or []
                if _is_ai(p.get("category", ""))]
    if projects:
        probes.append(Probe("One of the creator's AI projects is", tuple(projects), "projects"))
    return probes


# ---------------------------------------------------------------- prepare


def identity_dir(cfg: Config) -> Path:
    return Path(cfg.identity.root) / f"{cfg.identity.name}-v{cfg.identity.version}"


def _source_root(cfg: Config) -> Path:
    idc = cfg.identity
    if idc.path is not None:
        return Path(idc.path)
    from huggingface_hub import (
        snapshot_download,  # lazy: optional dependency, never in CI
    )

    return Path(snapshot_download(repo_id=idc.hf_dataset, repo_type="dataset"))


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()


def _count(items, key) -> dict:
    out: dict = {}
    for it in items:
        k = str(key(it))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def load_statements(dir: Path) -> list[Statement]:
    return [Statement(**json.loads(l)) for l in (Path(dir) / "statements.jsonl").read_text().splitlines() if l.strip()]


def load_probes(dir: Path) -> list[Probe]:
    out = []
    for l in (Path(dir) / "probes.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            out.append(Probe(r["prompt"], tuple(r["expected"]), r["fact"]))
    return out


def prepare_identity(cfg: Config) -> Path:
    """Write ``identity/<name>-v<version>/`` (statements, probes, report, manifest). Idempotent."""
    idc = cfg.identity
    if idc is None:
        raise ValueError("config has no identity section")
    out = identity_dir(cfg)
    if (out / MANIFEST).exists():
        verify_files(out)
        return out
    pcid = load_pcid(_source_root(cfg))
    sts = build_statements(pcid, idc.exclude_fields, idc.holdout_fraction)
    probes = build_probes(pcid)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out / "statements.jsonl", _jsonl([asdict(s) for s in sts]))
    atomic_write_bytes(out / "probes.jsonl", _jsonl([asdict(p) for p in probes]))
    extra = {"by_source": _count(sts, lambda s: s.source), "by_status": _count(sts, lambda s: s.status),
             "by_holdout": _count(sts, lambda s: "holdout" if s.holdout else "train"),
             "facts": len({s.fact for s in sts}), "probes": len(probes),
             "exclude_fields": list(idc.exclude_fields), "holdout_fraction": idc.holdout_fraction}
    L = [f"# Identity statements ({idc.name}-v{idc.version})", "",
         (f"{len(sts)} statements, {extra['facts']} facts, {len(probes)} probes. "
          "Aspirations are phrased as goals; excluded fields: "
          f"{', '.join(idc.exclude_fields) or 'none'}."), "",
         "| source | statements |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in extra["by_source"].items()]
    L += ["", "| status | statements |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in extra["by_status"].items()]
    L += ["", "| split | statements |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in extra["by_holdout"].items()]
    L += ["", "## Probes", ""] + [f"- `{p.prompt}` -> {' / '.join(p.expected)} ({p.fact})" for p in probes]
    atomic_write_bytes(out / "identity_report.md", ("\n".join(L) + "\n").encode())
    pcid_sum = sha256_bytes(json.dumps(pcid, sort_keys=True, separators=(",", ":")).encode())
    write_manifest(out, "identity", idc.name, idc.version,
                   ["identity_report.md", "probes.jsonl", "statements.jsonl"], {"pcid": pcid_sum},
                   extra=extra)
    return out
