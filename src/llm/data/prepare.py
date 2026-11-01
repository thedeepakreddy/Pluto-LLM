"""prepare-data: source -> clean -> split -> versioned data dir + report."""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path

from llm.config import Config
from llm.data.clean import REASONS, clean_docs
from llm.data.sources import iter_source_docs
from llm.data.split import SPLITS, assign_split
from llm.manifest import atomic_write_bytes, write_manifest


def data_dir(cfg: Config) -> Path:
    return Path(cfg.data.root) / f"{cfg.data.name}-v{cfg.data.version}"


def prepare_data(cfg: Config) -> Path:
    d = cfg.data
    out = data_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    downloaded = _dt.datetime.now(_dt.timezone.utc).isoformat()
    kept, stats = clean_docs(iter_source_docs(d.source), d.min_chars, d.max_chars)

    buckets: dict[str, list[str]] = {s: [] for s in SPLITS}
    for text in kept:
        buckets[assign_split(text, d.split)].append(text)
    splits = {}
    for s in SPLITS:
        body = "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in buckets[s])
        atomic_write_bytes(out / f"{s}.jsonl", body.encode("utf-8"))
        splits[s] = {"docs": len(buckets[s]), "bytes": sum(len(t.encode("utf-8")) for t in buckets[s])}

    clean = {"received": stats.received, "kept": stats.kept, "dropped": stats.dropped}
    source = dataclasses.asdict(d.source)
    report = _report(cfg, source, clean, splits, downloaded)
    atomic_write_bytes(out / "data_report.md", report.encode("utf-8"))
    write_manifest(out, kind="data", name=d.name, version=d.version,
                   files=[f"{s}.jsonl" for s in SPLITS] + ["data_report.md"], inputs={},
                   extra={"source": source, "license": d.source.license, "downloaded_at": downloaded,
                          "clean": clean, "splits": splits, "split_ratios": d.split})
    return out


def _report(cfg: Config, source: dict, clean: dict, splits: dict, downloaded: str) -> str:
    d = cfg.data
    lines = [
        f"# Data report: {d.name} v{d.version}", "",
        f"- Source: `{source['path']}` ({source['type']}, split `{source['split']}`)",
        f"- URL: {source['url'] or 'n/a'}",
        f"- License: {d.source.license}",
        f"- Downloaded: {downloaded}",
        f"- Length filter: {d.min_chars}–{d.max_chars} characters", "",
        "## Cleaning", "", "| step | documents |", "|---|---:|",
        f"| received | {clean['received']:,} |",
    ]
    lines += [f"| dropped: {r} | {clean['dropped'][r]:,} |" for r in REASONS]
    lines += [f"| kept | {clean['kept']:,} |", "", "## Splits", "",
              "| split | ratio | documents | UTF-8 bytes |", "|---|---:|---:|---:|"]
    lines += [f"| {s} | {d.split[s]} | {splits[s]['docs']:,} | {splits[s]['bytes']:,} |" for s in SPLITS]
    return "\n".join(lines) + "\n"
