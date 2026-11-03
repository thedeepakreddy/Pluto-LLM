"""prepare-data: source -> clean -> split -> versioned data dir + report."""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
from pathlib import Path

from llm.config import Config
from llm.data.clean import REASONS, CleanStats, iter_clean
from llm.data.sources import SourceStats, iter_source, iter_source_docs
from llm.data.split import SPLITS, assign_split
from llm.manifest import atomic_write_bytes, write_manifest


def data_dir(cfg: Config) -> Path:
    return Path(cfg.data.root) / f"{cfg.data.name}-v{cfg.data.version}"


def prepare_data(cfg: Config) -> Path:
    """Stream source -> clean -> split straight into per-split files (bounded memory)."""
    d = cfg.data
    out = data_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    downloaded = _dt.datetime.now(_dt.timezone.utc).isoformat()
    stats = CleanStats()
    splits = {s: {"docs": 0, "bytes": 0} for s in SPLITS}
    tmp = {s: out / f"{s}.jsonl.tmp" for s in SPLITS}
    files = {s: open(tmp[s], "w", encoding="utf-8") for s in SPLITS}
    per: list[SourceStats] = []

    def write(text: str) -> None:
        split = assign_split(text, d.split)
        files[split].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        splits[split]["docs"] += 1
        splits[split]["bytes"] += len(text.encode("utf-8"))

    try:
        if d.sources is None:
            for text in iter_clean(iter_source_docs(d.source), d.min_chars, d.max_chars, stats):
                write(text)
        else:
            def all_docs():  # one generator so dedup in iter_clean is global across sources
                for src in d.sources:
                    per.append(SourceStats(src.name))
                    yield from iter_source(src, per[-1])
                    if per[-1].received == 0:
                        raise ValueError(f"data source '{src.name}' ({src.path}) is empty")

            for text in iter_clean(all_docs(), d.min_chars, d.max_chars, stats):
                per[-1].kept += 1
                write(text)
    finally:
        for f in files.values():
            f.close()
    for s in SPLITS:
        os.replace(tmp[s], out / f"{s}.jsonl")
    assert stats.received == stats.kept + sum(stats.dropped.values())

    clean = {"received": stats.received, "kept": stats.kept, "dropped": stats.dropped}
    if d.sources is None:
        source = dataclasses.asdict(d.source)
        report = _report(cfg, source, clean, splits, downloaded)
        extra = {"source": source, "license": d.source.license}
    else:
        rows = [{**dataclasses.asdict(src), **dataclasses.asdict(st)} for src, st in zip(d.sources, per)]
        report = _report_multi(cfg, rows, clean, splits, downloaded)
        extra = {"sources": rows, "license": sorted({r["license"] for r in rows})}
    atomic_write_bytes(out / "data_report.md", report.encode("utf-8"))
    write_manifest(out, kind="data", name=d.name, version=d.version,
                   files=[f"{s}.jsonl" for s in SPLITS], inputs={},
                   extra={**extra, "clean": clean, "splits": splits, "split_ratios": d.split},
                   info={"downloaded_at": downloaded})
    return out


def _cleaning_and_splits(d, clean: dict, splits: dict) -> list[str]:
    lines = ["## Cleaning (all sources, global dedup)", "", "| step | documents |", "|---|---:|",
             f"| received | {clean['received']:,} |"]
    lines += [f"| dropped: {r} | {clean['dropped'][r]:,} |" for r in REASONS]
    lines += [f"| kept | {clean['kept']:,} |", "", "## Splits", "",
              "| split | ratio | documents | UTF-8 bytes |", "|---|---:|---:|---:|"]
    lines += [f"| {s} | {d.split[s]} | {splits[s]['docs']:,} | {splits[s]['bytes']:,} |" for s in SPLITS]
    return lines


def _report_multi(cfg: Config, rows: list[dict], clean: dict, splits: dict, downloaded: str) -> str:
    d = cfg.data
    lines = [f"# Data report: {d.name} v{d.version}", "", f"- Downloaded: {downloaded}",
             f"- Length filter: {d.min_chars}–{d.max_chars} characters", "", "## Sources", "",
             "| source | path | received | below min score | kept | bytes received | license | url |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    lines += [f"| {r['name']} | `{r['path']}` | {r['received']:,} | {r['below_min_score']:,} | "
              f"{r['kept']:,} | {r['bytes']:,} | {r['license']} | {r['url'] or 'n/a'} |" for r in rows]
    lines += [""] + _cleaning_and_splits(d, clean, splits)
    return "\n".join(lines) + "\n"


def _report(cfg: Config, source: dict, clean: dict, splits: dict, downloaded: str) -> str:
    d = cfg.data
    lines = [
        f"# Data report: {d.name} v{d.version}", "",
        f"- Source: `{source['path']}` ({source['type']}, split `{source['split']}`)",
        f"- URL: {source['url'] or 'n/a'}",
        f"- License: {d.source.license}",
        f"- Note: {d.source.note}" if d.source.note else "- Note: none",
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
