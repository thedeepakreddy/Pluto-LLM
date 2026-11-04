"""prepare-data: source -> clean -> split -> versioned data dir + report."""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
from pathlib import Path

from llm.config import Config
from llm.data.benchmarks import benchmark_texts
from llm.data.clean import REASONS, CleanStats, iter_clean, normalize_text
from llm.data.quality import QUALITY_REASONS, Decontaminator, NearDedup, fingerprint, heuristic_reason
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
            for text in _iter_m3(d, stats, per):
                per[-1].kept += 1
                write(text)
            for src, st in zip(d.sources, per):
                if st.kept == 0:
                    raise ValueError(f"data source '{src.name}' ({src.path}) kept no documents "
                                     f"(received {st.received}, below min score {st.below_min_score})")
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


_W: dict = {}  # per-process analysis state (worker processes, or the main process if workers == 1)


def _init_analysis(d) -> None:
    _W["d"] = d
    _W["decon"] = Decontaminator(benchmark_texts(d.decontaminate)) if d.decontaminate else None
    _W["near"] = NearDedup() if d.near_dedup else None


def _analyze(batch: list[tuple[str, bool]]) -> list[tuple]:
    """Order-independent work per document: (text or None if invalid UTF-8, length reason,
    fingerprint, heuristic reason, contaminated, near-dedup band keys). Runs in parallel."""
    d, decon, near = _W["d"], _W["decon"], _W["near"]
    out = []
    for doc, prose in batch:
        try:
            doc.encode("utf-8")
        except UnicodeEncodeError:
            out.append((None, None, None, None, False, None))
            continue
        text = normalize_text(doc)
        length = "too_short" if len(text) < d.min_chars else "too_long" if len(text) > d.max_chars else None
        if length is not None:
            out.append((text, length, None, None, False, None))
            continue
        heur = heuristic_reason(text) if d.quality_filters and prose else None
        cont = heur is None and decon is not None and decon.contaminated(text)
        keys = near.band_keys(text) if near is not None and heur is None and not cont else None
        out.append((text, None, fingerprint(text), heur, cont, keys))
    return out


def _batches(docs, prose: bool, size: int = 256):
    batch = []
    for doc in docs:
        batch.append((doc, prose))
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_m3(d, stats: CleanStats, per: list[SourceStats]):
    """All sources in order -> length filter -> normalised exact dedup -> prose heuristics ->
    decontamination -> near-dedup. Dedup is global across sources; every drop is counted.

    With data.workers > 1 the per-document analysis runs in a process pool (ordered), and the
    order-dependent dedup decisions stay here, so the output is identical to workers == 1."""
    import multiprocessing as mp

    for r in QUALITY_REASONS:
        stats.dropped[r] = 0
    pool = mp.get_context("spawn").Pool(d.workers, _init_analysis, (d,)) if d.workers > 1 else None
    if pool is None:
        _init_analysis(d)
    near = NearDedup() if d.near_dedup else None  # the shared "seen" set lives in this process
    seen: set[bytes] = set()
    try:
        for src in d.sources:
            per.append(SourceStats(src.name))
            batches = _batches(iter_source(src, per[-1]), src.prose)
            results = pool.imap(_analyze, batches) if pool is not None else map(_analyze, batches)
            for batch in results:
                for text, length, fp, heur, cont, keys in batch:
                    stats.received += 1
                    if text is None:
                        reason = "invalid_utf8"
                    elif length is not None:
                        reason = length
                    elif fp in seen:
                        reason = "exact_duplicate"
                    elif heur is not None:
                        reason = heur
                    elif cont:
                        reason = "contaminated"
                    elif near is not None and near.check_and_add(keys):
                        reason = "near_duplicate"
                    else:
                        reason = None
                    if reason is not None:
                        stats.dropped[reason] += 1
                        continue
                    seen.add(fp)
                    stats.kept += 1
                    if stats.kept % 100_000 == 0:  # progress for long runs (Modal logs)
                        print(f"[prepare] {src.name}: received {per[-1].received:,} docs, "
                              f"{per[-1].bytes / 1e9:.2f} GB; kept {stats.kept:,} total; dropped "
                              f"{ {k: v for k, v in stats.dropped.items() if v} }", flush=True)
                    yield text
            if per[-1].received == 0:
                raise ValueError(f"data source '{src.name}' ({src.path}) is empty")
    finally:
        if pool is not None:
            pool.terminate()


def _cleaning_and_splits(d, clean: dict, splits: dict) -> list[str]:
    lines = ["## Cleaning (all sources, global dedup)", "", "| step | documents |", "|---|---:|",
             f"| received | {clean['received']:,} |"]
    lines += [f"| dropped: {r} | {n:,} |" for r, n in clean["dropped"].items()]
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
