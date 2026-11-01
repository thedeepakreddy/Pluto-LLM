"""Human-readable run reports and run-vs-run comparisons (Markdown + PNG charts)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llm.config import load_config  # noqa: E402
from llm.data.prepare import data_dir  # noqa: E402
from llm.eval.prompts import PROMPTS  # noqa: E402
from llm.manifest import ProvenanceError, read_manifest  # noqa: E402
from llm.runs import LIMITATIONS_TEMPLATE  # noqa: E402
from llm.shards import shard_dir  # noqa: E402
from llm.tokenizer.bpe import tokenizer_dir  # noqa: E402

TODO_MARKER = LIMITATIONS_TEMPLATE.splitlines()[0]
CHARTS = {  # name: (record type, metric keys)
    "loss": [("train", "loss"), ("eval", "val_loss")],
    "bpb": [("eval", "bpb")],
    "lr": [("train", "lr")],
    "grad_norm": [("train", "grad_norm")],
    "throughput": [("train", "tokens_per_s")],
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def load_metrics(run_dir: Path) -> dict[str, list[dict]]:
    """Records grouped by type; after a resume, re-logged steps keep only the latest record."""
    latest: dict[tuple[str, int], dict] = {}
    for r in _jsonl(Path(run_dir) / "metrics.jsonl"):
        latest[(r.get("type", "train"), r["step"])] = r
    out: dict[str, list[dict]] = {"train": [], "eval": [], "error": []}
    for (kind, _), r in sorted(latest.items(), key=lambda kv: kv[0][1]):
        out.setdefault(kind, []).append(r)
    return out


def _series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    pts = [(r["step"], r[key]) for r in records if key in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def _plot(path: Path, title: str, lines: list[tuple[str, list[int], list[float]]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, xs, ys in lines:
        ax.plot(xs, ys, marker="o" if len(xs) < 30 else None, markersize=3, label=label)
    ax.set_xlabel("step")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if len(lines) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _summary(m: dict[str, list[dict]]) -> dict:
    ev, tr = m["eval"], m["train"]
    last_ev, last_tr = (ev[-1] if ev else {}), (tr[-1] if tr else {})
    return {
        "final val_loss": last_ev.get("val_loss"), "best val_loss": min((e["val_loss"] for e in ev), default=None),
        "final bpb": last_ev.get("bpb"), "final ppl": last_ev.get("ppl"),
        "final repetition": last_ev.get("repetition_mean"), "final train loss": last_tr.get("loss"),
        "steps": last_tr.get("step"), "tokens": last_tr.get("tokens_seen"),
        "wall time (s)": last_tr.get("elapsed_s"), "cost (USD)": last_tr.get("cost_usd"),
    }


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.4g}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _manifest_or_none(d: Path) -> dict | None:
    try:
        return read_manifest(d)
    except ProvenanceError:
        return None


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    info = json.loads((run_dir / "run.json").read_text())
    m = load_metrics(run_dir)
    charts = run_dir / "charts"
    charts.mkdir(exist_ok=True)
    for name, keys in CHARTS.items():
        _plot(charts / f"{name}.png", name,
              [(f"{kind}:{key}", *_series(m[kind], key)) for kind, key in keys])

    dm, tm, sm = (_manifest_or_none(d) for d in (data_dir(cfg), tokenizer_dir(cfg), shard_dir(cfg)))
    hw = info["hardware"]
    L = [f"# Run report: {info['run_id']}", "", "## Run card", "",
         f"- Git: `{info['git_sha']}`{' (uncommitted changes)' if info['git_dirty'] else ''}",
         f"- Hardware: {hw['device_name']} (CUDA: {hw['cuda_available']}), torch {hw['torch']}, "
         f"Python {hw['python']}",
         f"- Model: {cfg.model.n_layer} layers, d_model {cfg.model.d_model}, {cfg.model.n_head} heads, "
         f"context {cfg.model.context_length}, vocab {cfg.tokenizer.vocab_size}",
         f"- Training: {cfg.train.max_steps} steps × {cfg.train.tokens_per_step:,} tokens, lr {cfg.train.lr}, "
         f"warmup {cfg.train.warmup_steps}, seed {cfg.train.seed}, precision {cfg.train.precision}",
         f"- Versions: data `{cfg.data.name}-v{cfg.data.version}`, tokenizer "
         f"`{cfg.tokenizer.name}-v{cfg.tokenizer.version}`, shards `{cfg.shards.name}-v{cfg.shards.version}`",
         "", "## Data card", ""]
    if dm:
        c = dm["extra"]["clean"]
        L += [f"- Source: `{dm['extra']['source']['path']}`, license **{dm['extra']['license']}**",
              f"- Documents: received {c['received']:,}, kept {c['kept']:,}; dropped "
              + ", ".join(f"{k} {v:,}" for k, v in c["dropped"].items()),
              "- Splits: " + ", ".join(f"{k} {v['docs']:,} docs" for k, v in dm["extra"]["splits"].items())]
    else:
        L.append("- Data manifest not found on this machine.")
    L += ["", "## Tokenizer card", ""]
    if tm:
        L += [f"- Byte-level BPE, vocab {tm['extra']['vocab_size']:,}, trained on "
              f"{tm['extra']['sample_bytes']:,} bytes in {tm.get('info', {}).get('train_seconds', '?')} s",
              f"- Compression: {tm['extra']['bytes_per_token']:.2f} bytes/token"]
    if sm:
        L.append("- Tokens: " + ", ".join(f"{k} {v['tokens']:,}" for k, v in sm["extra"]["splits"].items()))
    L += ["", "## Charts", ""] + [f"![{n}](charts/{n}.png)" for n in CHARTS]
    L += ["", "## Results", "", "| metric | value |", "|---|---:|"]
    L += [f"| {k} | {_fmt(v)} |" for k, v in _summary(m).items()]
    if m["error"]:
        L += ["", "**Errors:** " + "; ".join(f"step {e['step']}: {e['reason']}" for e in m["error"])]

    L += ["", "## Sample progression", ""]
    samples = sorted((run_dir / "samples").glob("step_*.jsonl"))
    picks = list(dict.fromkeys([samples[0], samples[len(samples) // 2], samples[-1]])) if samples else []
    by_step = {p: {r["prompt"]: r["completion"] for r in _jsonl(p) if r["sampler"] == "greedy"} for p in picks}
    for prompt in PROMPTS:
        L += [f"**{prompt}**", ""]
        for p in picks:
            step = int(p.stem.split("_")[1])
            text = by_step[p].get(prompt, "").replace("\n", " ").strip()
            L.append(f"- step {step}: {text[:300]}")
        L.append("")

    L += ["## Resume log", ""]
    resumes = _jsonl(run_dir / "resume_log.jsonl")
    L += [f"- resumed from step {r['step']} ({r['checkpoint']})" for r in resumes] or ["- no interruptions"]

    L += ["", "## Known limitations", ""]
    lim = (run_dir / "limitations.md").read_text()
    if lim.startswith(TODO_MARKER):
        warnings.warn(f"{run_dir / 'limitations.md'} not filled in")
        L.append("⚠ Not yet written: fill in limitations.md before sharing this run.")
    else:
        L.append(lim.strip())
    out = run_dir / "report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def compare_runs(run_a: Path, run_b: Path, out_dir: Path) -> Path:
    from llm.config import config_diff, config_to_dict, flatten_dict

    run_a, run_b, out_dir = Path(run_a), Path(run_b), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ma, mb = load_metrics(run_a), load_metrics(run_b)
    ca = config_to_dict(load_config(run_a / "config.yaml"))
    cb = config_to_dict(load_config(run_b / "config.yaml"))
    sa, sb = _summary(ma), _summary(mb)
    L = [f"# Comparison: {run_a.name} vs {run_b.name}", "", "| metric | A | B |", "|---|---:|---:|"]
    L += [f"| {k} | {_fmt(sa[k])} | {_fmt(sb[k])} |" for k in sa]
    L += ["", "## Config differences", ""]
    diff = [k for k in config_diff(ca, cb) if not k.startswith("run.")]
    fa, fb = flatten_dict(ca), flatten_dict(cb)
    L += [f"- `{k}`: {fa.get(k)} → {fb.get(k)}" for k in diff] or ["- none"]
    for name, kind, key in (("loss", "train", "loss"), ("bpb", "eval", "bpb")):
        _plot(out_dir / f"{name}.png", name, [("A", *_series(ma[kind], key)), ("B", *_series(mb[kind], key))])
    L += ["", "![loss](loss.png)", "![bpb](bpb.png)"]
    out = out_dir / "comparison.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
