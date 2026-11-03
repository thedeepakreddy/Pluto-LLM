"""`llm` command-line tool. Each subcommand imports its module lazily so the
CLI starts fast and a missing optional dependency only affects its command."""
from __future__ import annotations

import argparse
import sys
import wave
from collections.abc import Callable

Handler = Callable[[argparse.Namespace], int]
_COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None], Handler]] = {}


def command(name: str, help: str, configure: Callable[[argparse.ArgumentParser], None]):
    def deco(fn: Handler) -> Handler:
        _COMMANDS[name] = (help, configure, fn)
        return fn

    return deco


def config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="path to the experiment YAML")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="SECTION.KEY=VALUE", help="override a config value (repeatable)")


def _load(args):
    from llm.config import load_config

    return load_config(args.config, args.overrides)


@command("prepare-data", "download, clean and split a dataset", config_args)
def _prepare_data(args) -> int:
    from llm.data.prepare import prepare_data

    print(prepare_data(_load(args)))
    return 0


@command("train-tokenizer", "train the byte-level BPE tokenizer", config_args)
def _train_tokenizer(args) -> int:
    from llm.tokenizer.bpe import train_tokenizer

    print(train_tokenizer(_load(args)))
    return 0


@command("tokenize", "encode the dataset splits into token shards", config_args)
def _tokenize(args) -> int:
    from llm.shards import write_shards

    print(write_shards(_load(args)))
    return 0


def _generate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("checkpoint", help="checkpoint .pt file or run directory")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)


@command("generate", "generate text from a checkpoint", _generate_args)
def _generate(args) -> int:
    from llm.generate import generate
    from llm.train.checkpoint import load_model_for_inference

    model, tok, _ = load_model_for_inference(args.checkpoint)
    ids = generate(model, tok.encode(args.prompt), args.max_new_tokens, temperature=args.temperature,
                   top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                   stop_ids=[tok.eot_id], seed=args.seed)
    print(args.prompt + tok.decode(ids))
    return 0


def _train_args(p: argparse.ArgumentParser) -> None:
    config_args(p)
    p.add_argument("--stop-at-step", type=int, default=None,
                   help="save a checkpoint and stop after this step (to test interrupt/resume)")


@command("train", "start a new training run", _train_args)
def _train(args) -> int:
    from llm.runs import create_run
    from llm.train.loop import train

    cfg = _load(args)
    run_dir = create_run(cfg)
    print(f"run: {run_dir}", flush=True)
    state, reason = train(cfg, run_dir, stop_at_step=args.stop_at_step)
    print(f"stopped: {reason} at step {state.step}")
    return 0


def _resume_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dir", help="run directory to continue")
    p.add_argument("--stop-at-step", type=int, default=None)


@command("resume", "continue a run from its latest valid checkpoint", _resume_args)
def _resume(args) -> int:
    from pathlib import Path

    from llm.config import load_config
    from llm.train.loop import train

    run_dir = Path(args.run_dir).resolve()
    state, reason = train(load_config(run_dir / "config.yaml"), run_dir, resume=True,
                          stop_at_step=args.stop_at_step)
    print(f"run: {run_dir}")
    print(f"stopped: {reason} at step {state.step}")
    return 0


@command("report", "write report.md and charts for a run",
         lambda p: p.add_argument("run_dir"))
def _report(args) -> int:
    from llm.report import build_report

    print(build_report(args.run_dir))
    return 0


def _compare_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--out", default=None, help="output directory (default runs/compare-<a>-vs-<b>)")


@command("compare", "compare two runs side by side", _compare_args)
def _compare(args) -> int:
    from pathlib import Path

    from llm.report import compare_runs

    a, b = Path(args.run_a), Path(args.run_b)
    out = Path(args.out) if args.out else a.parent / f"compare-{a.name}-vs-{b.name}"
    print(compare_runs(a, b, out))
    return 0


def _pull_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("hub_repo", help="Hugging Face repo id, e.g. your-name/pluto-runs")
    p.add_argument("run_id", help="run directory name, e.g. m1-tinystories-20261101-120000-abc1234")
    p.add_argument("--runs-root", default="runs")


@command("pull-run", "restore a run directory from the Hugging Face Hub (after a session dies)", _pull_args)
def _pull_run(args) -> int:
    from llm.train.checkpoint import pull_run

    print(pull_run(args.hub_repo, args.run_id, args.runs_root))
    return 0


def _build_speech_args(p: argparse.ArgumentParser) -> None:
    config_args(p)
    p.add_argument("--max-hours-session", type=float, default=None, metavar="H",
                   help="stop cleanly after H wall-clock hours (re-run to resume)")
    p.add_argument("--timings", choices=("tts", "whisper"), default="tts",
                   help="source of word timings: the TTS's own (default) or Whisper")


@command("build-speech", "synthesize the speech dataset (resumable)", _build_speech_args)
def _build_speech(args) -> int:
    from llm.speech.build import build_speech_data
    from llm.speech.interfaces import make_asr, make_codec, make_tts

    cfg = _load(args)
    if cfg.speech is None:
        print("error: config has no speech section", file=sys.stderr)
        return 2
    budget = None if args.max_hours_session is None else args.max_hours_session * 3600
    out, complete = build_speech_data(cfg, make_tts(cfg.speech), make_codec(cfg.speech),
                                      make_asr(cfg.speech), max_seconds_this_session=budget,
                                      progress=print, timings=args.timings)
    print(f"{out}: {'complete' if complete else 'incomplete (re-run to resume)'}")
    return 0


def _speech_check_args(p: argparse.ArgumentParser) -> None:
    config_args(p)
    p.add_argument("--n", type=int, default=100, help="val chunks to synthesize (default 100)")


@command("speech-check", "Stage 0: sanity-check Kokoro/Mimi/Whisper on real val chunks", _speech_check_args)
def _speech_check(args) -> int:
    from llm.speech.check import run_speech_check

    cfg = _load(args)
    if cfg.speech is None:
        print("error: config has no speech section", file=sys.stderr)
        return 2
    print(run_speech_check(cfg, n=args.n))
    return 0


@command("prepare-identity", "creator identity (PCID) -> statements, probes, manifest", config_args)
def _prepare_identity(args) -> int:
    from llm.identity.pcid import prepare_identity

    cfg = _load(args)
    if cfg.identity is None:
        print("error: config has no identity section", file=sys.stderr)
        return 2
    print(prepare_identity(cfg))
    return 0


def _eval_speech_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dir")
    p.add_argument("--n", type=int, default=200, help="held-out test chunks to score (default 200)")
    p.add_argument("--floor-n", type=int, default=None,
                   help="score the untrained floor on only the first N chunks (it is slow)")
    p.add_argument("--floor", action="store_true", help="also score a freshly initialised model (seed 0)")


@command("eval-speech", "Whisper-scored WER/CER, duration ratio and RTF for a speech run", _eval_speech_args)
def _eval_speech(args) -> int:
    from llm.speech.evaluate import run_eval_speech

    print(run_eval_speech(args.run_dir, n=args.n, floor=args.floor, floor_n=args.floor_n))
    return 0


def _sampling_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)


def _speak_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run", help="speech run directory or checkpoint .pt")
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True, help="output .wav (24 kHz) or .npy with --codes-only")
    p.add_argument("--codes-only", action="store_true", help="write codec codes (.npy), no codec needed")
    _sampling_args(p)


def _load_speech(run):
    from llm.train.checkpoint import load_speech_model_for_inference

    model, tok, cfg = load_speech_model_for_inference(run)
    if cfg.speech is None:
        raise ValueError("config has no speech section")
    return model, tok, cfg


@command("speak", "synthesize speech from text with a speech run", _speak_args)
def _speak(args) -> int:
    import numpy as np

    from llm.speech.generate import SpeechSampling, speak
    from llm.speech.interfaces import make_codec
    from llm.speech.wavio import write_wav

    try:
        model, tok, cfg = _load_speech(args.run)
        codes = speak(model, args.text, tok, cfg.audio,
                      SpeechSampling(temperature=args.temperature, top_k=args.top_k, seed=args.seed))
        if args.codes_only:
            np.save(args.out, codes)
        else:
            codec = make_codec(cfg.speech)
            write_wav(args.out, codec.decode(codes), codec.sample_rate)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(args.out)
    return 0


def _continue_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run", help="speech run directory or checkpoint .pt")
    p.add_argument("--audio", required=True, help="prompt wav (16-bit PCM)")
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-frames", type=int, default=250)
    p.add_argument("--codes-only", action="store_true")
    _sampling_args(p)


@command("continue-speech", "continue a spoken prompt with a speech run", _continue_args)
def _continue_speech(args) -> int:
    import numpy as np

    from llm.speech.generate import SpeechSampling, continue_speech
    from llm.speech.interfaces import make_asr, make_codec
    from llm.speech.wavio import read_wav, resample, write_wav

    try:
        model, tok, cfg = _load_speech(args.run)
        prompt, sr = read_wav(args.audio)
        codec = make_codec(cfg.speech)
        prompt = resample(prompt, sr, codec.sample_rate)
        codes = continue_speech(model, prompt, codec.sample_rate, tok, codec, make_asr(cfg.speech), cfg.audio,
                                SpeechSampling(temperature=args.temperature, top_k=args.top_k,
                                               seed=args.seed),
                                max_new_frames=args.max_new_frames)
        if args.codes_only:
            np.save(args.out, codes)
        else:
            write_wav(args.out, codec.decode(codes), codec.sample_rate)
    except (ValueError, FileNotFoundError, NotImplementedError, wave.Error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(args.out)
    return 0


def _demo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("run")
    p.add_argument("--share", action="store_true")


@command("demo", "launch the Gradio demo for a speech run", _demo_args)
def _demo(args) -> int:
    from pathlib import Path

    from llm.speech.app import build_app

    build_app(Path(args.run)).launch(share=args.share)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm", description="From-scratch LLM training system")
    sub = parser.add_subparsers(dest="command")
    for name, (help_, configure, fn) in _COMMANDS.items():
        p = sub.add_parser(name, help=help_)
        configure(p)
        p.set_defaults(handler=fn)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_usage(sys.stderr)
        return 2
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
