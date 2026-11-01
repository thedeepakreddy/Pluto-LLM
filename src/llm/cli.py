"""`llm` command-line tool. Each subcommand imports its module lazily so the
CLI starts fast and a missing optional dependency only affects its command."""
from __future__ import annotations

import argparse
import sys
from typing import Callable

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
