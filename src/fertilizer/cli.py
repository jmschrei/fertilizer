"""Command-line interface for fertilizer."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    from . import diff, extract

    parser = argparse.ArgumentParser(
        prog="fertilizer",
        description="Identify fertile ground in genomes for regulatory design.",
    )
    parser.add_argument("--version", action="version", version=f"fertilizer {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract.add_subparser(subparsers)
    diff.add_subparser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
