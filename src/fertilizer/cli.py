"""Command-line interface for fertilizer."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    from . import enrichment, extract

    parser = argparse.ArgumentParser(
        prog="fertilizer",
        description=(
            "Per-region enrichment from single-replicate bigWig signal "
            "across conditions. Pipeline: `extract` aggregates bigWig "
            "signal over BED regions; `enrich` runs a DESeq2-inspired "
            "NB-GLM LRT to call loci with significantly enriched signal "
            "in one condition vs the others."
        ),
    )
    parser.add_argument("--version", action="version", version=f"fertilizer {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract.add_subparser(subparsers)
    enrichment.add_subparser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"{parser.prog}: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
