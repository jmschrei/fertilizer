"""Signal aggregation over BED regions from bigWig tracks."""

from __future__ import annotations

import argparse
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyBigWig
from joblib import Parallel, delayed


class FertilizerWarning(UserWarning):
    """Warning category for locus-level problems encountered during aggregation."""


_WARNING_MESSAGES = {
    "missing_chrom": (
        "some regions reference a chromosome not present in a bigWig; "
        "those values were filled with 0.0"
    ),
    "out_of_bounds": (
        "some regions extend beyond the chromosome length of a bigWig; "
        "those values were filled with 0.0"
    ),
    "invalid_region": (
        "some regions have non-positive length or a negative start; "
        "those values were filled with 0.0"
    ),
}

STAT_CHOICES = ("mean", "max", "min", "sum", "std", "coverage")


def _empty_regions() -> pd.DataFrame:
    return pd.DataFrame({
        "chrom": pd.Series(dtype=str),
        "start": pd.Series(dtype=np.int64),
        "end": pd.Series(dtype=np.int64),
    })


def load_regions(bed_paths: list[str]) -> pd.DataFrame:
    """Load one or more BED files and return chrom/start/end as a single frame.

    Chromosome names are forced to string dtype so numeric-named chroms
    (e.g. "1", "2") survive round-trips against bigWig keys. `#`-prefixed
    comment lines are skipped. Empty BED files contribute zero rows.
    """
    frames: list[pd.DataFrame] = []
    for path in bed_paths:
        try:
            frames.append(pd.read_csv(
                path, sep="\t", header=None, comment="#",
                usecols=[0, 1, 2],
                names=["chrom", "start", "end"],
                dtype={"chrom": str, "start": np.int64, "end": np.int64},
            ))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else _empty_regions()


def _means_for_slice(
    bigwig_path: str,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    stat: str = "mean",
) -> tuple[np.ndarray, set[str]]:
    """Compute per-region summary statistic for pre-extracted coordinate arrays.

    Locus-level problems are reported via the returned issue-key set and the
    corresponding output value is left as 0.0. Uncovered but otherwise valid
    regions also yield 0.0 but are not reported.
    """
    bw = pyBigWig.open(bigwig_path)
    issues: set[str] = set()
    try:
        chrom_lengths = bw.chroms()
        n = len(chroms)
        means = np.zeros(n, dtype=np.float64)
        for i in range(n):
            chrom = chroms[i]
            start = int(starts[i])
            end = int(ends[i])
            length = chrom_lengths.get(chrom)
            if length is None:
                issues.add("missing_chrom")
                continue
            if start < 0 or start >= end:
                issues.add("invalid_region")
                continue
            if end > length:
                issues.add("out_of_bounds")
                continue
            value = bw.stats(chrom, start, end, type=stat, nBins=1)[0]
            if value is not None and not np.isnan(value):
                means[i] = value
    finally:
        bw.close()
    return means, issues


def bigwig_region_means(
    regions: pd.DataFrame, bigwig_path: str, stat: str = "mean",
) -> tuple[np.ndarray, set[str]]:
    """Return per-region summary statistic in `bigwig_path` and the set of
    locus-issue keys encountered.

    Locus issues (missing chrom, out-of-bounds, invalid region) yield 0.0 in the
    output and are reported via the issue set. Regions that are well-formed but
    simply have no coverage in the bigWig also yield 0.0 but are *not* reported,
    since missing coverage is a property of the data, not of the loci.
    """
    chroms = regions["chrom"].to_numpy()
    starts = regions["start"].to_numpy(dtype=np.int64)
    ends = regions["end"].to_numpy(dtype=np.int64)
    return _means_for_slice(bigwig_path, chroms, starts, ends, stat=stat)


def _chunk_slices(n: int, n_chunks: int) -> list[slice]:
    """Partition range(n) into up to n_chunks contiguous non-empty slices."""
    if n == 0:
        return []
    n_chunks = max(1, min(n_chunks, n))
    bounds = np.linspace(0, n, n_chunks + 1, dtype=np.int64)
    return [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(n_chunks)]


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the `fertilizer extract` subcommand."""
    parser = subparsers.add_parser(
        "extract",
        help="Extract a summary statistic from bigWigs over BED regions.",
    )
    parser.add_argument(
        "-w", "--bigwigs", nargs="+", required=True, metavar="BIGWIG",
        help="One or more input bigWig files.",
    )
    parser.add_argument(
        "-b", "--beds", nargs="+", required=True, metavar="BED",
        help="One or more input BED files.",
    )
    parser.add_argument(
        "-o", "--output", required=True, metavar="TSV",
        help="Path to output TSV.",
    )
    parser.add_argument(
        "-s", "--stat", choices=STAT_CHOICES, default="mean",
        help="Per-region summary statistic (default: mean).",
    )
    parser.add_argument(
        "-j", "--n-jobs", type=int, default=-1,
        help="Number of parallel workers. -1 uses all cores (default).",
    )
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.n_jobs != -1 and args.n_jobs < 1:
        raise ValueError(f"n_jobs must be -1 or >= 1, got {args.n_jobs}")

    stems = [Path(bw).stem for bw in args.bigwigs]
    dupes = [stem for stem, count in Counter(stems).items() if count > 1]
    if dupes:
        raise ValueError(
            f"duplicate bigWig filename stems would produce colliding columns: "
            f"{sorted(dupes)}"
        )

    regions = load_regions(args.beds)
    n = len(regions)

    chroms = regions["chrom"].to_numpy()
    starts = regions["start"].to_numpy(dtype=np.int64)
    ends = regions["end"].to_numpy(dtype=np.int64)

    # Sort by (chrom, start) so each worker reads its bigWig in index order.
    order = np.lexsort((starts, chroms))
    inverse_order = np.empty(n, dtype=np.int64)
    inverse_order[order] = np.arange(n)
    chroms, starts, ends = chroms[order], starts[order], ends[order]

    effective_n_jobs = joblib.cpu_count() if args.n_jobs == -1 else args.n_jobs
    chunks_per_bw = max(1, effective_n_jobs // len(args.bigwigs))
    slices = _chunk_slices(n, chunks_per_bw)

    tasks = [(bw_idx, sl, bw_path)
             for bw_idx, bw_path in enumerate(args.bigwigs)
             for sl in slices]
    results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
        delayed(_means_for_slice)(bw_path, chroms[sl], starts[sl], ends[sl], args.stat)
        for (_, sl, bw_path) in tasks
    )

    per_bw_means = [np.zeros(n, dtype=np.float64) for _ in args.bigwigs]
    per_bw_issues: list[set[str]] = [set() for _ in args.bigwigs]
    for (bw_idx, sl, _), (means, issues) in zip(tasks, results):
        per_bw_means[bw_idx][sl] = means
        per_bw_issues[bw_idx] |= issues

    out = regions.copy()
    all_issues: set[str] = set()
    for stem, sorted_means, issues in zip(stems, per_bw_means, per_bw_issues):
        out[stem] = sorted_means[inverse_order]
        all_issues |= issues

    for key in sorted(all_issues):
        warnings.warn(_WARNING_MESSAGES[key], FertilizerWarning, stacklevel=2)

    out.to_csv(args.output, sep="\t", index=False)
    return 0
