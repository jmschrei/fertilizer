"""Signal aggregation over BED regions from bigWig tracks."""

from __future__ import annotations

import argparse
import gzip
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyBigWig
from joblib import Parallel, delayed


def _open_text_write(path: str):
	"""Open `path` for text writing, transparently using gzip if path ends .gz."""
	if str(path).endswith(".gz"):
		return gzip.open(path, "wt")
	return open(path, "w")

__all__ = [
    "FertilizerWarning",
    "STAT_CHOICES",
    "bigwig_region_means",
    "load_regions",
    "run_extract",
]


def _open_bw(path: str):
	"""Open `path` with pyBigWig. The caller closes the handle.

	Handles are not cached across calls: a cached handle keeps reading the old
	file after the path is rewritten, and each slice opens its bigWig once.
	"""
	try:
		return pyBigWig.open(path)
	except RuntimeError as e:
		raise ValueError(f"could not open bigWig {path!r}: {e}") from e


class FertilizerWarning(UserWarning):
	"""Warning category for locus-level problems encountered during aggregation."""


def _format_issue_warning(
    key: str, bigwig_path: str, example_chrom: str, bigwig_chroms: list[str]
) -> str:
	"""Compose a biologist-friendly warning for a per-bigWig issue."""
	if key == "missing_chrom":
		sample = ", ".join(sorted(bigwig_chroms)[:5])
		ellipsis = ", ..." if len(bigwig_chroms) > 5 else ""
		return (
		    f"chromosome {example_chrom!r} (and possibly others) referenced "
		    f"in BED but missing from bigWig {bigwig_path!r}; those values "
		    f"were filled with 0.0. Bigwig has chroms: [{sample}{ellipsis}]. "
		    "Common cause: mixing 'chr1'-style and '1'-style names "
		    "(hg19/hg38 vs Ensembl)."
		)
	if key == "out_of_bounds":
		return (
		    f"some regions extend beyond the chromosome length of bigWig "
		    f"{bigwig_path!r} (first encountered on chrom {example_chrom!r}); "
		    "those values were filled with 0.0. Common cause: mixing "
		    "assemblies (e.g. hg19 BED against hg38 bigWig)."
		)
	if key == "invalid_region":
		return (
		    f"some regions have non-positive length or a negative start "
		    f"(first encountered on chrom {example_chrom!r}); those values "
		    f"were filled with 0.0 (bigWig {bigwig_path!r})."
		)
	return f"unknown locus-level issue {key!r} in bigWig {bigwig_path!r}"

STAT_CHOICES = ("mean", "max", "min", "sum", "std", "coverage")


def _empty_regions() -> pd.DataFrame:
	return pd.DataFrame({
	    "chrom": pd.Series(dtype=str),
	    "start": pd.Series(dtype=np.int64),
	    "end": pd.Series(dtype=np.int64),
	})


# Standard BED column names through BED6 — the only positions whose meaning
# is unambiguous across BED variants. BED12 columns 7-12 and narrowPeak
# columns 7-10 disagree on what each position means, so past column 6 we
# fall back to generic `bed_col_<i>` rather than guess. Users who need
# narrowPeak's signalValue / pValue / qValue / peak names can rename
# downstream — the data is preserved either way.
_BED_COL_NAMES = ("chrom", "start", "end", "name", "score", "strand")


def _count_header_lines(path: str) -> int:
	"""Number of leading UCSC `track`/`browser`, `#` comment or blank lines."""
	opener = gzip.open if str(path).endswith(".gz") else open
	n = 0
	with opener(path, "rt", errors="replace") as fh:
		for line in fh:
			if line.strip() and not line.startswith(("track", "browser", "#")):
				break
			n += 1
	return n


def load_regions(bed_paths: list[str]) -> pd.DataFrame:
	"""Load one or more BED files and return a single frame.

	The first three columns (chrom/start/end) are required. Additional
	columns are passed through with conventional BED/narrowPeak names
	(`name`, `score`, `strand`, ...) when present, so a peak `name`
	column survives the pipeline and can be used to join back to
	upstream annotations.

	Chromosome names are forced to string dtype so numeric-named chroms
	(e.g. "1", "2") survive round-trips against bigWig keys. `#`-prefixed
	comment lines are skipped, as are UCSC `track` and `browser` lines at the
	top of a file. Empty BED files contribute zero rows.

	All input BED files are expected to have the same number of columns;
	extra columns in some files but not others will produce NaN in the
	concatenated frame.
	"""
	frames: list[pd.DataFrame] = []
	for path in bed_paths:
		try:
			frame = pd.read_csv(
			    path, sep="\t", header=None, comment="#",
			    skiprows=_count_header_lines(path),
			    dtype={0: str, 1: np.int64, 2: np.int64},
			)
		except pd.errors.EmptyDataError:
			continue
		if frame.shape[1] < 3:
			raise ValueError(
			    f"BED file {path!r} has only {frame.shape[1]} column(s); "
			    "at least 3 (chrom, start, end) are required"
			)
		n_cols = frame.shape[1]
		named = [
		    _BED_COL_NAMES[i] if i < len(_BED_COL_NAMES) else f"bed_col_{i}"
		    for i in range(n_cols)
		]
		frame.columns = named
		frames.append(frame)
	return pd.concat(frames, ignore_index=True) if frames else _empty_regions()


def _means_for_slice(
    bigwig_path: str,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    stat: str = "mean",
) -> tuple[np.ndarray, dict[str, str]]:
	"""Compute per-region summary statistic for pre-extracted coordinate arrays.

	Locus-level problems are reported via the returned dict (issue key -> the
	first chromosome on which the issue was observed in this slice) and the
	corresponding output value is left as 0.0. Uncovered but otherwise valid
	regions also yield 0.0 but are not reported.
	"""
	bw = _open_bw(bigwig_path)
	try:
		return _stats_for_slice(bw, chroms, starts, ends, stat)
	finally:
		bw.close()


def _stats_for_slice(
    bw, chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray, stat: str,
) -> tuple[np.ndarray, dict[str, str]]:
	"""Body of `_means_for_slice` on an already-open pyBigWig handle."""
	issues: dict[str, str] = {}
	chrom_lengths = bw.chroms()
	n = len(chroms)
	means = np.zeros(n, dtype=np.float64)
	for i in range(n):
		chrom = chroms[i]
		start = int(starts[i])
		end = int(ends[i])
		length = chrom_lengths.get(chrom)
		if length is None:
			issues.setdefault("missing_chrom", chrom)
			continue
		if start < 0 or start >= end:
			issues.setdefault("invalid_region", chrom)
			continue
		if end > length:
			issues.setdefault("out_of_bounds", chrom)
			continue
		# exact=True reads the full-resolution data. The default answers from
		# zoom levels when one is coarse enough, and pyBigWig's zoom-level
		# `sum` is wrong by orders of magnitude (the others are approximate).
		value = bw.stats(chrom, start, end, type=stat, nBins=1, exact=True)[0]
		if value is not None and not np.isnan(value):
			means[i] = value
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

	In a partially covered region, every statistic except `coverage` is taken
	over the covered bases only: uncovered bases do not pull `mean` or `min`
	toward 0, and add nothing to `sum`. `coverage` is the covered fraction.
	"""
	if stat not in STAT_CHOICES:
		raise ValueError(
		    f"unknown stat {stat!r}; must be one of {STAT_CHOICES}"
		)
	chroms = regions["chrom"].to_numpy()
	starts = regions["start"].to_numpy(dtype=np.int64)
	ends = regions["end"].to_numpy(dtype=np.int64)
	means, issues = _means_for_slice(bigwig_path, chroms, starts, ends, stat=stat)
	return means, set(issues.keys())


def _chunk_slices(n: int, n_chunks: int) -> list[slice]:
	"""Partition range(n) into up to n_chunks contiguous non-empty slices."""
	if n == 0:
		return []
	n_chunks = max(1, min(n_chunks, n))
	bounds = np.linspace(0, n, n_chunks + 1, dtype=np.int64)
	return [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(n_chunks)]


_EXTRACT_EPILOG = """\
Example:
  fertilizer extract -w A.bw B.bw C.bw -b peaks.bed -o signals.tsv -s sum

Use --stat sum if the output will be passed to `fertilizer enrich` — the
NB-GLM there assumes count-like input. See the README for full docs:
https://github.com/jmschrei/fertilizer#fertilizer-extract--signal-aggregation
"""


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
	"""Register the `fertilizer extract` subcommand."""
	parser = subparsers.add_parser(
	    "extract",
	    help="Extract a summary statistic from bigWigs over BED regions.",
	    epilog=_EXTRACT_EPILOG,
	    formatter_class=argparse.RawDescriptionHelpFormatter,
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
	    "-n", "--names", nargs="+", default=None, metavar="NAME",
	    help="Override column names (one per --bigwigs entry). "
	         "Defaults to each bigWig's filename stem.",
	)
	parser.add_argument(
	    "-j", "--n-jobs", type=int, default=-1,
	    help="Number of parallel workers. -1 uses all cores (default).",
	)
	parser.set_defaults(func=run_extract)
	return parser


def run_extract(args: argparse.Namespace) -> int:
	if args.n_jobs != -1 and args.n_jobs < 1:
		raise ValueError(f"n_jobs must be -1 or >= 1, got {args.n_jobs}")

	if args.names is not None:
		if len(args.names) != len(args.bigwigs):
			raise ValueError(
			    f"--names has {len(args.names)} entries but --bigwigs has "
			    f"{len(args.bigwigs)}; they must match"
			)
		stems = list(args.names)
	else:
		stems = [Path(bw).stem for bw in args.bigwigs]
	source = "--names" if args.names is not None else "bigWig filename stems"
	dupes = [stem for stem, count in Counter(stems).items() if count > 1]
	if dupes:
		raise ValueError(
		    f"duplicate column names from {source} would collide: {sorted(dupes)}"
		)

	regions = load_regions(args.beds)
	clash = sorted(set(stems) & set(regions.columns))
	if clash:
		raise ValueError(
		    f"column names {clash} from {source} would overwrite the BED "
		    "column(s) of the same name; pass -n/--names to rename the tracks"
		)
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
	# Size chunks to fill the thread pool; joblib will schedule the
	# (bigwig, slice) cross-product across workers.
	slices = _chunk_slices(n, effective_n_jobs)

	tasks = [(bw_idx, sl, bw_path)
	         for bw_idx, bw_path in enumerate(args.bigwigs)
	         for sl in slices]
	results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
	    delayed(_means_for_slice)(bw_path, chroms[sl], starts[sl], ends[sl], args.stat)
	    for (_, sl, bw_path) in tasks
	)

	per_bw_means = [np.zeros(n, dtype=np.float64) for _ in args.bigwigs]
	per_bw_issues: list[dict[str, str]] = [{} for _ in args.bigwigs]
	for (bw_idx, sl, _), (means, issues) in zip(tasks, results, strict=True):
		per_bw_means[bw_idx][sl] = means
		# First-seen example chrom per issue per bigWig — preserved across
		# chunks (don't clobber an earlier example with a later one).
		for k, v in issues.items():
			per_bw_issues[bw_idx].setdefault(k, v)

	out = regions.copy()
	for stem, sorted_means in zip(stems, per_bw_means, strict=True):
		out[stem] = sorted_means[inverse_order]

	for bw_path, issues in zip(args.bigwigs, per_bw_issues, strict=True):
		if not issues:
			continue
		try:
			bw = _open_bw(bw_path)
			chroms_in_bw = list(bw.chroms().keys())
			bw.close()
		except Exception:
			chroms_in_bw = []
		for key in sorted(issues):
			warnings.warn(
			    _format_issue_warning(key, bw_path, issues[key], chroms_in_bw),
			    FertilizerWarning, stacklevel=2,
			)

	if n > 0:
		signal_block = out[stems].to_numpy()
		zero_frac = float((signal_block == 0).mean())
		if zero_frac > 0.95:
			warnings.warn(
			    f"{zero_frac:.1%} of region-by-bigWig cells are exactly zero; "
			    "this often means a wrong bigWig path, a chromosome-naming "
			    "mismatch (chr1 vs 1), or BED regions outside the assembly.",
			    FertilizerWarning, stacklevel=2,
			)

	# Write a metadata header so `fertilizer enrich` can verify that the
	# aggregation used here is compatible with the NB-GLM it applies.
	# Output is gzipped transparently when args.output ends in .gz.
	with _open_text_write(args.output) as fh:
		fh.write(f"# fertilizer-extract stat={args.stat}\n")
		out.to_csv(fh, sep="\t", index=False)
	return 0
