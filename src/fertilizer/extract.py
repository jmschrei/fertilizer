"""Signal aggregation over BED regions from bigWig tracks, and read or
fragment counts from BAM/SAM/CRAM and 10x fragment files."""

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

from . import counting


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
    key: str, bigwig_path: str, example_chrom: str, bigwig_chroms: list[str],
    kind: str = "bigWig",
) -> str:
	"""Compose a biologist-friendly warning for a per-input-file issue.

	`kind` names the input type in the message ("bigWig", "BAM" or
	"fragment file").
	"""
	if key == "missing_chrom":
		sample = ", ".join(sorted(bigwig_chroms)[:5])
		ellipsis = ", ..." if len(bigwig_chroms) > 5 else ""
		return (
		    f"chromosome {example_chrom!r} (and possibly others) referenced "
		    f"in BED but missing from {kind} {bigwig_path!r}; those values "
		    f"were filled with 0.0. The {kind} has chroms: [{sample}{ellipsis}]. "
		    "Common cause: mixing 'chr1'-style and '1'-style names "
		    "(hg19/hg38 vs Ensembl)."
		)
	if key == "out_of_bounds":
		return (
		    f"some regions extend beyond the chromosome length of {kind} "
		    f"{bigwig_path!r} (first encountered on chrom {example_chrom!r}); "
		    "those values were filled with 0.0. Common cause: mixing "
		    f"assemblies (e.g. hg19 BED against hg38 {kind})."
		)
	if key == "invalid_region":
		return (
		    f"some regions have non-positive length or a negative start "
		    f"(first encountered on chrom {example_chrom!r}); those values "
		    f"were filled with 0.0 ({kind} {bigwig_path!r})."
		)
	return f"unknown locus-level issue {key!r} in {kind} {bigwig_path!r}"

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
Examples:
  fertilizer extract -w A.bw B.bw C.bw -b peaks.bed -o signals.tsv -s sum
  fertilizer extract -a A.bam B.bam C.bam -b peaks.bed -o counts.tsv -ps 4 -ns -5
  fertilizer extract -f fragments.tsv.gz -g cells.tsv --group-column cluster \\
      -b peaks.bed -o counts.tsv

Use --stat sum with bigWigs if the output will be passed to `fertilizer
enrich` — the NB-GLM there assumes count-like input. BAM and fragment input
are always counted. See the README for full docs:
https://github.com/jmschrei/fertilizer#fertilizer-extract--signal-aggregation
"""


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
	"""Register the `fertilizer extract` subcommand."""
	parser = subparsers.add_parser(
	    "extract",
	    help="Summarize bigWigs, or count BAM reads or fragment ends, over BED regions.",
	    epilog=_EXTRACT_EPILOG,
	    formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	inputs = parser.add_mutually_exclusive_group(required=True)
	inputs.add_argument(
	    "-w", "--bigwigs", nargs="+", metavar="BIGWIG",
	    help="One or more input bigWig files.",
	)
	inputs.add_argument(
	    "-a", "--bams", nargs="+", metavar="BAM",
	    help="One or more BAM/SAM/CRAM files. Counts the 5' end of each read; "
	         "each mate of a pair counts separately. CRAM needs no reference "
	         "FASTA.",
	)
	inputs.add_argument(
	    "-f", "--fragments", nargs="+", metavar="FRAGMENTS",
	    help="One or more 10x-style fragment files (chrom, start, end, "
	         "barcode, count; plain or gzipped). Counts both ends of each "
	         "fragment (its Tn5 insertions); each line counts once.",
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
	    "-s", "--stat", choices=STAT_CHOICES, default=None,
	    help="Per-region summary statistic for bigWig input (default: mean). "
	         "BAM and fragment input are always counted.",
	)
	parser.add_argument(
	    "-n", "--names", nargs="+", default=None, metavar="NAME",
	    help="Override column names (one per input file). Defaults to each "
	         "file's name without its extension.",
	)
	parser.add_argument(
	    "-j", "--n-jobs", type=int, default=-1,
	    help="Number of parallel workers. -1 uses all cores (default).",
	)
	counts = parser.add_argument_group("BAM and fragment input")
	counts.add_argument(
	    "-ps", "--pos-shift", "--pos_shift", dest="pos_shift", type=int, default=None,
	    help="Added to each read's or fragment's start coordinate before "
	         "counting, as in bam2bw (default 0). `-ps 4 -ns -5` applies the "
	         "standard Tn5 offset; 10x fragment files are already shifted.",
	)
	counts.add_argument(
	    "-ns", "--neg-shift", "--neg_shift", dest="neg_shift", type=int, default=None,
	    help="Added to each read's or fragment's end coordinate before "
	         "counting, as in bam2bw (default 0).",
	)
	counts.add_argument(
	    "--min-mapq", type=int, default=None,
	    help="BAM/CRAM only: skip reads with mapping quality below this (default 30).",
	)
	counts.add_argument(
	    "--include-flagged", nargs="+", choices=sorted(counting.FLAG_BITS),
	    default=None, metavar="FLAG",
	    help="BAM/CRAM only: count reads with these flags, which are skipped by "
	         "default: duplicate, secondary, supplementary, qcfail. Unmapped "
	         "reads are always skipped.",
	)
	counts.add_argument(
	    "-g", "--groups", default=None, metavar="TSV",
	    help="Fragment files only: a tab-separated table with a header row "
	         "mapping cell barcodes to groups. Writes one column per group "
	         "(pseudobulk), summed over all fragment files; barcodes not in "
	         "the table are ignored.",
	)
	counts.add_argument(
	    "--barcode-column", default=None, metavar="COL",
	    help="Barcode column of the --groups table (default: barcode).",
	)
	counts.add_argument(
	    "--group-column", default=None, metavar="COL",
	    help="Group column of the --groups table (default: group).",
	)
	parser.set_defaults(func=run_extract)
	return parser


_KIND_LABELS = {"bigwigs": "bigWig", "bams": "BAM", "fragments": "fragment file"}


def _validate_input_options(args: argparse.Namespace, kind: str) -> None:
	"""Reject options that do not apply to the chosen input kind."""
	count_only = {"--pos-shift": args.pos_shift, "--neg-shift": args.neg_shift}
	bam_only = {"--min-mapq": args.min_mapq, "--include-flagged": args.include_flagged}
	fragment_only = {
	    "--groups": args.groups, "--barcode-column": args.barcode_column,
	    "--group-column": args.group_column,
	}
	not_allowed = {
	    "bigwigs": {**count_only, **bam_only, **fragment_only},
	    "bams": fragment_only,
	    "fragments": bam_only,
	}[kind]
	used = [flag for flag, value in not_allowed.items() if value is not None]
	if used:
		raise ValueError(f"{', '.join(used)} do(es) not apply to --{kind} input")
	if kind != "bigwigs" and args.stat is not None:
		raise ValueError(
		    "--stat applies to bigWig input only; BAM and fragment input are "
		    "always counted"
		)
	if args.groups is not None and args.names is not None:
		raise ValueError("--names cannot be combined with --groups; columns are the group names")
	if args.groups is None and (args.barcode_column or args.group_column):
		raise ValueError("--barcode-column and --group-column require --groups")
	if args.min_mapq is not None and args.min_mapq < 0:
		raise ValueError(f"--min-mapq must be >= 0, got {args.min_mapq}")


def _extract_bigwigs(
    paths: list[str], chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray,
    stat: str, n_jobs: int,
) -> tuple[np.ndarray, list[tuple[str, dict[str, str], list[str]]]]:
	"""Per-region statistic for each bigWig, as an (n_regions, n_files) array,
	plus (path, issues, chromosomes in file) for each file."""
	n = len(chroms)
	effective_n_jobs = joblib.cpu_count() if n_jobs == -1 else n_jobs
	# Size chunks to fill the thread pool; joblib will schedule the
	# (bigwig, slice) cross-product across workers.
	slices = _chunk_slices(n, effective_n_jobs)

	tasks = [(bw_idx, sl, bw_path)
	         for bw_idx, bw_path in enumerate(paths)
	         for sl in slices]
	results = Parallel(n_jobs=n_jobs, prefer="threads")(
	    delayed(_means_for_slice)(bw_path, chroms[sl], starts[sl], ends[sl], stat)
	    for (_, sl, bw_path) in tasks
	)

	values = np.zeros((n, len(paths)), dtype=np.float64)
	per_bw_issues: list[dict[str, str]] = [{} for _ in paths]
	for (bw_idx, sl, _), (means, issues) in zip(tasks, results, strict=True):
		values[sl, bw_idx] = means
		# First-seen example chrom per issue per bigWig — preserved across
		# chunks (don't clobber an earlier example with a later one).
		for k, v in issues.items():
			per_bw_issues[bw_idx].setdefault(k, v)

	file_issues = []
	for bw_path, issues in zip(paths, per_bw_issues, strict=True):
		chroms_in_bw: list[str] = []
		if issues:
			try:
				bw = _open_bw(bw_path)
				chroms_in_bw = list(bw.chroms().keys())
				bw.close()
			except Exception:
				chroms_in_bw = []
		file_issues.append((bw_path, issues, chroms_in_bw))
	return values, file_issues


def _extract_counts(
    kind: str, paths: list[str], chroms: np.ndarray, starts: np.ndarray,
    ends: np.ndarray, args: argparse.Namespace,
    groups: counting.BarcodeGroups | None,
) -> tuple[np.ndarray, list[tuple[str, dict[str, str], list[str]]]]:
	"""Counts per region for BAM or fragment input, as an (n_regions,
	n_columns) array, plus (path, issues, chromosomes in file) per file.

	Fragment files are split into byte ranges (see
	`counting.fragment_ranges`) so one file can use several workers. An
	indexed BAM or CRAM is split into chromosome pieces (see
	`counting.contig_ranges`); an unindexed one, or a SAM, is one task.
	Tasks run in separate processes because pysam iteration holds the GIL.
	"""
	n = len(chroms)
	pos_shift = args.pos_shift or 0
	neg_shift = args.neg_shift or 0
	effective_n_jobs = joblib.cpu_count() if args.n_jobs == -1 else args.n_jobs
	file_issues = []

	if kind == "fragments":
		# Split each file into byte ranges so that one large file also uses
		# every worker; files that cannot be split (plain gzip) or are small
		# are read as one stream.
		ranges_per_file = max(1, effective_n_jobs // len(paths))
		tasks = []
		for i, path in enumerate(paths):
			ranges = None
			if ranges_per_file > 1:
				ranges = counting.fragment_ranges(path, ranges_per_file)
			tasks += [(i, r) for r in ranges] if ranges else [(i, None)]
		results = Parallel(n_jobs=max(1, min(effective_n_jobs, len(tasks))))(
		    delayed(counting.count_fragments)(
		        paths[i], chroms, starts, ends, pos_shift, neg_shift, groups,
		        byte_range=byte_range,
		    )
		    for i, byte_range in tasks
		)
		n_columns = 1 if groups is None else len(groups.names)
		per_file = [np.zeros((n, n_columns), dtype=np.int64) for _ in paths]
		observed: list[set[str]] = [set() for _ in paths]
		for (i, _), (task_counts, task_observed) in zip(tasks, results, strict=True):
			per_file[i] += task_counts
			observed[i] |= task_observed
		for path, file_counts, file_observed in zip(paths, per_file, observed, strict=True):
			issues, bad = counting.count_issues(chroms, starts, ends, None, file_observed)
			file_counts[bad] = 0
			file_issues.append((path, issues, sorted(file_observed)))
		values = sum(per_file) if groups is not None else np.hstack(per_file)
		return values, file_issues

	min_mapq = 30 if args.min_mapq is None else args.min_mapq
	included = set(args.include_flagged or ())
	skip_flags = sum(bit for name, bit in counting.FLAG_BITS.items() if name not in included)
	lengths = [counting.bam_chrom_lengths(path) for path in paths]
	# An indexed file is split into chromosome pieces, about four per worker
	# so that dense and sparse pieces balance out. Each piece task gets the
	# regions of its chromosome only.
	by_chrom = pd.DataFrame({"chrom": chroms}).groupby("chrom", sort=False).indices
	pieces_per_file = max(1, 4 * effective_n_jobs // len(paths))
	tasks = []
	for i, path in enumerate(paths):
		if counting.bam_has_index(path):
			present = {c: lengths[i][c] for c in by_chrom if c in lengths[i]}
			for contig, lo, hi in counting.contig_ranges(present, pieces_per_file):
				tasks.append((i, contig, lo, hi, by_chrom[contig]))
		else:
			tasks.append((i, None, None, None, None))
	results = Parallel(n_jobs=max(1, min(effective_n_jobs, len(tasks))))(
	    delayed(counting.count_bam)(
	        paths[i],
	        chroms if rows is None else chroms[rows],
	        starts if rows is None else starts[rows],
	        ends if rows is None else ends[rows],
	        pos_shift, neg_shift, min_mapq, skip_flags,
	        contig=contig, start=lo, stop=hi,
	    )
	    for i, contig, lo, hi, rows in tasks
	)
	values = np.zeros((n, len(paths)), dtype=np.int64)
	for (i, _, _, _, rows), task_counts in zip(tasks, results, strict=True):
		if rows is None:
			values[:, i] += task_counts[:, 0]
		else:
			values[rows, i] += task_counts[:, 0]
	for i, path in enumerate(paths):
		issues, bad = counting.count_issues(chroms, starts, ends, lengths[i])
		values[bad, i] = 0
		file_issues.append((path, issues, list(lengths[i])))
	return values, file_issues


def run_extract(args: argparse.Namespace) -> int:
	if args.n_jobs != -1 and args.n_jobs < 1:
		raise ValueError(f"n_jobs must be -1 or >= 1, got {args.n_jobs}")

	kind = next(k for k in ("bigwigs", "bams", "fragments") if getattr(args, k) is not None)
	paths = list(getattr(args, kind))
	label = _KIND_LABELS[kind]
	_validate_input_options(args, kind)
	stat = (args.stat or "mean") if kind == "bigwigs" else "count"

	groups = None
	if args.groups is not None:
		groups = counting.BarcodeGroups.from_table(
		    args.groups, args.barcode_column or "barcode", args.group_column or "group",
		)
		stems = list(groups.names)
		source = f"--groups table {args.groups!r}"
	elif args.names is not None:
		if len(args.names) != len(paths):
			raise ValueError(
			    f"--names has {len(args.names)} entries but --{kind} has "
			    f"{len(paths)}; they must match"
			)
		stems = list(args.names)
		source = "--names"
	else:
		if kind == "bigwigs":
			stems = [Path(path).stem for path in paths]
		else:
			stems = [counting.input_stem(path) for path in paths]
		source = f"{label} filename stems"
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

	# Sort by (chrom, start) so each worker reads its input in index order.
	order = np.lexsort((starts, chroms))
	inverse_order = np.empty(n, dtype=np.int64)
	inverse_order[order] = np.arange(n)
	chroms, starts, ends = chroms[order], starts[order], ends[order]

	if kind == "bigwigs":
		values, file_issues = _extract_bigwigs(paths, chroms, starts, ends, stat, args.n_jobs)
	else:
		values, file_issues = _extract_counts(kind, paths, chroms, starts, ends, args, groups)

	out = regions.copy()
	for j, stem in enumerate(stems):
		out[stem] = values[inverse_order, j]

	for path, issues, chroms_in_file in file_issues:
		for key in sorted(issues):
			warnings.warn(
			    _format_issue_warning(key, path, issues[key], chroms_in_file, kind=label),
			    FertilizerWarning, stacklevel=2,
			)

	if n > 0:
		signal_block = out[stems].to_numpy()
		zero_frac = float((signal_block == 0).mean())
		if zero_frac > 0.95:
			cells = "region-by-bigWig" if kind == "bigwigs" else "region-by-column"
			warnings.warn(
			    f"{zero_frac:.1%} of {cells} cells are exactly zero; "
			    f"this often means a wrong {label} path, a chromosome-naming "
			    "mismatch (chr1 vs 1), or BED regions outside the assembly.",
			    FertilizerWarning, stacklevel=2,
			)

	# Write a metadata header so `fertilizer enrich` can verify that the
	# aggregation used here is compatible with the NB-GLM it applies.
	# Output is gzipped transparently when args.output ends in .gz.
	if kind == "bigwigs":
		header = f"# fertilizer-extract stat={stat}\n"
	else:
		source_name = "bam" if kind == "bams" else "fragments"
		header = (
		    f"# fertilizer-extract stat=count source={source_name} "
		    f"pos_shift={args.pos_shift or 0} neg_shift={args.neg_shift or 0}\n"
		)
	with _open_text_write(args.output) as fh:
		fh.write(header)
		out.to_csv(fh, sep="\t", index=False)
	return 0
