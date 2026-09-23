"""Read and fragment counts over BED regions from BAM/SAM and 10x fragment files.

Every input is reduced to positions, and a region's count is the number of
positions falling in its half-open interval [start, end):

- **BAM/SAM:** the 5' end of each read that passes the filters: the leftmost
  aligned base for a forward read, the rightmost for a reverse read. Each mate
  of a pair is a separate read, so for paired-end ATAC-seq this counts both
  Tn5 insertions of every fragment.
- **Fragment files** (10x `chrom, start, end, barcode, count`): both ends of
  every fragment, i.e. its two Tn5 insertions, at `start` and `end - 1`. Each
  line counts once; the duplicate count in column 5 is ignored.

Shifts follow bam2bw exactly: `pos_shift` is added to a read's or fragment's
start coordinate and `neg_shift` to its end coordinate before the positions
above are taken, so `pos_shift=4, neg_shift=-5` applies the standard Tn5
offset. 10x fragment files are already shifted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pysam

__all__ = [
    "FLAG_BITS",
    "BarcodeGroups",
    "RegionCounter",
    "bam_chrom_lengths",
    "bam_has_index",
    "count_bam",
    "count_fragments",
    "count_issues",
    "input_stem",
]

# SAM flag bits skipped by default and re-admitted by --include-flagged.
# Unmapped reads (0x4) are always skipped: they have no position to count.
FLAG_BITS = {
    "duplicate": 0x400,
    "secondary": 0x100,
    "supplementary": 0x800,
    "qcfail": 0x200,
}
_UNMAPPED = 0x4
_REVERSE = 0x10

_STRIP_SUFFIXES = (".tsv.gz", ".tsv.bgz", ".bed.gz", ".tsv", ".bed", ".bam", ".sam")


def input_stem(path: str) -> str:
	"""Column name for a BAM or fragment file: its file name without the
	known extensions, so `C1.fragments.tsv.gz` becomes `C1.fragments`."""
	name = Path(path).name
	for suffix in _STRIP_SUFFIXES:
		if name.endswith(suffix) and len(name) > len(suffix):
			return name[: -len(suffix)]
	return Path(path).stem


class RegionCounter:
	"""Accumulate counts of positions in half-open regions, per output column.

	Regions are indexed by chromosome once. Each batch of positions is sorted
	and every region on that chromosome gets
	`searchsorted(positions, end) - searchsorted(positions, start)` added, so
	overlapping regions each receive every position they contain and a batch
	costs O(m log m + r log m) for m positions and r regions. Regions with a
	negative start or non-positive length never receive counts.
	"""

	def __init__(
	    self, chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray,
	    n_columns: int = 1,
	):
		self.starts = np.asarray(starts, dtype=np.int64)
		self.ends = np.asarray(ends, dtype=np.int64)
		valid = (self.starts >= 0) & (self.starts < self.ends)
		frame = pd.DataFrame({"chrom": np.asarray(chroms)[valid]})
		positions = np.flatnonzero(valid)
		self.index = {
		    chrom: positions[rows]
		    for chrom, rows in frame.groupby("chrom", sort=False).indices.items()
		}
		self.counts = np.zeros((len(self.starts), n_columns), dtype=np.int64)

	def add(
	    self, chrom: str, positions: np.ndarray, columns: np.ndarray | None = None,
	) -> None:
		"""Count `positions` on `chrom`; `columns[i]` is the output column of
		`positions[i]` (all column 0 when None)."""
		idx = self.index.get(chrom)
		if idx is None or len(positions) == 0:
			return
		starts, ends = self.starts[idx], self.ends[idx]
		if columns is None:
			p = np.sort(positions)
			self.counts[idx, 0] += np.searchsorted(p, ends) - np.searchsorted(p, starts)
			return
		order = np.lexsort((positions, columns))
		p, c = positions[order], columns[order]
		bounds = np.searchsorted(c, np.arange(self.counts.shape[1] + 1))
		for col in range(self.counts.shape[1]):
			lo, hi = bounds[col], bounds[col + 1]
			if hi > lo:
				seg = p[lo:hi]
				self.counts[idx, col] += np.searchsorted(seg, ends) - np.searchsorted(seg, starts)


class BarcodeGroups:
	"""Map cell barcodes to output columns (one per group) for pseudobulking
	a fragment file."""

	def __init__(self, barcodes: np.ndarray, group_index: np.ndarray, names: list[str]):
		self.barcodes = pd.Index(barcodes)
		self.group_index = np.asarray(group_index, dtype=np.int64)
		self.names = list(names)

	@classmethod
	def from_table(
	    cls, path: str, barcode_column: str = "barcode", group_column: str = "group",
	) -> BarcodeGroups:
		"""Read a tab-separated table with a header row (optionally gzipped).

		Rows with an empty group are dropped. Groups become output columns in
		order of first appearance. A barcode listed under two different groups
		is an error.
		"""
		table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
		missing = [c for c in (barcode_column, group_column) if c not in table.columns]
		if missing:
			raise ValueError(
			    f"--groups table {path!r} has no column(s) {missing}; its columns "
			    f"are {list(table.columns)[:20]}. Choose them with "
			    "--barcode-column and --group-column."
			)
		table = table.loc[table[group_column] != "", [barcode_column, group_column]]
		table = table.drop_duplicates()
		conflicted = table[barcode_column][table[barcode_column].duplicated()].unique()
		if len(conflicted) > 0:
			raise ValueError(
			    f"--groups table {path!r} assigns {len(conflicted)} barcode(s) to "
			    f"more than one group, e.g. {conflicted[0]!r}"
			)
		if len(table) == 0:
			raise ValueError(f"--groups table {path!r} has no barcodes with a group")
		names = list(pd.unique(table[group_column]))
		lookup = {name: i for i, name in enumerate(names)}
		return cls(
		    table[barcode_column].to_numpy(),
		    table[group_column].map(lookup).to_numpy(),
		    names,
		)

	def columns_for(self, barcodes: np.ndarray) -> np.ndarray:
		"""Output column of each barcode, or -1 for barcodes not in the table."""
		codes = self.barcodes.get_indexer(barcodes)
		return np.where(codes >= 0, self.group_index[codes], -1)


def count_fragments(
    path: str,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    pos_shift: int = 0,
    neg_shift: int = 0,
    groups: BarcodeGroups | None = None,
    chunksize: int = 2_000_000,
) -> tuple[np.ndarray, set[str]]:
	"""Count fragment ends (Tn5 insertions) in each region.

	Streams the file in chunks, so memory does not grow with its size and no
	index is needed. Returns `(counts, observed_chroms)`: counts has shape
	(n_regions, 1), or (n_regions, n_groups) with `groups`, and
	observed_chroms is every chromosome that appeared in the file.
	"""
	n_columns = 1 if groups is None else len(groups.names)
	counter = RegionCounter(chroms, starts, ends, n_columns)
	usecols = [0, 1, 2] if groups is None else [0, 1, 2, 3]
	dtypes = {0: str, 1: np.int64, 2: np.int64, 3: str}
	observed: set[str] = set()
	reader = pd.read_csv(
	    path, sep="\t", header=None, comment="#", usecols=usecols,
	    dtype={c: dtypes[c] for c in usecols}, chunksize=chunksize,
	)
	for chunk in reader:
		chrom_col = chunk[0].to_numpy()
		observed.update(pd.unique(chrom_col))
		left = chunk[1].to_numpy(dtype=np.int64) + pos_shift
		right = chunk[2].to_numpy(dtype=np.int64) + neg_shift - 1
		columns = None
		if groups is not None:
			columns = groups.columns_for(chunk[3].to_numpy())
			keep = columns >= 0
			chrom_col, left, right, columns = (
			    chrom_col[keep], left[keep], right[keep], columns[keep]
			)
		codes, uniques = pd.factorize(chrom_col)
		order = np.argsort(codes, kind="stable")
		bounds = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=len(uniques)))])
		for k, chrom in enumerate(uniques):
			if chrom not in counter.index:
				continue
			rows = order[bounds[k]:bounds[k + 1]]
			positions = np.concatenate([left[rows], right[rows]])
			cols = None if columns is None else np.concatenate([columns[rows], columns[rows]])
			counter.add(chrom, positions, cols)
	return counter.counts, observed


def _open_alignments(path: str):
	mode = "r" if str(path).endswith(".sam") else "rb"
	try:
		return pysam.AlignmentFile(path, mode)
	except (OSError, ValueError) as e:
		raise ValueError(f"could not open BAM/SAM {path!r}: {e}") from e


def bam_chrom_lengths(path: str) -> dict[str, int]:
	"""Chromosome lengths from the BAM/SAM header."""
	with _open_alignments(path) as bam:
		return dict(zip(bam.references, bam.lengths, strict=True))


def bam_has_index(path: str) -> bool:
	"""Whether `path` is a BAM with an index, so it can be read per chromosome."""
	if str(path).endswith(".sam"):
		return False
	with _open_alignments(path) as bam:
		return bam.has_index()


def count_bam(
    path: str,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    pos_shift: int = 0,
    neg_shift: int = 0,
    min_mapq: int = 30,
    skip_flags: int = sum(FLAG_BITS.values()),
    contig: str | None = None,
    batch: int = 1_000_000,
) -> np.ndarray:
	"""Count read 5' ends in each region; returns shape (n_regions, 1).

	Reads that are unmapped, have any flag in `skip_flags`, or have mapping
	quality below `min_mapq` are skipped. With `contig`, only that chromosome
	is read (requires an index); otherwise the whole file is streamed.
	"""
	counter = RegionCounter(chroms, starts, ends, 1)
	skip = skip_flags | _UNMAPPED
	with _open_alignments(path) as bam:
		names = bam.references
		wanted = {i for i, name in enumerate(names) if name in counter.index}
		buffers: dict[int, list[int]] = {}
		n_buffered = 0
		reads = bam.fetch(contig) if contig is not None else bam.fetch(until_eof=True)
		for read in reads:
			flag = read.flag
			if flag & skip or read.mapping_quality < min_mapq:
				continue
			rid = read.reference_id
			if rid not in wanted:
				continue
			if flag & _REVERSE:
				end = read.reference_end
				if end is None:
					raise ValueError(
					    f"{path}: read {read.query_name} is mapped to {names[rid]} "
					    "but has no CIGAR, so its reference end is unknown"
					)
				position = end + neg_shift - 1
			else:
				position = read.reference_start + pos_shift
			buffers.setdefault(rid, []).append(position)
			n_buffered += 1
			if n_buffered >= batch:
				for r, buf in buffers.items():
					counter.add(names[r], np.array(buf, dtype=np.int64))
				buffers, n_buffered = {}, 0
		for r, buf in buffers.items():
			counter.add(names[r], np.array(buf, dtype=np.int64))
	return counter.counts


def count_issues(
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    chrom_lengths: dict[str, int] | None,
    observed_chroms: set[str] | None = None,
) -> tuple[dict[str, str], np.ndarray]:
	"""Locus-level issues for counted input, in the same form as the bigWig path.

	Returns `(issues, bad)`: issues maps each issue key to the first
	chromosome it was seen on, and `bad` marks regions whose value must be
	0.0. A chromosome is missing when it is not in `chrom_lengths` (BAM) or,
	for fragment files, which carry no lengths, never appeared in the file.
	Out-of-bounds is only checked when lengths are known.
	"""
	issues: dict[str, str] = {}
	bad = np.zeros(len(chroms), dtype=bool)
	known = chrom_lengths if chrom_lengths is not None else dict.fromkeys(observed_chroms or ())
	for i, (chrom, start, end) in enumerate(zip(chroms, starts, ends, strict=True)):
		if chrom not in known:
			issues.setdefault("missing_chrom", chrom)
			bad[i] = True
		elif start < 0 or start >= end:
			issues.setdefault("invalid_region", chrom)
			bad[i] = True
		elif chrom_lengths is not None and end > chrom_lengths[chrom]:
			issues.setdefault("out_of_bounds", chrom)
			bad[i] = True
	return issues, bad
