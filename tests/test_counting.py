"""Tests for read and fragment counting from BAM/SAM and 10x fragment files."""

from __future__ import annotations

import gzip
import re
import struct
import warnings
import zlib

import numpy as np
import pandas as pd
import pyBigWig
import pysam
import pytest

from fertilizer.cli import main
from fertilizer.counting import (
    FLAG_BITS,
    BarcodeGroups,
    RegionCounter,
    bam_chrom_lengths,
    bam_has_index,
    contig_ranges,
    count_bam,
    count_fragments,
    count_issues,
    fragment_ranges,
    input_stem,
)
from fertilizer.extract import FertilizerWarning

REFS = (("chr1", 5000), ("chr2", 3000))
CIGARS = ("50M", "20M5D30M", "10S40M", "30M2I18M", "25M100N25M")


def _ref_len(cigar):
	return sum(int(n) for n, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar) if op in "MDN=X")


def _query_len(cigar):
	return sum(int(n) for n, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar) if op in "MIS=X")


def _random_reads(rng, n=400):
	"""(chrom_index, start, cigar, flag, mapq) with every filter case represented."""
	reads = []
	for _ in range(n):
		ref = int(rng.integers(0, len(REFS)))
		cigar = CIGARS[int(rng.integers(0, len(CIGARS)))]
		start = int(rng.integers(0, REFS[ref][1] - _ref_len(cigar)))
		flag = 0x10 if rng.random() < 0.5 else 0
		for bit in FLAG_BITS.values():
			if rng.random() < 0.08:
				flag |= bit
		if rng.random() < 0.05:
			flag |= 0x4
		reads.append((ref, start, cigar, flag, int(rng.integers(0, 61))))
	return reads


def _write_reference(path):
	"""A random FASTA for REFS, indexed, for writing CRAMs against."""
	rng = np.random.default_rng(99)
	with open(path, "w") as fh:
		for name, length in REFS:
			fh.write(f">{name}\n{''.join(rng.choice(list('ACGT'), size=length))}\n")
	pysam.faidx(str(path))
	return path


def _write_alignments(path, reads, index=True, reference=None):
	header = {"HD": {"VN": "1.6", "SO": "coordinate"},
	          "SQ": [{"SN": name, "LN": length} for name, length in REFS]}
	mode = {".sam": "w", "cram": "wc"}.get(str(path)[-4:], "wb")
	extra = {"reference_filename": str(reference)} if mode == "wc" else {}
	with pysam.AlignmentFile(str(path), mode, header=header, **extra) as out:
		for i, (ref, start, cigar, flag, mapq) in enumerate(sorted(reads, key=lambda r: (r[0], r[1]))):
			seg = pysam.AlignedSegment()
			seg.query_name = f"read{i}"
			seg.flag = flag
			seg.reference_id = ref
			seg.reference_start = start
			seg.mapping_quality = mapq
			seg.cigarstring = cigar
			seg.query_sequence = "A" * _query_len(cigar)
			seg.query_qualities = pysam.qualitystring_to_array("I" * _query_len(cigar))
			out.write(seg)
	if index:
		pysam.index(str(path))
	return path


def _bam_reference_counts(reads, regions, pos_shift=0, neg_shift=0, min_mapq=30, include=()):
	"""Brute-force 5'-end counts using bam2bw's position rules."""
	skip = 0x4 | sum(bit for name, bit in FLAG_BITS.items() if name not in include)
	counts = np.zeros(len(regions), dtype=np.int64)
	for ref, start, cigar, flag, mapq in reads:
		if flag & skip or mapq < min_mapq:
			continue
		reverse = flag & 0x10
		pos = start + _ref_len(cigar) + neg_shift - 1 if reverse else start + pos_shift
		chrom = REFS[ref][0]
		for i, (c, s, e) in enumerate(regions):
			counts[i] += c == chrom and s <= pos < e
	return counts


def _random_regions(rng, n=60):
	regions = []
	for _ in range(n):
		name, length = REFS[int(rng.integers(0, len(REFS)))]
		s = int(rng.integers(0, length - 10))
		regions.append((name, s, int(min(length, s + rng.integers(1, 800)))))
	return regions


def _arrays(regions):
	chroms, starts, ends = zip(*regions, strict=True)
	return np.array(chroms, dtype=object), np.array(starts), np.array(ends)


def _random_fragments(rng, n=500, barcodes=("AAA-1", "CCC-1", "GGG-2", "TTT-2")):
	rows = []
	for _ in range(n):
		name, length = REFS[int(rng.integers(0, len(REFS)))]
		s = int(rng.integers(0, length - 600))
		rows.append((name, s, s + int(rng.integers(20, 600)),
		             barcodes[int(rng.integers(0, len(barcodes)))], int(rng.integers(1, 9))))
	return rows


def _write_fragments(path, rows, header=True):
	opener = gzip.open if str(path).endswith(".gz") else open
	with opener(path, "wt") as fh:
		if header:
			fh.write("# id=test\n# primary_contig=chr1\n")
		for row in rows:
			fh.write("\t".join(map(str, row)) + "\n")
	return path


def _fragment_reference_counts(rows, regions, pos_shift=0, neg_shift=0, group_of=None, n_groups=1):
	counts = np.zeros((len(regions), n_groups), dtype=np.int64)
	for chrom, s, e, barcode, _ in rows:
		col = 0 if group_of is None else group_of.get(barcode)
		if col is None:
			continue
		for pos in (s + pos_shift, e + neg_shift - 1):
			for i, (c, rs, re_) in enumerate(regions):
				counts[i, col] += c == chrom and rs <= pos < re_
	return counts


##


class TestInputStem:
	@pytest.mark.parametrize("path,stem", [
	    ("dir/C1.fragments.tsv.gz", "C1.fragments"),
	    ("atac.tsv", "atac"),
	    ("x/y/sample.bam", "sample"),
	    ("reads.sam", "reads"),
	    ("reads.cram", "reads"),
	    ("peaks.bed.gz", "peaks"),
	    ("odd.name.txt", "odd.name"),
	])
	def test_strips_known_extensions(self, path, stem):
		assert input_stem(path) == stem


class TestRegionCounter:
	def test_half_open_boundaries_and_overlaps(self):
		counter = RegionCounter(np.array(["chr1"] * 3, dtype=object),
		                        np.array([10, 10, 15]), np.array([20, 15, 30]))
		counter.add("chr1", np.array([9, 10, 14, 15, 19, 20, 29, 30]))
		np.testing.assert_array_equal(counter.counts[:, 0], [4, 2, 4])

	def test_invalid_regions_get_nothing(self):
		counter = RegionCounter(np.array(["chr1"] * 3, dtype=object),
		                        np.array([-5, 10, 10]), np.array([5, 10, 5]))
		counter.add("chr1", np.arange(-10, 20))
		np.testing.assert_array_equal(counter.counts[:, 0], [0, 0, 0])

	def test_unknown_chrom_is_ignored(self):
		counter = RegionCounter(np.array(["chr1"], dtype=object), np.array([0]), np.array([10]))
		counter.add("chr9", np.array([1, 2, 3]))
		assert counter.counts.sum() == 0

	def test_columns_are_counted_separately(self):
		counter = RegionCounter(np.array(["chr1", "chr1"], dtype=object),
		                        np.array([0, 50]), np.array([100, 60]), n_columns=3)
		counter.add("chr1", np.array([55, 5, 55, 70]), np.array([2, 0, 2, 1]))
		np.testing.assert_array_equal(counter.counts, [[1, 1, 2], [0, 0, 2]])


class TestBarcodeGroups:
	def test_reads_named_columns_in_order_of_first_appearance(self, tmp_path):
		path = tmp_path / "cells.tsv"
		path.write_text("cell\tqc\tcluster\nA-1\t5\tT\nB-1\t3\tB\nC-1\t4\tT\nD-1\t2\t\n")
		groups = BarcodeGroups.from_table(str(path), "cell", "cluster")
		assert groups.names == ["T", "B"]
		np.testing.assert_array_equal(groups.columns_for(np.array(["C-1", "B-1", "D-1", "Z-9"])),
		                              [0, 1, -1, -1])

	def test_gzipped_table(self, tmp_path):
		path = tmp_path / "cells.tsv.gz"
		with gzip.open(path, "wt") as fh:
			fh.write("barcode\tgroup\nA-1\tX\n")
		assert BarcodeGroups.from_table(str(path)).names == ["X"]

	def test_missing_column_names_the_available_ones(self, tmp_path):
		path = tmp_path / "cells.tsv"
		path.write_text("cell\tcluster\nA-1\tT\n")
		with pytest.raises(ValueError, match="--barcode-column"):
			BarcodeGroups.from_table(str(path))

	def test_barcode_in_two_groups_rejected(self, tmp_path):
		path = tmp_path / "cells.tsv"
		path.write_text("barcode\tgroup\nA-1\tT\nA-1\tB\n")
		with pytest.raises(ValueError, match="more than one group"):
			BarcodeGroups.from_table(str(path))

	def test_repeated_identical_rows_are_fine(self, tmp_path):
		path = tmp_path / "cells.tsv"
		path.write_text("barcode\tgroup\nA-1\tT\nA-1\tT\n")
		assert BarcodeGroups.from_table(str(path)).names == ["T"]

	def test_no_grouped_barcodes_rejected(self, tmp_path):
		path = tmp_path / "cells.tsv"
		path.write_text("barcode\tgroup\nA-1\t\n")
		with pytest.raises(ValueError, match="no barcodes"):
			BarcodeGroups.from_table(str(path))


class TestCountFragments:
	@pytest.mark.parametrize("suffix", [".tsv.gz", ".tsv"])
	@pytest.mark.parametrize("shift", [(0, 0), (4, -5)])
	def test_matches_brute_force(self, tmp_path, suffix, shift):
		rng = np.random.default_rng(0)
		rows, regions = _random_fragments(rng), _random_regions(rng)
		path = _write_fragments(tmp_path / f"frags{suffix}", rows)
		counts, observed = count_fragments(str(path), *_arrays(regions), *shift, chunksize=37)
		np.testing.assert_array_equal(counts, _fragment_reference_counts(rows, regions, *shift))
		assert observed == {"chr1", "chr2"}

	def test_each_line_counts_once_regardless_of_duplicate_column(self, tmp_path):
		path = _write_fragments(tmp_path / "f.tsv", [("chr1", 100, 200, "A-1", 50)])
		counts, _ = count_fragments(str(path), *_arrays([("chr1", 0, 1000)]))
		assert counts[0, 0] == 2

	def test_fragment_ends_are_start_and_end_minus_one(self, tmp_path):
		path = _write_fragments(tmp_path / "f.tsv", [("chr1", 100, 200, "A-1", 1)])
		regions = [("chr1", 100, 101), ("chr1", 199, 200), ("chr1", 200, 201), ("chr1", 101, 199)]
		counts, _ = count_fragments(str(path), *_arrays(regions))
		np.testing.assert_array_equal(counts[:, 0], [1, 1, 0, 0])

	def test_groups_match_brute_force(self, tmp_path):
		rng = np.random.default_rng(1)
		rows, regions = _random_fragments(rng), _random_regions(rng)
		path = _write_fragments(tmp_path / "f.tsv.gz", rows)
		table = tmp_path / "cells.tsv"
		table.write_text("barcode\tgroup\nGGG-2\tB\nAAA-1\tA\nTTT-2\tA\n")    # CCC-1 unlisted
		groups = BarcodeGroups.from_table(str(table))
		counts, _ = count_fragments(str(path), *_arrays(regions), groups=groups, chunksize=53)
		expected = _fragment_reference_counts(rows, regions, group_of={"GGG-2": 0, "AAA-1": 1, "TTT-2": 1},
		                                      n_groups=2)
		np.testing.assert_array_equal(counts, expected)

	def test_file_without_header_lines(self, tmp_path):
		rng = np.random.default_rng(2)
		rows, regions = _random_fragments(rng, n=50), _random_regions(rng, n=10)
		path = _write_fragments(tmp_path / "f.tsv", rows, header=False)
		counts, _ = count_fragments(str(path), *_arrays(regions))
		np.testing.assert_array_equal(counts, _fragment_reference_counts(rows, regions))


class TestCountBam:
	@pytest.fixture
	def reads(self):
		return _random_reads(np.random.default_rng(0))

	@pytest.fixture
	def regions(self):
		return _random_regions(np.random.default_rng(1))

	@pytest.mark.parametrize("kwargs", [
	    {},
	    {"pos_shift": 4, "neg_shift": -5},
	    {"min_mapq": 0},
	    {"min_mapq": 50},
	])
	def test_default_filters_match_brute_force(self, tmp_path, reads, regions, kwargs):
		path = _write_alignments(tmp_path / "a.bam", reads)
		counts = count_bam(str(path), *_arrays(regions), **kwargs)
		np.testing.assert_array_equal(counts[:, 0], _bam_reference_counts(reads, regions, **kwargs))

	@pytest.mark.parametrize("name", sorted(FLAG_BITS))
	def test_each_flag_can_be_included(self, tmp_path, reads, regions, name):
		path = _write_alignments(tmp_path / "a.bam", reads)
		skip = sum(bit for n, bit in FLAG_BITS.items() if n != name)
		counts = count_bam(str(path), *_arrays(regions), skip_flags=skip)
		expected = _bam_reference_counts(reads, regions, include=(name,))
		np.testing.assert_array_equal(counts[:, 0], expected)
		assert expected.sum() > _bam_reference_counts(reads, regions).sum()

	def test_unfiltered_counts_every_mapped_read(self, tmp_path, reads, regions):
		path = _write_alignments(tmp_path / "a.bam", reads)
		counts = count_bam(str(path), *_arrays(regions), min_mapq=0, skip_flags=0)
		expected = _bam_reference_counts(reads, regions, min_mapq=0, include=tuple(FLAG_BITS))
		np.testing.assert_array_equal(counts[:, 0], expected)

	def test_per_contig_sum_equals_whole_file(self, tmp_path, reads, regions):
		path = _write_alignments(tmp_path / "a.bam", reads)
		arrays = _arrays(regions)
		whole = count_bam(str(path), *arrays)
		per_contig = sum(count_bam(str(path), *arrays, contig=name) for name, _ in REFS)
		np.testing.assert_array_equal(whole, per_contig)

	def test_sam_input(self, tmp_path, reads, regions):
		path = _write_alignments(tmp_path / "a.sam", reads, index=False)
		counts = count_bam(str(path), *_arrays(regions))
		np.testing.assert_array_equal(counts[:, 0], _bam_reference_counts(reads, regions))

	def test_small_batches_match(self, tmp_path, reads, regions):
		path = _write_alignments(tmp_path / "a.bam", reads)
		arrays = _arrays(regions)
		np.testing.assert_array_equal(count_bam(str(path), *arrays, batch=7), count_bam(str(path), *arrays))

	def test_corrupt_records_raise_value_error(self, tmp_path, reads, regions):
		"""A file whose header opens but whose records fail to decode (here,
		garbage in the middle of a BAM whose end-of-file marker is intact)
		surfaces as ValueError, not a raw OSError."""
		path = _write_alignments(tmp_path / "a.bam", reads, index=False)
		data = path.read_bytes()
		mid = len(data) // 2
		path.write_bytes(data[:mid] + b"\x00" * 64 + data[mid + 64:])
		with pytest.raises(ValueError, match="could not read"):
			count_bam(str(path), *_arrays(regions))

	def test_unreadable_file_raises_value_error(self, tmp_path):
		bad = tmp_path / "bad.bam"
		bad.write_text("not a bam")
		with pytest.raises(ValueError, match="could not open BAM"):
			count_bam(str(bad), *_arrays([("chr1", 0, 10)]))


class TestCountCram:
	"""CRAM is read without decoding sequences, so counts must match the BAM
	path exactly and must not need the reference FASTA."""

	@pytest.fixture
	def reads(self):
		return _random_reads(np.random.default_rng(10))

	@pytest.fixture
	def regions(self):
		return _random_regions(np.random.default_rng(11))

	@pytest.fixture
	def cram(self, tmp_path, reads, monkeypatch):
		"""An indexed CRAM whose reference is deleted after writing, with
		htslib's reference lookup (REF_PATH, REF_CACHE) disabled."""
		ref = _write_reference(tmp_path / "ref.fa")
		path = _write_alignments(tmp_path / "a.cram", reads, reference=ref)
		for f in (ref, tmp_path / "ref.fa.fai"):
			f.unlink()
		monkeypatch.setenv("REF_PATH", ":")
		monkeypatch.setenv("REF_CACHE", str(tmp_path / "no_cache"))
		return path

	@pytest.mark.parametrize("kwargs", [
	    {},
	    {"pos_shift": 4, "neg_shift": -5},
	    {"pos_shift": 0, "neg_shift": -9},
	    {"min_mapq": 0, "skip_flags": 0},
	])
	def test_matches_brute_force_without_reference(self, cram, reads, regions, kwargs):
		counts = count_bam(str(cram), *_arrays(regions), **kwargs)
		ref_kwargs = {k: v for k, v in kwargs.items() if k != "skip_flags"}
		if "skip_flags" in kwargs:
			ref_kwargs["include"] = tuple(FLAG_BITS)
		np.testing.assert_array_equal(counts[:, 0], _bam_reference_counts(reads, regions, **ref_kwargs))

	def test_indexed_per_contig_sum_equals_whole_file(self, cram, regions):
		arrays = _arrays(regions)
		assert bam_has_index(str(cram))
		whole = count_bam(str(cram), *arrays)
		np.testing.assert_array_equal(whole, sum(count_bam(str(cram), *arrays, contig=n) for n, _ in REFS))

	def test_unindexed_cram(self, tmp_path, reads, regions):
		ref = _write_reference(tmp_path / "ref.fa")
		path = _write_alignments(tmp_path / "a.cram", reads, index=False, reference=ref)
		assert not bam_has_index(str(path))
		counts = count_bam(str(path), *_arrays(regions))
		np.testing.assert_array_equal(counts[:, 0], _bam_reference_counts(reads, regions))

	def test_chrom_lengths_from_header(self, cram):
		assert bam_chrom_lengths(str(cram)) == dict(REFS)


class TestCountIssues:
	def test_with_lengths(self):
		regions = [("chr1", 0, 10), ("chrX", 0, 10), ("chr1", 5, 5), ("chr1", 4990, 5010)]
		issues, bad = count_issues(*_arrays(regions), dict(REFS))
		assert issues == {"missing_chrom": "chrX", "invalid_region": "chr1", "out_of_bounds": "chr1"}
		np.testing.assert_array_equal(bad, [False, True, True, True])

	def test_without_lengths_uses_observed_chroms_and_skips_bounds(self):
		regions = [("chr1", 0, 10**9), ("chrX", 0, 10)]
		issues, bad = count_issues(*_arrays(regions), None, {"chr1"})
		assert issues == {"missing_chrom": "chrX"}
		np.testing.assert_array_equal(bad, [False, True])


##


def _write_bed(path, regions):
	path.write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in regions))
	return path


def _read_output(path):
	with open(path) as fh:
		header = fh.readline()
	return header, pd.read_csv(path, sep="\t", comment="#")


class TestExtractCLI:
	def test_bams_end_to_end(self, tmp_path):
		rng = np.random.default_rng(3)
		regions = _random_regions(rng)
		reads_a, reads_b = _random_reads(rng), _random_reads(rng)
		a = _write_alignments(tmp_path / "A.bam", reads_a)
		b = _write_alignments(tmp_path / "B.bam", reads_b, index=False)
		bed = _write_bed(tmp_path / "r.bed", regions)
		out = tmp_path / "out.tsv"
		assert main(["extract", "-a", str(a), str(b), "-b", str(bed), "-o", str(out),
		             "-ps", "4", "-ns", "-5", "-j", "2"]) == 0
		header, df = _read_output(out)
		assert header == "# fertilizer-extract stat=count source=bam pos_shift=4 neg_shift=-5\n"
		np.testing.assert_array_equal(df["A"], _bam_reference_counts(reads_a, regions, 4, -5))
		np.testing.assert_array_equal(df["B"], _bam_reference_counts(reads_b, regions, 4, -5))

	def test_cram_end_to_end(self, tmp_path):
		rng = np.random.default_rng(12)
		regions, reads = _random_regions(rng), _random_reads(rng)
		ref = _write_reference(tmp_path / "ref.fa")
		cram = _write_alignments(tmp_path / "S.cram", reads, reference=ref)
		bed = _write_bed(tmp_path / "r.bed", regions)
		out = tmp_path / "out.tsv"
		assert main(["extract", "-a", str(cram), "-b", str(bed), "-o", str(out),
		             "-ps", "4", "-ns", "-5", "-j", "2"]) == 0
		header, df = _read_output(out)
		assert header.startswith("# fertilizer-extract stat=count source=bam")
		np.testing.assert_array_equal(df["S"], _bam_reference_counts(reads, regions, 4, -5))

	def test_bam_filter_flags(self, tmp_path):
		rng = np.random.default_rng(4)
		regions, reads = _random_regions(rng), _random_reads(rng)
		a = _write_alignments(tmp_path / "A.bam", reads)
		bed = _write_bed(tmp_path / "r.bed", regions)
		out = tmp_path / "out.tsv"
		assert main(["extract", "-a", str(a), "-b", str(bed), "-o", str(out), "-j", "1",
		             "--min-mapq", "10", "--include-flagged", "duplicate", "qcfail"]) == 0
		expected = _bam_reference_counts(reads, regions, min_mapq=10, include=("duplicate", "qcfail"))
		np.testing.assert_array_equal(_read_output(out)[1]["A"], expected)

	def test_fragments_end_to_end_with_groups(self, tmp_path):
		rng = np.random.default_rng(5)
		regions = _random_regions(rng)
		rows1, rows2 = _random_fragments(rng), _random_fragments(rng)
		f1 = _write_fragments(tmp_path / "lib1.tsv.gz", rows1)
		f2 = _write_fragments(tmp_path / "lib2.tsv.gz", rows2)
		table = tmp_path / "cells.tsv"
		table.write_text("cell\tcluster\nAAA-1\tT\nCCC-1\tB\nGGG-2\tT\n")
		bed = _write_bed(tmp_path / "r.bed", regions)
		out = tmp_path / "out.tsv"
		assert main(["extract", "-f", str(f1), str(f2), "-g", str(table), "--barcode-column", "cell",
		             "--group-column", "cluster", "-b", str(bed), "-o", str(out), "-j", "2"]) == 0
		header, df = _read_output(out)
		assert header.startswith("# fertilizer-extract stat=count source=fragments")
		group_of = {"AAA-1": 0, "CCC-1": 1, "GGG-2": 0}
		expected = (_fragment_reference_counts(rows1, regions, group_of=group_of, n_groups=2)
		            + _fragment_reference_counts(rows2, regions, group_of=group_of, n_groups=2))
		np.testing.assert_array_equal(df[["T", "B"]].to_numpy(), expected)

	def test_fragments_one_column_per_file(self, tmp_path):
		rng = np.random.default_rng(6)
		regions = _random_regions(rng)
		rows1, rows2 = _random_fragments(rng), _random_fragments(rng)
		f1 = _write_fragments(tmp_path / "C1.fragments.tsv.gz", rows1)
		f2 = _write_fragments(tmp_path / "C2.fragments.tsv.gz", rows2)
		bed = _write_bed(tmp_path / "r.bed", regions)
		out = tmp_path / "out.tsv"
		assert main(["extract", "-f", str(f1), str(f2), "-b", str(bed), "-o", str(out), "-j", "1"]) == 0
		df = _read_output(out)[1]
		np.testing.assert_array_equal(df["C1.fragments"], _fragment_reference_counts(rows1, regions)[:, 0])
		np.testing.assert_array_equal(df["C2.fragments"], _fragment_reference_counts(rows2, regions)[:, 0])

	def test_output_preserves_input_order(self, tmp_path):
		f = _write_fragments(tmp_path / "f.tsv", [("chr1", 100, 200, "A-1", 1), ("chr2", 10, 20, "A-1", 1)])
		bed = _write_bed(tmp_path / "r.bed", [("chr2", 0, 100), ("chr1", 0, 150), ("chr1", 150, 300)])
		out = tmp_path / "out.tsv"
		assert main(["extract", "-f", str(f), "-b", str(bed), "-o", str(out), "-n", "x"]) == 0
		np.testing.assert_array_equal(_read_output(out)[1]["x"], [2, 1, 1])

	def test_locus_issues_warn_and_zero(self, tmp_path):
		reads = [(0, 100, "50M", 0, 60), (1, 100, "50M", 0, 60)]
		a = _write_alignments(tmp_path / "A.bam", reads)
		bed = _write_bed(tmp_path / "r.bed", [("chr1", 0, 200), ("chrX", 0, 10), ("chr2", 50, 3500)])
		out = tmp_path / "out.tsv"
		with pytest.warns(FertilizerWarning) as rec:
			assert main(["extract", "-a", str(a), "-b", str(bed), "-o", str(out), "-j", "1"]) == 0
		msgs = " ".join(str(w.message) for w in rec)
		assert "missing from BAM" in msgs and "chromosome length of BAM" in msgs
		np.testing.assert_array_equal(_read_output(out)[1]["A"], [1, 0, 0])

	def test_fragment_chrom_not_in_file_warns(self, tmp_path):
		f = _write_fragments(tmp_path / "f.tsv", [("chr1", 100, 200, "A-1", 1)])
		bed = _write_bed(tmp_path / "r.bed", [("chr1", 0, 300), ("1", 0, 300)])
		out = tmp_path / "out.tsv"
		with pytest.warns(FertilizerWarning, match="missing from fragment file"):
			assert main(["extract", "-f", str(f), "-b", str(bed), "-o", str(out)]) == 0

	def test_counts_feed_enrich(self, tmp_path):
		rng = np.random.default_rng(7)
		n = 300
		regions = [("chr1", s, s + 10) for s in range(0, 10 * n, 10)]
		rows = []
		for cond_barcode in ("AAA-1", "CCC-1", "GGG-2"):
			for s in rng.integers(0, 10 * n - 30, size=6000):
				rows.append(("chr1", int(s), int(s) + 25, cond_barcode, 1))
		f = _write_fragments(tmp_path / "f.tsv.gz", rows)
		table = tmp_path / "cells.tsv"
		table.write_text("barcode\tgroup\nAAA-1\tA\nCCC-1\tB\nGGG-2\tC\n")
		bed = _write_bed(tmp_path / "r.bed", regions)
		counts = tmp_path / "counts.tsv"
		assert main(["extract", "-f", str(f), "-g", str(table), "-b", str(bed), "-o", str(counts)]) == 0
		out = tmp_path / "enrich.tsv"
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			assert main(["enrich", "-i", str(counts), "-c", "A", "B", "C", "-o", str(out),
			             "--q-threshold", "1.0"]) == 0
		assert len(pd.read_csv(out, sep="\t")) == n

	@pytest.mark.parametrize("argv,message", [
	    (["-w", "x.bw", "-ps", "4"], "--pos-shift"),
	    (["-w", "x.bw", "--min-mapq", "5"], "--min-mapq"),
	    (["-w", "x.bw", "-g", "t.tsv"], "--groups"),
	    (["-a", "x.bam", "-g", "t.tsv"], "--groups"),
	    (["-f", "x.tsv", "--include-flagged", "duplicate"], "--include-flagged"),
	    (["-f", "x.tsv", "-s", "sum"], "--stat applies to bigWig"),
	    (["-f", "x.tsv", "--group-column", "c"], "require --groups"),
	    (["-f", "x.tsv", "-g", "t.tsv", "-n", "a"], "--names cannot"),
	    (["-a", "x.bam", "--min-mapq", "-1"], "--min-mapq must"),
	])
	def test_options_rejected_for_the_wrong_input(self, tmp_path, capsys, argv, message):
		bed = _write_bed(tmp_path / "r.bed", [("chr1", 0, 10)])
		assert main(["extract", *argv, "-b", str(bed), "-o", str(tmp_path / "o.tsv")]) == 2
		assert message in capsys.readouterr().err

	def test_mixing_input_kinds_rejected(self, tmp_path):
		bed = _write_bed(tmp_path / "r.bed", [("chr1", 0, 10)])
		with pytest.raises(SystemExit):
			main(["extract", "-w", "x.bw", "-a", "y.bam", "-b", str(bed), "-o", str(tmp_path / "o.tsv")])

	def test_bigwig_header_unchanged(self, tmp_path):
		bw_path = tmp_path / "A.bw"
		bw = pyBigWig.open(str(bw_path), "w")
		bw.addHeader([("chr1", 1000)])
		bw.addEntries(["chr1"], [0], ends=[1000], values=[2.0])
		bw.close()
		bed = _write_bed(tmp_path / "r.bed", [("chr1", 0, 100)])
		out = tmp_path / "o.tsv"
		assert main(["extract", "-w", str(bw_path), "-b", str(bed), "-o", str(out)]) == 0
		assert _read_output(out)[0] == "# fertilizer-extract stat=mean\n"


##


def _bgzf_block(data):
	"""One BGZF block holding `data` (raw deflate with the BGZF header)."""
	comp = zlib.compressobj(6, zlib.DEFLATED, -15)
	deflated = comp.compress(data) + comp.flush()
	bsize = 18 + len(deflated) + 8 - 1
	header = b"\x1f\x8b\x08\x04" + b"\x00" * 4 + b"\x00\xff" + struct.pack("<H", 6) + b"BC" \
	    + struct.pack("<H", 2) + struct.pack("<H", bsize)
	return header + deflated + struct.pack("<II", zlib.crc32(data), len(data))


_BGZF_EOF = _bgzf_block(b"")


def _write_bgzf(path, text, cuts):
	"""Write `text` as BGZF with block boundaries at the text offsets `cuts`;
	returns the compressed offset where each block starts."""
	bounds = [0, *sorted(cuts), len(text)]
	offsets, out = [], b""
	for a, b in zip(bounds[:-1], bounds[1:], strict=True):
		offsets.append(len(out))
		out += _bgzf_block(text[a:b])
	path.write_bytes(out + _BGZF_EOF)
	return offsets


def _fragment_text(rng, n=40, header=True):
	rows = _random_fragments(rng, n=n)
	lines = (["# id=test", "# primary_contig=chr1"] if header else []) + ["\t".join(map(str, r)) for r in rows]
	return rows, ("\n".join(lines) + "\n").encode()


def _split_counts(path, regions, ranges, **kwargs):
	total, observed = None, set()
	for r in ranges:
		counts, obs = count_fragments(str(path), *_arrays(regions), byte_range=r, chunk_bytes=64, **kwargs)
		total = counts if total is None else total + counts
		observed |= obs
	return total, observed


class TestFragmentRanges:
	@pytest.fixture
	def data(self):
		rng = np.random.default_rng(20)
		rows, text = _fragment_text(rng)
		return rows, text, _random_regions(rng)

	def test_bgzf_block_boundary_at_every_line_edge(self, tmp_path, data):
		"""Two-block BGZF files cut just before, at and after every newline,
		plus other offsets: the two ranges together must count every line
		exactly once."""
		rows, text, regions = data
		expected = _fragment_reference_counts(rows, regions)
		newlines = [i for i, ch in enumerate(text) if ch == ord("\n")]
		cuts = sorted({c for nl in newlines for c in (nl - 1, nl, nl + 1, nl + 2) if 0 < c < len(text)}
		              | {1, 2, 5, len(text) - 1})
		for cut in cuts:
			path = tmp_path / "f.tsv.gz"
			offsets = _write_bgzf(path, text, [cut])
			size = path.stat().st_size
			counts, observed = _split_counts(path, regions, [(0, offsets[1]), (offsets[1], size)])
			np.testing.assert_array_equal(counts, expected, err_msg=f"cut at text offset {cut}")
			assert observed == {"chr1", "chr2"}

	def test_plain_text_split_at_every_byte(self, tmp_path):
		rng = np.random.default_rng(21)
		rows, text = _fragment_text(rng, n=12)
		regions = _random_regions(rng, n=20)
		path = tmp_path / "f.tsv"
		path.write_bytes(text)
		expected = _fragment_reference_counts(rows, regions)
		for k in range(1, len(text)):
			counts, _ = _split_counts(path, regions, [(0, k), (k, len(text))])
			np.testing.assert_array_equal(counts, expected, err_msg=f"split at byte {k}")

	def test_range_without_a_line_start(self, tmp_path, data):
		"""Blocks 2 and 3 lie inside one line, so their ranges own nothing and
		the first range finishes the line across them."""
		rows, text, regions = data
		first_nl = text.index(b"\n", 40)
		second_nl = text.index(b"\n", first_nl + 1)
		line_start = first_nl + 1
		cuts = [line_start + 3, line_start + 8, second_nl - 2]
		path = tmp_path / "f.tsv.gz"
		offsets = _write_bgzf(path, text, cuts)
		size = path.stat().st_size
		ranges = list(zip(offsets, [*offsets[1:], size], strict=True))
		counts, _ = _split_counts(path, regions, ranges)
		np.testing.assert_array_equal(counts, _fragment_reference_counts(rows, regions))
		for r in ranges[1:3]:
			assert count_fragments(str(path), *_arrays(regions), byte_range=r)[0].sum() == 0

	def test_no_trailing_newline(self, tmp_path, data):
		rows, text, regions = data
		text = text.rstrip(b"\n")
		path = tmp_path / "f.tsv.gz"
		offsets = _write_bgzf(path, text, [len(text) // 3, 2 * len(text) // 3])
		size = path.stat().st_size
		counts, _ = _split_counts(path, regions, list(zip(offsets, [*offsets[1:], size], strict=True)))
		np.testing.assert_array_equal(counts, _fragment_reference_counts(rows, regions))

	@pytest.mark.parametrize("n", [2, 3, 5, 17, 100])
	def test_fragment_ranges_cover_the_file_at_block_starts(self, tmp_path, data, n):
		rows, text, regions = data
		path = tmp_path / "f.tsv.gz"
		offsets = _write_bgzf(path, text, list(range(37, len(text), 53)))
		size = path.stat().st_size
		ranges = fragment_ranges(str(path), n, min_bytes=1)
		assert ranges[0][0] == 0 and ranges[-1][1] == size
		assert all(a < b for a, b in ranges)
		assert all(b == c for (_, b), (c, _) in zip(ranges[:-1], ranges[1:], strict=True))
		assert {a for a, _ in ranges} <= {*offsets, size - len(_BGZF_EOF)}     # block starts, incl. EOF block
		assert len(ranges) <= n
		counts, _ = _split_counts(path, regions, ranges)
		np.testing.assert_array_equal(counts, _fragment_reference_counts(rows, regions))

	def test_real_bgzip_output_with_groups(self, tmp_path):
		"""A file compressed by htslib (pysam.tabix_compress) rather than the
		test writer, counted per barcode group across many ranges."""
		rng = np.random.default_rng(22)
		rows = _random_fragments(rng, n=3000)
		regions = _random_regions(rng)
		plain = _write_fragments(tmp_path / "f.tsv", rows)
		path = tmp_path / "f.tsv.gz"
		pysam.tabix_compress(str(plain), str(path))
		table = tmp_path / "cells.tsv"
		table.write_text("barcode\tgroup\nGGG-2\tB\nAAA-1\tA\nTTT-2\tA\n")
		groups = BarcodeGroups.from_table(str(table))
		ranges = fragment_ranges(str(path), 8, min_bytes=1)
		counts, _ = _split_counts(path, regions, ranges, groups=groups)
		expected = _fragment_reference_counts(rows, regions, group_of={"GGG-2": 0, "AAA-1": 1, "TTT-2": 1},
		                                      n_groups=2)
		np.testing.assert_array_equal(counts, expected)
		stream, _ = count_fragments(str(path), *_arrays(regions), groups=groups)
		np.testing.assert_array_equal(counts, stream)

	def test_plain_gzip_and_small_files_are_not_split(self, tmp_path, data):
		_, text, _ = data
		path = tmp_path / "f.tsv.gz"
		with gzip.open(path, "wb") as fh:
			fh.write(text)
		assert fragment_ranges(str(path), 8, min_bytes=1) is None
		bgzf = tmp_path / "g.tsv.gz"
		_write_bgzf(bgzf, text, [100])
		assert fragment_ranges(str(bgzf), 8) is None            # below the 16 MB default
		assert fragment_ranges(str(bgzf), 1, min_bytes=1) is None

	def test_corrupt_block_raises_value_error(self, tmp_path, data):
		_, text, regions = data
		path = tmp_path / "f.tsv.gz"
		offsets = _write_bgzf(path, text, [200, 400])
		raw = bytearray(path.read_bytes())
		raw[offsets[1] + 20:offsets[1] + 30] = b"\xff" * 10
		path.write_bytes(bytes(raw))
		with pytest.raises((ValueError, zlib.error)):
			count_fragments(str(path), *_arrays(regions), byte_range=(offsets[1], path.stat().st_size))

	@pytest.mark.parametrize("n_jobs", ["1", "2", "3", "7"])
	def test_cli_splits_one_file_across_workers(self, tmp_path, monkeypatch, n_jobs):
		import fertilizer.counting as counting_module

		real = counting_module.fragment_ranges
		monkeypatch.setattr(counting_module, "fragment_ranges",
		                    lambda path, n, min_bytes=None: real(path, n, min_bytes=1))
		rng = np.random.default_rng(23)
		rows = _random_fragments(rng, n=2000)
		regions = _random_regions(rng)
		plain = _write_fragments(tmp_path / "f.tsv", rows)
		path = tmp_path / "f.tsv.gz"
		pysam.tabix_compress(str(plain), str(path))
		bed = _write_bed(tmp_path / "r.bed", regions)
		out = tmp_path / "out.tsv"
		assert main(["extract", "-f", str(path), "-b", str(bed), "-o", str(out), "-n", "x", "-j", n_jobs]) == 0
		np.testing.assert_array_equal(_read_output(out)[1]["x"], _fragment_reference_counts(rows, regions)[:, 0])


class TestContigPieces:
	@pytest.fixture
	def reads(self):
		return _random_reads(np.random.default_rng(30), n=600)

	@pytest.fixture
	def regions(self):
		return _random_regions(np.random.default_rng(31))

	@pytest.mark.parametrize("n", [1, 2, 3, 8, 50])
	def test_contig_ranges_cover_each_chromosome(self, n):
		lengths = {"chr1": 5000, "chr2": 3000, "chrM": 16}
		ranges = contig_ranges(lengths, n)
		for contig, length in lengths.items():
			pieces = [(a, b) for c, a, b in ranges if c == contig]
			if pieces == [(None, None)]:
				continue
			assert pieces[0][0] == 0 and pieces[-1][1] == length
			assert all(b == c for (_, b), (c, _) in zip(pieces[:-1], pieces[1:], strict=True))
			assert all(a < b for a, b in pieces)
		assert ("chrM", None, None) in ranges or n == 1 or len(ranges) >= n

	@pytest.mark.parametrize("fmt", ["bam", "cram"])
	@pytest.mark.parametrize("n", [2, 5, 13, 50])
	@pytest.mark.parametrize("kwargs", [{}, {"pos_shift": 4, "neg_shift": -5},
	                                    {"pos_shift": -100, "neg_shift": 100, "min_mapq": 0, "skip_flags": 0}])
	def test_pieces_sum_to_whole_file(self, tmp_path, reads, regions, fmt, n, kwargs):
		ref = _write_reference(tmp_path / "ref.fa") if fmt == "cram" else None
		path = _write_alignments(tmp_path / f"a.{fmt}", reads, reference=ref)
		arrays = _arrays(regions)
		whole = count_bam(str(path), *arrays, **kwargs)
		pieced = np.zeros_like(whole)
		for contig, lo, hi in contig_ranges(dict(REFS), n):
			pieced += count_bam(str(path), *arrays, contig=contig, start=lo, stop=hi, **kwargs)
		np.testing.assert_array_equal(pieced, whole)

	def test_cuts_at_read_starts(self, tmp_path, reads, regions):
		"""Piece boundaries exactly at, just before and just after read starts,
		including reads with a 100 bp N skip that span several pieces."""
		path = _write_alignments(tmp_path / "a.bam", reads)
		arrays = _arrays(regions)
		whole = count_bam(str(path), *arrays, min_mapq=0, skip_flags=0)
		starts_chr1 = sorted({s for ref, s, *_ in reads if ref == 0})[::7]
		for s in starts_chr1:
			for cut in (s - 1, s, s + 1):
				if not 0 < cut < REFS[0][1]:
					continue
				pieced = (count_bam(str(path), *arrays, min_mapq=0, skip_flags=0, contig="chr1", start=0, stop=cut)
				          + count_bam(str(path), *arrays, min_mapq=0, skip_flags=0, contig="chr1",
				                      start=cut, stop=REFS[0][1])
				          + count_bam(str(path), *arrays, min_mapq=0, skip_flags=0, contig="chr2"))
				np.testing.assert_array_equal(pieced, whole, err_msg=f"cut at {cut}")

	@pytest.mark.parametrize("n_jobs", ["1", "2", "3", "7"])
	@pytest.mark.parametrize("fmt", ["bam", "cram"])
	def test_cli_split_matches_brute_force(self, tmp_path, reads, regions, n_jobs, fmt):
		ref = _write_reference(tmp_path / "ref.fa") if fmt == "cram" else None
		path = _write_alignments(tmp_path / f"S.{fmt}", reads, reference=ref)
		bed = _write_bed(tmp_path / "r.bed", regions)
		out = tmp_path / "out.tsv"
		assert main(["extract", "-a", str(path), "-b", str(bed), "-o", str(out), "-j", n_jobs,
		             "-ps", "4", "-ns", "-5"]) == 0
		np.testing.assert_array_equal(_read_output(out)[1]["S"], _bam_reference_counts(reads, regions, 4, -5))
