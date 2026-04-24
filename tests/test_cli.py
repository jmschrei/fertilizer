"""Tests for the fertilizer CLI and aggregation helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pyBigWig
import pytest

from fertilizer.cli import _build_parser
from fertilizer.extract import (
    FertilizerWarning,
    _chunk_slices,
    bigwig_region_means,
    load_regions,
    run,
)


def _write_bed(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")


def _make_bw(path, header, entries=None):
    """Create a bigWig at `path`; entries is (chroms, starts, ends, values) or None."""
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader(header)
    if entries is not None:
        chroms, starts, ends, values = entries
        bw.addEntries(chroms, starts, ends=ends, values=values)
    bw.close()
    return path


def _make_args(bigwigs, beds, out, n_jobs=1, stat=None):
    argv = ["extract", "-w", *map(str, bigwigs), "-b", *map(str, beds),
            "-o", str(out), "-j", str(n_jobs)]
    if stat is not None:
        argv += ["-s", stat]
    return _build_parser().parse_args(argv)


@pytest.fixture
def dense_bw(tmp_path):
    """chr1 (1000bp): [0, 500)=2.0, [500, 1000)=6.0."""
    return _make_bw(tmp_path / "dense.bw", [("chr1", 1000)],
                    (["chr1", "chr1"], [0, 500], [500, 1000], [2.0, 6.0]))


@pytest.fixture
def empty_bw(tmp_path):
    """Header-only bigWig with no signal entries."""
    return _make_bw(tmp_path / "empty.bw", [("chr1", 1000)])


@pytest.fixture
def sparse_bw(tmp_path):
    """chr1 (1000bp) with coverage only on [0, 100)=5.0."""
    return _make_bw(tmp_path / "sparse.bw", [("chr1", 1000)],
                    (["chr1"], [0], [100], [5.0]))


@pytest.fixture
def multi_sparse_bw(tmp_path):
    """Multi-chromosome bigWig with several non-zero blocks and real gaps.

        chr1 (2000 bp): [100, 300)=4.0, [800, 1000)=8.0
        chr2 (1500 bp): [500, 700)=2.0, [1000, 1200)=10.0
        chr3 (1000 bp): [0, 100)=6.0
    """
    return _make_bw(
        tmp_path / "multi_sparse.bw",
        [("chr1", 2000), ("chr2", 1500), ("chr3", 1000)],
        (["chr1", "chr1", "chr2", "chr2", "chr3"],
         [100, 800, 500, 1000, 0],
         [300, 1000, 700, 1200, 100],
         [4.0, 8.0, 2.0, 10.0, 6.0]),
    )


class TestLoadRegions:
    def test_single_bed(self, tmp_path):
        path = tmp_path / "a.bed"
        _write_bed(path, [("chr1", 0, 100), ("chr2", 5, 15)])
        df = load_regions([str(path)])
        assert list(df.columns) == ["chrom", "start", "end"]
        assert df["chrom"].tolist() == ["chr1", "chr2"]
        assert df["start"].tolist() == [0, 5]
        assert df["end"].tolist() == [100, 15]

    def test_multiple_beds_concat(self, tmp_path):
        a = tmp_path / "a.bed"
        b = tmp_path / "b.bed"
        _write_bed(a, [("chr1", 0, 100)])
        _write_bed(b, [("chr2", 200, 300), ("chr3", 400, 500)])
        df = load_regions([str(a), str(b)])
        assert len(df) == 3
        assert df["chrom"].tolist() == ["chr1", "chr2", "chr3"]

    def test_ignores_extra_columns(self, tmp_path):
        path = tmp_path / "a.bed"
        _write_bed(path, [("chr1", 0, 100, "name", "500", "+")])
        df = load_regions([str(path)])
        assert list(df.columns) == ["chrom", "start", "end"]
        assert df.iloc[0].tolist() == ["chr1", 0, 100]

    def test_numeric_chroms_stay_strings(self, tmp_path):
        path = tmp_path / "a.bed"
        _write_bed(path, [("1", 0, 100), ("2", 0, 100), ("X", 0, 100)])
        df = load_regions([str(path)])
        assert df["chrom"].dtype == object
        assert df["chrom"].tolist() == ["1", "2", "X"]

    def test_comment_lines_skipped(self, tmp_path):
        path = tmp_path / "a.bed"
        with open(path, "w") as f:
            f.write("# header comment\n")
            f.write("chr1\t0\t100\n")
            f.write("# another comment\n")
            f.write("chr2\t200\t300\n")
        df = load_regions([str(path)])
        assert df["chrom"].tolist() == ["chr1", "chr2"]

    def test_empty_bed_produces_empty_frame(self, tmp_path):
        path = tmp_path / "empty.bed"
        path.write_text("")
        df = load_regions([str(path)])
        assert len(df) == 0
        assert list(df.columns) == ["chrom", "start", "end"]

    def test_int_dtype_enforced(self, tmp_path):
        path = tmp_path / "a.bed"
        _write_bed(path, [("chr1", 0, 100)])
        df = load_regions([str(path)])
        assert df["start"].dtype == np.int64
        assert df["end"].dtype == np.int64


class TestChunkSlices:
    def test_covers_full_range(self):
        slices = _chunk_slices(100, 4)
        assert slices[0].start == 0
        assert slices[-1].stop == 100
        assert sum(sl.stop - sl.start for sl in slices) == 100

    def test_contiguous(self):
        slices = _chunk_slices(97, 5)
        for a, b in zip(slices, slices[1:]):
            assert a.stop == b.start

    def test_zero_length_input(self):
        assert _chunk_slices(0, 4) == []

    def test_more_chunks_than_elements_clamps(self):
        slices = _chunk_slices(3, 10)
        assert len(slices) == 3
        assert [(sl.start, sl.stop) for sl in slices] == [(0, 1), (1, 2), (2, 3)]


class TestBigwigRegionMeans:
    def test_dense_regions_no_issues(self, dense_bw):
        regions = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 500, 0],
            "end":   [500, 1000, 1000],
        })
        vals, issues = bigwig_region_means(regions, str(dense_bw))
        np.testing.assert_allclose(vals, [2.0, 6.0, 4.0])
        assert issues == set()

    def test_partial_overlap_is_averaged(self, dense_bw):
        regions = pd.DataFrame({"chrom": ["chr1"], "start": [400], "end": [600]})
        vals, issues = bigwig_region_means(regions, str(dense_bw))
        np.testing.assert_allclose(vals, [4.0])
        assert issues == set()

    def test_empty_bigwig_returns_zeros_without_issues(self, empty_bw):
        """No coverage is a data property, not a locus issue -> zeros, no warnings."""
        regions = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [0, 500],
            "end":   [100, 1000],
        })
        vals, issues = bigwig_region_means(regions, str(empty_bw))
        np.testing.assert_array_equal(vals, [0.0, 0.0])
        assert not np.any(np.isnan(vals))
        assert issues == set()

    def test_sparse_uncovered_region_is_zero_without_issue(self, sparse_bw):
        regions = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [0, 500],
            "end":   [100, 600],
        })
        vals, issues = bigwig_region_means(regions, str(sparse_bw))
        assert vals[0] == 5.0
        assert vals[1] == 0.0
        assert issues == set()

    def test_missing_chrom_issue(self, dense_bw):
        regions = pd.DataFrame({"chrom": ["chr2"], "start": [0], "end": [100]})
        vals, issues = bigwig_region_means(regions, str(dense_bw))
        np.testing.assert_array_equal(vals, [0.0])
        assert issues == {"missing_chrom"}

    def test_out_of_bounds_issue(self, dense_bw):
        regions = pd.DataFrame({"chrom": ["chr1"], "start": [900], "end": [2000]})
        vals, issues = bigwig_region_means(regions, str(dense_bw))
        np.testing.assert_array_equal(vals, [0.0])
        assert issues == {"out_of_bounds"}

    def test_zero_length_region_issue(self, dense_bw):
        regions = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [100]})
        vals, issues = bigwig_region_means(regions, str(dense_bw))
        np.testing.assert_array_equal(vals, [0.0])
        assert issues == {"invalid_region"}

    def test_negative_start_issue(self, dense_bw):
        regions = pd.DataFrame({"chrom": ["chr1"], "start": [-10], "end": [100]})
        vals, issues = bigwig_region_means(regions, str(dense_bw))
        np.testing.assert_array_equal(vals, [0.0])
        assert issues == {"invalid_region"}

    def test_nan_from_bigwig_is_silently_zeroed(self, monkeypatch, dense_bw):
        """A NaN returned by pyBigWig (possible for bigWigs with NaN spans)
        must become 0.0 in the output with no warning — NaN is data absence,
        not a locus problem."""
        from fertilizer import extract

        real_open = extract.pyBigWig.open

        class NaNStats:
            def __init__(self, inner):
                self._inner = inner
            def chroms(self):
                return self._inner.chroms()
            def stats(self, *args, **kwargs):
                return [float("nan")]
            def close(self):
                self._inner.close()

        monkeypatch.setattr(extract.pyBigWig, "open",
                            lambda p, *a, **k: NaNStats(real_open(p, *a, **k)))

        regions = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [500]})
        vals, issues = bigwig_region_means(regions, str(dense_bw))
        assert vals[0] == 0.0
        assert not np.any(np.isnan(vals))
        assert issues == set()

    def test_boundary_coords(self, dense_bw):
        """start=0 and end==chrom_length are valid; end==length+1 is not."""
        regions = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0,     999,    0],
            "end":   [1000,  1000,   1001],
        })
        vals, issues = bigwig_region_means(regions, str(dense_bw))
        assert vals[0] == 4.0      # full chrom, mean of 2.0 and 6.0
        assert vals[1] == 6.0      # last base of [500, 1000) block
        assert vals[2] == 0.0      # one base past end
        assert issues == {"out_of_bounds"}

    def test_mixed_valid_and_invalid_collects_all_issues(self, dense_bw):
        regions = pd.DataFrame({
            "chrom": ["chr1", "chr2", "chr1", "chr1", "chr1"],
            "start": [0,      0,      500,    950,    300],
            "end":   [500,    100,    1000,   1500,   300],
        })
        vals, issues = bigwig_region_means(regions, str(dense_bw))
        np.testing.assert_allclose(vals, [2.0, 0.0, 6.0, 0.0, 0.0])
        assert issues == {"missing_chrom", "out_of_bounds", "invalid_region"}

    def test_issue_set_is_deduplicated_within_worker(self, dense_bw):
        regions = pd.DataFrame({
            "chrom": ["chr2", "chr2", "chr3"],
            "start": [0, 0, 0],
            "end":   [100, 200, 300],
        })
        _, issues = bigwig_region_means(regions, str(dense_bw))
        assert issues == {"missing_chrom"}

    def test_preserves_region_order(self, dense_bw):
        regions = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [500, 0],
            "end":   [1000, 500],
        })
        vals, _ = bigwig_region_means(regions, str(dense_bw))
        np.testing.assert_allclose(vals, [6.0, 2.0])

class TestMultiChromSparseBigwig:
    """A multi-chromosome, sparse bigWig exercises the full matrix of
    block/gap combinations: fully-in-block, fully-in-gap, block-gap boundary
    crossings, regions spanning multiple blocks, interleaved chroms, and
    regions that cover an entire chromosome."""

    def test_regions_fully_in_blocks(self, multi_sparse_bw):
        regions = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2", "chr2", "chr3"],
            "start": [100,    800,    500,    1000,   0],
            "end":   [300,    1000,   700,    1200,   100],
        })
        vals, issues = bigwig_region_means(regions, str(multi_sparse_bw))
        np.testing.assert_allclose(vals, [4.0, 8.0, 2.0, 10.0, 6.0])
        assert issues == set()

    def test_regions_fully_in_gaps_are_zero_without_warning(self, multi_sparse_bw):
        regions = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2", "chr3"],
            "start": [400,    1500,   0,      200],
            "end":   [600,    2000,   400,    900],
        })
        vals, issues = bigwig_region_means(regions, str(multi_sparse_bw))
        np.testing.assert_array_equal(vals, [0.0, 0.0, 0.0, 0.0])
        assert issues == set()

    def test_region_spans_block_and_gap_means_over_covered(self, multi_sparse_bw):
        # pyBigWig "mean" ignores uncovered bases, so the block value is
        # returned verbatim whenever it is the only covered portion.
        regions = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2", "chr3"],
            "start": [0,      200,    400,    0],
            "end":   [200,    400,    800,    500],
        })
        vals, _ = bigwig_region_means(regions, str(multi_sparse_bw))
        np.testing.assert_allclose(vals, [4.0, 4.0, 2.0, 6.0])

    def test_region_spans_two_blocks_same_chrom_weighted_mean(self, multi_sparse_bw):
        # chr1 [100, 1000): 200 bp @ 4.0 + 500 bp gap + 200 bp @ 8.0
        #   -> covered mean = (200*4 + 200*8) / 400 = 6.0
        # chr2 [500, 1200): 200 bp @ 2.0 + 300 bp gap + 200 bp @ 10.0
        #   -> covered mean = (200*2 + 200*10) / 400 = 6.0
        regions = pd.DataFrame({
            "chrom": ["chr1", "chr2"],
            "start": [100, 500],
            "end":   [1000, 1200],
        })
        vals, _ = bigwig_region_means(regions, str(multi_sparse_bw))
        np.testing.assert_allclose(vals, [6.0, 6.0])

    def test_region_covers_full_chromosome(self, multi_sparse_bw):
        regions = pd.DataFrame({
            "chrom": ["chr1", "chr2", "chr3"],
            "start": [0, 0, 0],
            "end":   [2000, 1500, 1000],
        })
        vals, _ = bigwig_region_means(regions, str(multi_sparse_bw))
        # chr1 full: (200*4 + 200*8)/400 = 6.0
        # chr2 full: (200*2 + 200*10)/400 = 6.0
        # chr3 full: 100 bp @ 6.0 -> 6.0
        np.testing.assert_allclose(vals, [6.0, 6.0, 6.0])

    def test_interleaved_chroms(self, multi_sparse_bw):
        """Order of regions across chroms must not affect values."""
        regions = pd.DataFrame({
            "chrom": ["chr2", "chr1", "chr3", "chr1", "chr2", "chr1"],
            "start": [1000,   100,    0,      800,    500,    400],
            "end":   [1200,   300,    100,    1000,   700,    500],
        })
        vals, issues = bigwig_region_means(regions, str(multi_sparse_bw))
        np.testing.assert_allclose(vals, [10.0, 4.0, 6.0, 8.0, 2.0, 0.0])
        assert issues == set()

    def test_partial_block_overlap_returns_block_value(self, multi_sparse_bw):
        # A 10 bp region fully inside the chr1 [100, 300) block.
        regions = pd.DataFrame({
            "chrom": ["chr1"], "start": [150], "end": [160],
        })
        vals, _ = bigwig_region_means(regions, str(multi_sparse_bw))
        np.testing.assert_allclose(vals, [4.0])

    def test_missing_chrom_among_valid_multi_chrom(self, multi_sparse_bw):
        regions = pd.DataFrame({
            "chrom": ["chr1", "chrX", "chr3"],
            "start": [100,    0,      0],
            "end":   [300,    500,    100],
        })
        vals, issues = bigwig_region_means(regions, str(multi_sparse_bw))
        np.testing.assert_allclose(vals, [4.0, 0.0, 6.0])
        assert issues == {"missing_chrom"}

    def test_end_to_end_preserves_input_order(self, tmp_path, multi_sparse_bw):
        bed = tmp_path / "a.bed"
        _write_bed(bed, [
            ("chr2", 1000, 1200),    # 10.0
            ("chr1", 100, 300),      # 4.0
            ("chr3", 0, 100),        # 6.0
            ("chr1", 400, 600),      # gap -> 0.0
            ("chr2", 500, 700),      # 2.0
            ("chr1", 100, 1000),     # weighted mean -> 6.0
        ])
        out = tmp_path / "out.tsv"
        args = _make_args([multi_sparse_bw], [bed], out)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FertilizerWarning)
            run(args)

        df = pd.read_csv(out, sep="\t")
        assert df["chrom"].tolist() == ["chr2", "chr1", "chr3", "chr1", "chr2", "chr1"]
        assert df["start"].tolist() == [1000, 100, 0, 400, 500, 100]
        np.testing.assert_allclose(
            df["multi_sparse"].tolist(), [10.0, 4.0, 6.0, 0.0, 2.0, 6.0]
        )


class TestRun:
    def test_end_to_end_fills_zeros_no_nan(self, tmp_path, dense_bw, empty_bw, sparse_bw):
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 500), ("chr1", 500, 1000), ("chr2", 0, 100)])
        out = tmp_path / "out.tsv"
        args = _make_args([dense_bw, empty_bw, sparse_bw], [bed], out)

        with pytest.warns(FertilizerWarning, match="not present"):
            assert run(args) == 0

        df = pd.read_csv(out, sep="\t")
        assert list(df.columns) == ["chrom", "start", "end", "dense", "empty", "sparse"]
        assert len(df) == 3
        # chr2:0-100 is a missing-chrom locus in every bigWig -> 0.0 everywhere.
        np.testing.assert_allclose(df["dense"].tolist(), [2.0, 6.0, 0.0])
        np.testing.assert_allclose(df["empty"].tolist(), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(df["sparse"].tolist(), [5.0, 0.0, 0.0])
        assert not df[["dense", "empty", "sparse"]].isna().any().any()

    def test_no_warning_when_only_uncovered(self, tmp_path, empty_bw, sparse_bw):
        """Empty/sparse coverage of otherwise-valid loci must not warn."""
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 500), ("chr1", 500, 1000)])
        out = tmp_path / "out.tsv"
        args = _make_args([empty_bw, sparse_bw], [bed], out)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FertilizerWarning)
            assert run(args) == 0

    def test_missing_chrom_warning_emitted_once(self, tmp_path, dense_bw):
        # Many regions with missing chroms across two bigwigs -> still one warning.
        other = tmp_path / "dense2.bw"
        bw = pyBigWig.open(str(other), "w")
        bw.addHeader([("chr1", 1000)])
        bw.addEntries(["chr1"], [0], ends=[1000], values=[1.0])
        bw.close()

        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr2", 0, 100), ("chr3", 0, 100), ("chr4", 0, 100)])
        out = tmp_path / "out.tsv"
        args = _make_args([dense_bw, other], [bed], out)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", FertilizerWarning)
            run(args)
        missing = [w for w in recorded
                   if issubclass(w.category, FertilizerWarning)
                   and "not present" in str(w.message)]
        assert len(missing) == 1

    def test_each_issue_type_warns_once(self, tmp_path, dense_bw):
        bed = tmp_path / "a.bed"
        _write_bed(bed, [
            ("chr1", 0, 500),        # valid
            ("chr2", 0, 100),        # missing_chrom
            ("chr9", 0, 100),        # missing_chrom (duplicate type)
            ("chr1", 900, 2000),     # out_of_bounds
            ("chr1", -1, 10),        # invalid_region (negative start)
            ("chr1", 100, 100),      # invalid_region (zero-length; duplicate type)
        ])
        out = tmp_path / "out.tsv"
        args = _make_args([dense_bw], [bed], out)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", FertilizerWarning)
            run(args)

        categories = [
            str(w.message) for w in recorded
            if issubclass(w.category, FertilizerWarning)
        ]
        assert len(categories) == 3
        assert len(set(categories)) == 3

    def test_duplicate_bigwig_stems_raises(self, tmp_path, dense_bw):
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 500)])
        out = tmp_path / "out.tsv"
        args = _make_args([dense_bw, dense_bw], [bed], out)
        with pytest.raises(ValueError, match="duplicate"):
            run(args)

    def test_numeric_chrom_bigwig_matches_numeric_bed(self, tmp_path):
        """Regression: chrom "1" in BED must match chrom "1" in bigWig."""
        bw_path = tmp_path / "numeric.bw"
        bw = pyBigWig.open(str(bw_path), "w")
        bw.addHeader([("1", 1000), ("2", 1000)])
        bw.addEntries(["1", "2"], [0, 0], ends=[1000, 1000], values=[3.0, 9.0])
        bw.close()

        bed = tmp_path / "a.bed"
        _write_bed(bed, [("1", 0, 500), ("2", 0, 500)])
        out = tmp_path / "out.tsv"
        args = _make_args([bw_path], [bed], out)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FertilizerWarning)
            run(args)

        df = pd.read_csv(out, sep="\t", dtype={"chrom": str})
        np.testing.assert_allclose(df["numeric"].tolist(), [3.0, 9.0])

    def test_output_preserves_input_order_after_sort(self, tmp_path, dense_bw):
        """Internal sort for bigWig locality must not change row order on output."""
        bed = tmp_path / "a.bed"
        # Deliberately out of (chrom, start) order.
        _write_bed(bed, [
            ("chr1", 500, 1000),
            ("chr1", 0, 500),
            ("chr1", 250, 750),
        ])
        out = tmp_path / "out.tsv"
        args = _make_args([dense_bw], [bed], out)
        run(args)

        df = pd.read_csv(out, sep="\t")
        assert df["start"].tolist() == [500, 0, 250]
        assert df["end"].tolist() == [1000, 500, 750]
        np.testing.assert_allclose(df["dense"].tolist(), [6.0, 2.0, 4.0])

    def test_empty_bed_produces_empty_tsv(self, tmp_path, dense_bw):
        bed = tmp_path / "empty.bed"
        bed.write_text("")
        out = tmp_path / "out.tsv"
        args = _make_args([dense_bw], [bed], out)
        run(args)
        df = pd.read_csv(out, sep="\t")
        assert list(df.columns) == ["chrom", "start", "end", "dense"]
        assert len(df) == 0

    def test_bad_n_jobs_raises(self, tmp_path, dense_bw):
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 500)])
        out = tmp_path / "out.tsv"
        args = _make_args([dense_bw], [bed], out, n_jobs=0)
        with pytest.raises(ValueError, match="n_jobs"):
            run(args)

    def test_parallel_matches_serial(self, tmp_path, dense_bw, sparse_bw):
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 500), ("chr1", 500, 1000), ("chr1", 0, 100)])
        serial_out = tmp_path / "serial.tsv"
        parallel_out = tmp_path / "parallel.tsv"
        run(_make_args([dense_bw, sparse_bw], [bed], serial_out, n_jobs=1))
        run(_make_args([dense_bw, sparse_bw], [bed], parallel_out, n_jobs=2))

        a = pd.read_csv(serial_out, sep="\t")
        b = pd.read_csv(parallel_out, sep="\t")
        pd.testing.assert_frame_equal(a, b)


class TestStat:
    """The -s/--stat flag maps to pyBigWig's `type=` argument."""

    def test_default_is_mean(self, tmp_path, dense_bw):
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 1000)])
        out = tmp_path / "out.tsv"
        run(_make_args([dense_bw], [bed], out))
        df = pd.read_csv(out, sep="\t")
        # Mean of [2.0 over 500 bp, 6.0 over 500 bp] = 4.0
        np.testing.assert_allclose(df["dense"].tolist(), [4.0])

    def test_max(self, tmp_path, dense_bw):
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 1000)])
        out = tmp_path / "out.tsv"
        run(_make_args([dense_bw], [bed], out, stat="max"))
        df = pd.read_csv(out, sep="\t")
        np.testing.assert_allclose(df["dense"].tolist(), [6.0])

    def test_min(self, tmp_path, dense_bw):
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 1000)])
        out = tmp_path / "out.tsv"
        run(_make_args([dense_bw], [bed], out, stat="min"))
        df = pd.read_csv(out, sep="\t")
        np.testing.assert_allclose(df["dense"].tolist(), [2.0])

    def test_sum(self, tmp_path, dense_bw):
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 1000)])
        out = tmp_path / "out.tsv"
        run(_make_args([dense_bw], [bed], out, stat="sum"))
        df = pd.read_csv(out, sep="\t")
        # 500*2.0 + 500*6.0 = 4000
        np.testing.assert_allclose(df["dense"].tolist(), [4000.0])

    def test_std(self, tmp_path, dense_bw):
        # dense_bw: [0,500)=2.0, [500,1000)=6.0. Over [0, 1000) the per-base
        # values are 500 copies of 2.0 and 500 copies of 6.0, mean 4.0;
        # pyBigWig returns the sample std (ddof=1): sqrt(4000/999) ≈ 2.001.
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 1000)])
        out = tmp_path / "out.tsv"
        run(_make_args([dense_bw], [bed], out, stat="std"))
        df = pd.read_csv(out, sep="\t")
        np.testing.assert_allclose(df["dense"].tolist(), [np.sqrt(4000 / 999)], rtol=1e-6)

    def test_coverage_on_sparse(self, tmp_path, sparse_bw):
        # sparse_bw covers [0,100) at 5.0 within chr1 (1000 bp).
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 1000), ("chr1", 0, 100)])
        out = tmp_path / "out.tsv"
        run(_make_args([sparse_bw], [bed], out, stat="coverage"))
        df = pd.read_csv(out, sep="\t")
        # Coverage = fraction of region with signal.
        np.testing.assert_allclose(df["sparse"].tolist(), [0.1, 1.0])

    def test_invalid_stat_rejected(self, tmp_path, dense_bw):
        bed = tmp_path / "a.bed"
        _write_bed(bed, [("chr1", 0, 1000)])
        out = tmp_path / "out.tsv"
        with pytest.raises(SystemExit):
            _make_args([dense_bw], [bed], out, stat="median")
