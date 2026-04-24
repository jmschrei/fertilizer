"""Tests for NB-GLM LRT differential analysis across >=2 conditions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fertilizer.cli import _build_parser, main
from fertilizer.diff import (
    DiffResult,
    FertilizerDiffWarning,
    bh_qvalues,
    differential_analysis,
    size_factors,
)


class TestSizeFactors:
    def test_identical_samples_give_unit_size_factors(self):
        x = np.array([1.0, 5.0, 20.0, 100.0, 500.0])
        sf = size_factors(np.column_stack([x, x, x]))
        np.testing.assert_allclose(sf, [1.0, 1.0, 1.0])

    def test_proportional_scaling_is_recovered(self):
        a = np.array([1.0, 5.0, 20.0, 100.0, 500.0])
        sf = size_factors(np.column_stack([a, a * 2.0, a * 4.0]))
        assert sf[1] / sf[0] == pytest.approx(2.0)
        assert sf[2] / sf[0] == pytest.approx(4.0)

    def test_zeros_are_ignored_but_positives_suffice(self):
        a = np.array([0.0, 5.0, 20.0, 100.0])
        b = np.array([3.0, 5.0, 20.0, 100.0])
        c = np.array([0.0, 5.0, 20.0, 100.0])
        sf = size_factors(np.column_stack([a, b, c]))
        np.testing.assert_allclose(sf, [1.0, 1.0, 1.0])

    def test_fails_when_too_few_shared_positive_loci(self):
        a = np.array([0.0, 1.0, 0.0])
        b = np.array([1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="at least 2 loci"):
            size_factors(np.column_stack([a, b]))

    def test_hand_computed_median_of_ratios(self):
        """With sample A = [10, 10, 10] and B = [20, 40, 60], the per-locus
        geometric means are sqrt(200), sqrt(400), sqrt(600). Log ratios of A
        vs these are [-0.5·log 2, -log 2, -0.5·log 6], whose median is -log 2
        → sf_A = exp(-log 2) = 0.5. Symmetrically sf_B = 2.0."""
        counts = np.array([[10.0, 20.0], [10.0, 40.0], [10.0, 60.0]])
        sf = size_factors(counts)
        np.testing.assert_allclose(sf, [0.5, 2.0], rtol=1e-12)


class TestBHQValues:
    def test_monotone_in_sorted_order(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=200)
        q = bh_qvalues(p)
        order = np.argsort(p)
        assert np.all(np.diff(q[order]) >= -1e-12)

    def test_q_at_least_p(self):
        p = np.array([0.001, 0.01, 0.1, 0.5, 0.9])
        q = bh_qvalues(p)
        assert np.all(q >= p - 1e-12)

    def test_uniform_nulls_have_fdr_controlled(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(size=10_000)
        q = bh_qvalues(p)
        assert (q < 0.05).mean() < 0.01

    def test_empty_input(self):
        assert bh_qvalues(np.array([])).shape == (0,)

    def test_hand_computed_step_up(self):
        """Against hand-computed BH q-values on a shuffled input.

        Sorted p = [0.001, 0.004, 0.02, 0.04, 0.05], n=5:
          raw q = p·n/rank = [0.005, 0.010, 1/30, 0.050, 0.050]
          step-up (non-increasing from the right) keeps them monotone, so
          q at sorted positions = [0.005, 0.010, 1/30, 0.050, 0.050].
        Shuffling the input must give the same q-values at the original
        positions (verifies the ranking bookkeeping)."""
        p = np.array([0.04, 0.001, 0.05, 0.004, 0.02])
        expected = np.array([0.05, 0.005, 0.05, 0.010, 1.0 / 30.0])
        np.testing.assert_allclose(bh_qvalues(p), expected, rtol=1e-12)


class TestDifferentialAnalysis:
    def test_identical_samples_yield_zero_effect_and_p_one(self):
        rng = np.random.default_rng(0)
        x = rng.poisson(100, size=500).astype(float)
        counts = np.column_stack([x, x, x])
        res = differential_analysis(counts)
        np.testing.assert_allclose(res.effect_size, 0.0, atol=1e-12)
        np.testing.assert_allclose(res.p_value, 1.0, atol=1e-12)

    def test_diffresult_shapes(self):
        rng = np.random.default_rng(2)
        counts = rng.poisson(50, size=(100, 4)).astype(float)
        res = differential_analysis(counts)
        assert isinstance(res, DiffResult)
        assert res.size_factors.shape == (4,)
        assert res.per_locus_dispersion.shape == (100,)
        assert res.effect_size.shape == (100,)
        assert res.lrt_stat.shape == (100,)
        assert res.p_value.shape == (100,)
        assert res.q_value.shape == (100,)
        assert res.max_condition_idx.shape == (100,)
        assert res.max_condition_idx.dtype.kind == "i"
        assert res.dispersion_fit in {"parametric", "common", "common-fallback", "zero", "override"}
        assert np.all(res.per_locus_dispersion >= 0.0)

    def test_rejects_fewer_than_two_conditions(self):
        with pytest.raises(ValueError, match="n_conditions"):
            differential_analysis(np.arange(5).reshape(5, 1).astype(float))

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="n_conditions"):
            differential_analysis(np.array([1.0, 2.0, 3.0]))

    def test_rejects_negative_counts(self):
        with pytest.raises(ValueError, match="non-negative"):
            differential_analysis(np.array([[1.0, -1.0], [2.0, 3.0]]))

    def test_rejects_nonpositive_pseudocount(self):
        rng = np.random.default_rng(20)
        counts = rng.poisson(50, size=(50, 3)).astype(float)
        with pytest.raises(ValueError, match="pseudocount"):
            differential_analysis(counts, pseudocount=0.0)
        with pytest.raises(ValueError, match="pseudocount"):
            differential_analysis(counts, pseudocount=-0.1)

    def test_k_equals_two_warns_about_calibration(self):
        """K=2 is a degenerate regime for the NB-GLM LRT — per-locus variance
        has 1 df, χ²(1) asymptotics are poor, and the likelihood adds little
        over a raw ratio. The run should still succeed, but must warn."""
        rng = np.random.default_rng(21)
        counts = rng.poisson(80, size=(200, 2)).astype(float)
        with pytest.warns(FertilizerDiffWarning, match="K=2"):
            res = differential_analysis(counts)
        assert res.p_value.shape == (200,)
        assert res.effect_size.shape == (200,)

    def test_effect_size_is_max_abs_log2fc_vs_grand_mean(self):
        """Per spec, effect_size = max_j |log2((X_j/s_j) + pc) - log2(mean + pc)|.
        With flat anchor loci forcing size factors ~= 1 and a small pc,
        the variable locus below has per-condition normalized values
        (10, 20, 40, 80) and grand mean 37.5; the maximum absolute log2
        deviation is at condition 3 (80): log2(80.5/38) ~= 1.083."""
        counts = np.array([
            [10.0, 10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0, 20.0],
            [50.0, 50.0, 50.0, 50.0],
            [100.0, 100.0, 100.0, 100.0],
            [10.0, 20.0, 40.0, 80.0],
        ])
        res = differential_analysis(counts, pseudocount=0.5)
        np.testing.assert_allclose(res.effect_size[:4], 0.0, atol=1e-10)
        # Compute expected value by the same formula.
        norm = counts[4]  # size factors ~= 1 here
        pc = 0.5
        grand_mean = norm.mean()
        expected = np.max(np.abs(np.log2(norm + pc) - np.log2(grand_mean + pc)))
        assert res.effect_size[4] == pytest.approx(expected, rel=1e-6)

    def test_max_condition_idx_points_to_max_normalized_value(self):
        counts = np.array([
            [10.0, 10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0, 20.0],
            [50.0, 50.0, 50.0, 50.0],
            [100.0, 100.0, 100.0, 100.0],
            [10.0, 20.0, 40.0, 80.0],
            [80.0, 40.0, 20.0, 10.0],
        ])
        res = differential_analysis(counts)
        assert res.max_condition_idx[4] == 3
        assert res.max_condition_idx[5] == 0

    def test_detects_true_positives_with_controlled_fdr(self):
        """10% of loci up-regulated in one condition by 4x on a 4-condition
        Poisson background. We expect near-full power and ~5% empirical
        FDR at q <= 0.05."""
        rng = np.random.default_rng(3)
        n, K = 3000, 4
        mu = rng.uniform(20, 200, size=n)
        counts = np.column_stack([rng.poisson(mu) for _ in range(K)]).astype(float)
        diff = np.zeros(n, dtype=bool)
        diff[: n // 10] = True
        counts[diff, 2] = rng.poisson(mu[diff] * 4.0)

        res = differential_analysis(counts)
        assert (res.q_value[diff] < 0.05).mean() > 0.8
        discoveries = res.q_value < 0.05
        if discoveries.sum() > 0:
            fdr = (discoveries & ~diff).sum() / discoveries.sum()
            assert fdr < 0.1
        assert (res.max_condition_idx[diff] == 2).mean() > 0.9

    def test_size_factor_spread_warning(self):
        """Large library-size ratios should trigger a FertilizerDiffWarning."""
        rng = np.random.default_rng(4)
        n = 500
        base = rng.poisson(100, size=n).astype(float)
        counts = np.column_stack([base, base * 10.0, base * 0.1])
        with pytest.warns(FertilizerDiffWarning, match="size factors span"):
            differential_analysis(counts, size_factor_warn_ratio=5.0)

    def test_dispersion_override(self):
        rng = np.random.default_rng(5)
        counts = rng.poisson(100, size=(500, 4)).astype(float)
        res = differential_analysis(counts, dispersion_override=0.123)
        assert res.dispersion_fit == "override"
        np.testing.assert_allclose(res.per_locus_dispersion, 0.123)

    def test_dispersion_override_shifts_p_values(self):
        """Same data, different forced α: higher α broadens the NB null,
        so the LRT statistic for a clearly differential locus shrinks and
        its p-value moves toward 1. This locks in that the override actually
        enters the likelihood, not just the reported metadata."""
        rng = np.random.default_rng(14)
        counts = rng.poisson(100, size=(500, 3)).astype(float)
        counts[0] = [50.0, 100.0, 400.0]  # clearly differential
        res_poisson = differential_analysis(counts, dispersion_override=0.0)
        res_nb = differential_analysis(counts, dispersion_override=0.1)
        assert res_poisson.p_value[0] < res_nb.p_value[0]
        assert not np.allclose(res_poisson.p_value, res_nb.p_value)

    def test_fit_type_zero_forces_poisson(self):
        rng = np.random.default_rng(6)
        counts = rng.poisson(100, size=(500, 4)).astype(float)
        with pytest.warns(FertilizerDiffWarning, match="forces Poisson"):
            res = differential_analysis(counts, fit_type="zero")
        assert res.dispersion_fit == "zero"
        np.testing.assert_array_equal(res.per_locus_dispersion, 0.0)

    def test_fit_type_common_yields_scalar_alpha(self):
        rng = np.random.default_rng(7)
        counts = rng.poisson(100, size=(500, 4)).astype(float)
        res = differential_analysis(counts, fit_type="common")
        assert res.dispersion_fit in {"common", "common-fallback"}
        assert np.all(res.per_locus_dispersion == res.per_locus_dispersion[0])

    def test_poisson_fallback_emits_warning(self):
        """When too few loci pass --min-signal, dispersion silently dropping
        to 0 is strictly anti-conservative — must warn."""
        rng = np.random.default_rng(8)
        counts = rng.poisson(100, size=(500, 4)).astype(float)
        with pytest.warns(FertilizerDiffWarning, match="Poisson"):
            differential_analysis(
                counts, fit_type="common", dispersion_min_signal=1e6,
            )

    def test_parametric_common_fallback_emits_warning(self):
        """Parametric fit that degrades to common should say so."""
        rng = np.random.default_rng(9)
        counts = rng.poisson(100, size=(500, 4)).astype(float)
        with pytest.warns(FertilizerDiffWarning, match="Poisson"):
            differential_analysis(
                counts, fit_type="parametric", dispersion_min_signal=1e6,
            )


class TestCalibration:
    """Null-distribution calibration. These simulations assert that Type-I
    error at nominal alpha=0.05 does not exceed ~1.6x nominal across the
    parameter grid we actually care about. Bounds are loose because n_loci
    is only a few thousand and simulation noise is real."""

    @pytest.mark.parametrize("K,mu_val", [
        (3, 30),
        (3, 100),
        (3, 500),
        (5, 100),
        (8, 100),
    ])
    def test_type_one_under_poisson_null(self, K, mu_val):
        rng = np.random.default_rng(10 + K * 97 + mu_val)
        n = 4000
        counts = rng.poisson(mu_val, size=(n, K)).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FertilizerDiffWarning)
            res = differential_analysis(counts, fit_type="parametric")
        t1 = (res.p_value < 0.05).mean()
        assert t1 < 0.08, f"K={K}, mu={mu_val}: T1@0.05 = {t1:.4f}"

    @pytest.mark.parametrize("K,mu_val,alpha_true", [
        (3, 100, 0.05),
        (3, 100, 0.10),
        (5, 100, 0.05),
        (5, 100, 0.10),
        (8, 100, 0.05),
        (8, 100, 0.10),
    ])
    def test_type_one_under_nb_null_with_trend(self, K, mu_val, alpha_true):
        """Under NB with moderate overdispersion, the parametric trend
        should recover alpha well enough to keep T1 near nominal."""
        rng = np.random.default_rng(100 + K * 53 + int(alpha_true * 1000))
        n = 5000
        r = 1.0 / alpha_true
        p = r / (r + mu_val)
        counts = rng.negative_binomial(r, p, size=(n, K)).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FertilizerDiffWarning)
            res = differential_analysis(counts, fit_type="parametric")
        t1 = (res.p_value < 0.05).mean()
        assert t1 < 0.09, (
            f"K={K}, mu={mu_val}, alpha={alpha_true}: T1@0.05 = {t1:.4f}"
        )

    def test_dispersion_recovery_from_nb_data(self):
        """With known NB dispersion, the common-fit estimator should
        recover it within a small factor on data where most loci are null."""
        rng = np.random.default_rng(202)
        n, K = 4000, 5
        mu_val = 200
        alpha_true = 0.1
        r = 1.0 / alpha_true
        p = r / (r + mu_val)
        counts = rng.negative_binomial(r, p, size=(n, K)).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FertilizerDiffWarning)
            res = differential_analysis(counts, fit_type="common")
        alpha_hat = res.per_locus_dispersion[0]
        # Allow 2x slack; MoM + median is known to be moderately biased
        # downward at low K, but should not be off by more than this.
        assert alpha_true / 2.0 < alpha_hat < alpha_true * 2.0, (
            f"alpha_true={alpha_true}, alpha_hat={alpha_hat:.4f}"
        )


class TestDiffCLI:
    def _write_input(self, path, **cols):
        pd.DataFrame(cols).to_csv(path, sep="\t", index=False)

    def _diff_argv(self, inp, conditions, out, extra=()):
        return [
            "diff",
            "-i", str(inp),
            "-c", *conditions,
            "-o", str(out),
            *extra,
        ]

    def test_end_to_end_three_conditions(self, tmp_path):
        rng = np.random.default_rng(10)
        n = 800
        a = rng.poisson(60, size=n).astype(float)
        b = rng.poisson(60, size=n).astype(float)
        c = rng.poisson(60, size=n).astype(float)
        c[-80:] = rng.poisson(300, size=80)

        inp = tmp_path / "in.tsv"
        out = tmp_path / "out.tsv"
        self._write_input(
            inp,
            chrom=[f"chr{i%3 + 1}" for i in range(n)],
            start=np.arange(n) * 100,
            end=np.arange(n) * 100 + 50,
            A=a, B=b, C=c,
        )
        assert main(self._diff_argv(inp, ["A", "B", "C"], out)) == 0

        df = pd.read_csv(out, sep="\t")
        assert list(df.columns) == [
            "chrom", "start", "end", "A", "B", "C",
            "effect_size", "p_value", "q_value", "max_condition",
        ]
        assert (df["max_condition"] == "C").mean() > 0.5
        assert df["q_value"].le(0.05).all()

    def test_filter_by_p_threshold(self, tmp_path):
        """--p-threshold AND-combines with --q-threshold, tightening output."""
        rng = np.random.default_rng(13)
        n = 500
        counts = {col: rng.poisson(100, size=n).astype(float) for col in "ABCD"}
        inp = tmp_path / "in.tsv"
        self._write_input(
            inp,
            chrom=["chr1"] * n,
            start=np.arange(n),
            end=np.arange(n) + 1,
            **counts,
        )
        out_q_only = tmp_path / "q.tsv"
        main(self._diff_argv(inp, list("ABCD"), out_q_only, ["--q-threshold", "1.0"]))
        df_q = pd.read_csv(out_q_only, sep="\t")
        assert len(df_q) == n

        out_pq = tmp_path / "pq.tsv"
        main(self._diff_argv(
            inp, list("ABCD"), out_pq,
            ["--q-threshold", "1.0", "--p-threshold", "0.1"],
        ))
        df_pq = pd.read_csv(out_pq, sep="\t")
        assert len(df_pq) < len(df_q)
        assert (df_pq["p_value"] <= 0.1).all()

    def test_filter_by_q_threshold(self, tmp_path):
        rng = np.random.default_rng(11)
        n = 500
        counts = {col: rng.poisson(100, size=n).astype(float) for col in "ABCD"}
        inp = tmp_path / "in.tsv"
        self._write_input(
            inp,
            chrom=["chr1"] * n,
            start=np.arange(n),
            end=np.arange(n) + 1,
            **counts,
        )
        out_strict = tmp_path / "strict.tsv"
        main(self._diff_argv(inp, list("ABCD"), out_strict, ["--q-threshold", "0.05"]))
        df_strict = pd.read_csv(out_strict, sep="\t")

        out_loose = tmp_path / "loose.tsv"
        main(self._diff_argv(inp, list("ABCD"), out_loose, ["--q-threshold", "1.0"]))
        df_loose = pd.read_csv(out_loose, sep="\t")

        assert len(df_loose) == n
        assert len(df_strict) <= len(df_loose)

    def test_preserves_input_order_among_kept_rows(self, tmp_path):
        inp = tmp_path / "in.tsv"
        out = tmp_path / "out.tsv"
        self._write_input(
            inp,
            chrom=["chr3", "chr1", "chr2", "chr1", "chr3", "chr2"],
            start=[500, 0, 100, 200, 300, 400],
            end=[600, 100, 200, 300, 400, 500],
            A=[10.0, 20.0, 15.0, 5.0, 8.0, 12.0],
            B=[12.0, 22.0, 14.0, 6.0, 9.0, 11.0],
            C=[11.0, 21.0, 13.0, 7.0, 10.0, 13.0],
        )
        main(self._diff_argv(inp, ["A", "B", "C"], out, ["--q-threshold", "1.0"]))
        df = pd.read_csv(out, sep="\t")
        assert df["chrom"].tolist() == ["chr3", "chr1", "chr2", "chr1", "chr3", "chr2"]
        assert df["start"].tolist() == [500, 0, 100, 200, 300, 400]

    def test_missing_column_raises(self, tmp_path):
        inp = tmp_path / "in.tsv"
        out = tmp_path / "out.tsv"
        self._write_input(
            inp,
            chrom=["chr1", "chr1"], start=[0, 100], end=[50, 150],
            A=[10.0, 20.0], B=[12.0, 18.0],
        )
        with pytest.raises(ValueError, match="columns not found"):
            main(self._diff_argv(inp, ["A", "nope"], out))

    def test_single_condition_rejected(self, tmp_path):
        inp = tmp_path / "in.tsv"
        out = tmp_path / "out.tsv"
        self._write_input(
            inp,
            chrom=["chr1"], start=[0], end=[100],
            A=[10.0],
        )
        with pytest.raises(ValueError, match="at least 2 conditions"):
            main(self._diff_argv(inp, ["A"], out))

    def test_dispersion_override_flag(self, tmp_path):
        rng = np.random.default_rng(12)
        n = 300
        inp = tmp_path / "in.tsv"
        out = tmp_path / "out.tsv"
        cols = {c: rng.poisson(60, size=n).astype(float) for c in "ABC"}
        self._write_input(
            inp,
            chrom=["chr1"] * n,
            start=np.arange(n),
            end=np.arange(n) + 1,
            **cols,
        )
        assert main(self._diff_argv(
            inp, list("ABC"), out,
            ["--q-threshold", "1.0", "--dispersion", "0.05"],
        )) == 0
        # All rows present (loose threshold)
        df = pd.read_csv(out, sep="\t")
        assert len(df) == n


def test_parser_has_both_subcommands():
    parser = _build_parser()
    for sub in ("extract", "diff"):
        with pytest.raises(SystemExit):
            parser.parse_args([sub, "--help"])
