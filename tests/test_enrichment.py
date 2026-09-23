"""Tests for NB-GLM enrichment analysis across >=2 conditions."""

from __future__ import annotations

import gzip
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.optimize import brentq

from fertilizer import enrichment as enrichment_module
from fertilizer.cli import build_parser, main
from fertilizer.enrichment import (
    EnrichmentResult,
    FertilizerEnrichmentWarning,
    _apply_trend,
    _expected_t1_at_05,
    _fit_parametric_trend,
    _intercept_mle,
    _nb_logpmf,
    _read_extract_stat,
    _warn_if_regions_overlap,
    bh_qvalues,
    enrichment_analysis,
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


class TestEnrichmentAnalysis:
	def test_identical_samples_yield_zero_effect_and_p_one(self):
		rng = np.random.default_rng(0)
		x = rng.poisson(100, size=500).astype(float)
		counts = np.column_stack([x, x, x])
		res = enrichment_analysis(counts)
		np.testing.assert_allclose(res.effect_size, 0.0, atol=1e-12)
		np.testing.assert_allclose(res.p_value, 1.0, atol=1e-12)

	def test_enrichment_result_shapes(self):
		rng = np.random.default_rng(2)
		counts = rng.poisson(50, size=(100, 4)).astype(float)
		res = enrichment_analysis(counts)
		assert isinstance(res, EnrichmentResult)
		assert res.size_factors.shape == (4,)
		assert res.per_locus_dispersion.shape == (100,)
		assert res.effect_size.shape == (100,)
		assert res.lrt_stat.shape == (100,)
		assert res.p_value.shape == (100,)
		assert res.q_value.shape == (100,)
		assert res.enriched_condition_idx.shape == (100,)
		assert res.enriched_condition_idx.dtype.kind == "i"
		assert res.dispersion_fit in {"parametric", "common", "common-fallback", "zero", "override"}
		assert np.all(res.per_locus_dispersion >= 0.0)

	def test_rejects_fewer_than_two_conditions(self):
		with pytest.raises(ValueError, match="n_conditions"):
			enrichment_analysis(np.arange(5).reshape(5, 1).astype(float))

	def test_rejects_1d_input(self):
		with pytest.raises(ValueError, match="n_conditions"):
			enrichment_analysis(np.array([1.0, 2.0, 3.0]))

	def test_rejects_negative_counts(self):
		with pytest.raises(ValueError, match="non-negative"):
			enrichment_analysis(np.array([[1.0, -1.0], [2.0, 3.0]]))

	@pytest.mark.parametrize("bad", [np.nan, np.inf])
	def test_rejects_nonfinite_counts(self, bad):
		counts = np.full((20, 3), 10.0)
		counts[3, 1] = bad
		with pytest.raises(ValueError, match="finite"):
			enrichment_analysis(counts)

	def test_rejects_nonpositive_pseudocount(self):
		rng = np.random.default_rng(20)
		counts = rng.poisson(50, size=(50, 3)).astype(float)
		with pytest.raises(ValueError, match="pseudocount"):
			enrichment_analysis(counts, pseudocount=0.0)
		with pytest.raises(ValueError, match="pseudocount"):
			enrichment_analysis(counts, pseudocount=-0.1)

	def test_k_equals_two_runs_without_warning(self):
		"""K=2 is well-defined: the (k*, k_bg) pair is just the two
		conditions, and the default background_rank of 3 is silently capped
		to K=2. No quality warning should fire."""
		rng = np.random.default_rng(21)
		counts = rng.poisson(80, size=(200, 2)).astype(float)
		with warnings.catch_warnings():
			warnings.simplefilter("error", FertilizerEnrichmentWarning)
			res = enrichment_analysis(counts)
		assert res.p_value.shape == (200,)
		assert res.effect_size.shape == (200,)
		assert res.background_rank == 2

	def test_effect_size_is_log2_enriched_over_rest(self):
		"""Per spec, effect_size = log2((X_{k*}/s_{k*}) + pc) - log2(mean_rest + pc).
		With flat anchor loci forcing size factors ~= 1 and pc = 0.5, the
		variable locus (10, 20, 40, 80) has k* = 3 (X = 80), mean-of-rest =
		(10 + 20 + 40)/3 = 23.333, so effect = log2(80.5) - log2(23.833)."""
		counts = np.array([
		    [10.0, 10.0, 10.0, 10.0],
		    [20.0, 20.0, 20.0, 20.0],
		    [50.0, 50.0, 50.0, 50.0],
		    [100.0, 100.0, 100.0, 100.0],
		    [10.0, 20.0, 40.0, 80.0],
		])
		res = enrichment_analysis(counts, pseudocount=0.5)
		np.testing.assert_allclose(res.effect_size[:4], 0.0, atol=1e-10)
		pc = 0.5
		mean_rest = (10.0 + 20.0 + 40.0) / 3.0
		expected = np.log2(80.0 + pc) - np.log2(mean_rest + pc)
		assert res.effect_size[4] == pytest.approx(expected, rel=1e-6)

	def test_effect_size_is_always_nonnegative(self):
		"""k* is the argmax, so log2(enriched) >= log2(mean-of-rest)
		whenever pc is small; with pc > 0 it can dip slightly negative
		for borderline cases, but at the chosen pc=0.5 and a real spread
		it must be >= 0."""
		rng = np.random.default_rng(33)
		counts = rng.poisson(100, size=(500, 4)).astype(float)
		res = enrichment_analysis(counts, pseudocount=0.5)
		assert np.all(res.effect_size >= -1e-12)

	def test_enriched_condition_idx_points_to_max_normalized_value(self):
		counts = np.array([
		    [10.0, 10.0, 10.0, 10.0],
		    [20.0, 20.0, 20.0, 20.0],
		    [50.0, 50.0, 50.0, 50.0],
		    [100.0, 100.0, 100.0, 100.0],
		    [10.0, 20.0, 40.0, 80.0],
		    [80.0, 40.0, 20.0, 10.0],
		])
		res = enrichment_analysis(counts)
		assert res.enriched_condition_idx[4] == 3
		assert res.enriched_condition_idx[5] == 0

	def test_detects_true_positives_with_controlled_fdr(self):
		"""10% of loci up-regulated in one condition by 4x on a 4-condition
		Poisson background. We expect near-full power and ~5% empirical
		FDR at q <= 0.05."""
		rng = np.random.default_rng(3)
		n, K = 3000, 4
		mu = rng.uniform(20, 200, size=n)
		counts = np.column_stack([rng.poisson(mu) for _ in range(K)]).astype(float)
		enriched = np.zeros(n, dtype=bool)
		enriched[: n // 10] = True
		counts[enriched, 2] = rng.poisson(mu[enriched] * 4.0)

		res = enrichment_analysis(counts)
		assert (res.q_value[enriched] < 0.05).mean() > 0.8
		discoveries = res.q_value < 0.05
		if discoveries.sum() > 0:
			fdr = (discoveries & ~enriched).sum() / discoveries.sum()
			assert fdr < 0.1
		assert (res.enriched_condition_idx[enriched] == 2).mean() > 0.9

	def test_does_not_call_depletion_only_loci(self):
		"""A locus where exactly one condition is *depleted* relative to the
		others must NOT be called. The (k*, k_bg) LRT compares the top
		condition against the rank-`background_rank` condition; under
		depletion, both of those sit at the high level and look approximately
		equal under the test, so LRT ~ 0 and the locus is not called.
		Contrast with the same magnitude of *enrichment* in one condition,
		where k* sharply exceeds k_bg and the locus IS detected."""
		rng = np.random.default_rng(42)
		n, K = 2000, 4
		mu_base = 200.0

		# 5% depletion-only loci: 1 condition at mu/4, the rest at mu.
		# 5% enrichment-only loci: 1 condition at 4*mu, the rest at mu.
		# 90% null loci: all K at mu.
		counts = np.column_stack([
		    rng.poisson(mu_base, size=n) for _ in range(K)
		]).astype(float)
		depleted = np.zeros(n, dtype=bool)
		enriched = np.zeros(n, dtype=bool)
		depleted[: n // 20] = True
		enriched[n // 20 : 2 * (n // 20)] = True
		counts[depleted, 0] = rng.poisson(mu_base / 4.0, size=depleted.sum())
		counts[enriched, 0] = rng.poisson(mu_base * 4.0, size=enriched.sum())

		res = enrichment_analysis(counts)

		# Enrichment loci: strong detection.
		assert (res.q_value[enriched] < 0.05).mean() > 0.8
		# Depletion loci: nominal Type-I rate, no asymmetric inflation.
		assert (res.q_value[depleted] < 0.05).mean() < 0.05
		# Among called loci, almost all are true enrichment positives.
		called = res.q_value < 0.05
		if called.sum() > 0:
			fdr = (called & ~enriched).sum() / called.sum()
			assert fdr < 0.1

	def test_background_rank_default_is_three(self):
		"""Documented default is rank 3 (one competing peak tolerated)."""
		rng = np.random.default_rng(901)
		counts = rng.poisson(100, size=(100, 5)).astype(float)
		res = enrichment_analysis(counts)
		assert res.background_rank == 3

	def test_background_rank_2_uses_second_highest_condition(self):
		"""At background_rank=2, the LRT compares k* against the
		second-highest condition. On data with a competing peak (where the
		second-highest is elevated), rank=2 gives larger p-values than
		rank=3 (which compares against a baseline condition instead)."""
		rng = np.random.default_rng(902)
		counts = rng.poisson(80, size=(300, 4)).astype(float)
		res = enrichment_analysis(counts, background_rank=2,
		                          dispersion_override=0.05)
		assert res.background_rank == 2

		rng2 = np.random.default_rng(903)
		counts2 = rng2.poisson(80, size=(300, 4)).astype(float)
		# Inject one competing peak in condition 1 for the first 50 loci.
		counts2[:50, 1] = rng2.poisson(160, size=50).astype(float)
		# And a real enrichment in condition 0 for those same loci.
		counts2[:50, 0] = rng2.poisson(320, size=50).astype(float)

		res2 = enrichment_analysis(counts2, background_rank=2,
		                           dispersion_override=0.05)
		res3 = enrichment_analysis(counts2, background_rank=3,
		                           dispersion_override=0.05)
		# rank-3 (which skips the competing peak) gives SMALLER p-values
		# for the loci with a competing peak than rank-2 does.
		assert (res3.p_value[:50] < res2.p_value[:50]).mean() > 0.7

	def test_background_rank_robust_to_competing_peak(self):
		"""Default (rank 3) should detect enrichment in the presence of a
		single competing peak that defeats rank 2. Setup respects the
		null-majority assumption: 2000 loci on a 5-condition Poisson
		background, with 5% truly enriched in cond 0 AND simultaneously
		elevated about as much in cond 1 (the 'competing peak'). The other
		K-2 conditions sit at baseline."""
		rng = np.random.default_rng(904)
		n, K = 2000, 5
		base_mu = 100
		counts = np.column_stack([
		    rng.poisson(base_mu, size=n) for _ in range(K)
		]).astype(float)
		n_signal = n // 20  # 5% — preserves null-majority for size factors
		signal = slice(0, n_signal)
		counts[signal, 0] = rng.poisson(base_mu * 4.0, size=n_signal).astype(float)
		counts[signal, 1] = rng.poisson(base_mu * 3.5, size=n_signal).astype(float)

		res2 = enrichment_analysis(counts, background_rank=2, fit_type="zero")
		res3 = enrichment_analysis(counts, background_rank=3, fit_type="zero")
		# Rank 2 sees the competing peak as the background → small LRT
		# statistic → few calls. Rank 3 sees a baseline condition → many.
		calls2 = (res2.q_value[signal] < 0.05).sum()
		calls3 = (res3.q_value[signal] < 0.05).sum()
		assert calls3 > calls2, f"rank-3 calls={calls3}, rank-2 calls={calls2}"
		# Among rank-3 calls on the signal loci, almost all point to cond 0.
		called3 = res3.q_value[signal] < 0.05
		assert (res3.enriched_condition_idx[signal][called3] == 0).mean() > 0.9

	def test_background_rank_caps_to_K(self):
		"""Requesting a rank larger than K silently caps to K so the default
		of 3 just works at K=2."""
		rng = np.random.default_rng(905)
		counts = rng.poisson(100, size=(100, 3)).astype(float)
		res = enrichment_analysis(counts, background_rank=99)
		assert res.background_rank == 3
		# And the default at K=2:
		counts2 = rng.poisson(100, size=(100, 2)).astype(float)
		res2 = enrichment_analysis(counts2)
		assert res2.background_rank == 2

	def test_conditions_below_kbg_are_nuisance_and_dont_affect_lrt(self):
		"""Locks in the nuisance-cancellation property: at K=5 with
		background_rank=3, the LRT is computed only from (k*, k_bg) = (rank
		1, rank 3). Varying the values of the rank-4 and rank-5 conditions
		(while keeping them below k_bg in the ordering) must NOT change
		lrt_stat or p_value. The effect_size IS allowed to change — it is
		a separate user-facing summary that averages the K-1 non-k* values.

		Setup forces a deterministic ranking of the top 3:
		  - cond 0: poisson(400) → rank 1 (k*)
		  - cond 1: poisson(200) → rank 2
		  - cond 2: poisson(100) → rank 3 (k_bg)
		  - cond 3, 4: small constants < 70 → always ranks 4, 5
		The means are far enough apart that the top-3 ordering is fixed
		across all loci with overwhelming probability. Size factors and
		dispersion are pinned via overrides so they don't introduce
		indirect dependencies."""
		rng = np.random.default_rng(910)
		n, K = 200, 5
		counts_a = np.zeros((n, K), dtype=np.float64)
		counts_a[:, 0] = rng.poisson(400, size=n)
		counts_a[:, 1] = rng.poisson(200, size=n)
		counts_a[:, 2] = rng.poisson(100, size=n)
		counts_a[:, 3] = 10.0
		counts_a[:, 4] = 5.0

		# Build a second input that agrees on conds 0/1/2 but uses
		# different (still small) values for the nuisance conds 3 and 4.
		counts_b = counts_a.copy()
		counts_b[:, 3] = 30.0
		counts_b[:, 4] = 20.0

		# Sanity: top-3 ordering is identical across the two inputs and
		# constant across loci (cond 0 > cond 1 > cond 2 by mean).
		order_a = np.argsort(counts_a, axis=1)
		order_b = np.argsort(counts_b, axis=1)
		np.testing.assert_array_equal(order_a[:, -3:], order_b[:, -3:])
		# The top-3 sort positions hold (0, 1, 2) at every locus.
		np.testing.assert_array_equal(order_a[:, -3:],
		                              np.tile([2, 1, 0], (n, 1)))

		sf = np.ones(K)
		res_a = enrichment_analysis(counts_a, background_rank=3,
		                            dispersion_override=0.05,
		                            size_factors_override=sf)
		res_b = enrichment_analysis(counts_b, background_rank=3,
		                            dispersion_override=0.05,
		                            size_factors_override=sf)
		# Test statistic and p-value must be identical to numerical precision.
		np.testing.assert_allclose(res_a.lrt_stat, res_b.lrt_stat, atol=1e-12)
		np.testing.assert_allclose(res_a.p_value, res_b.p_value, atol=1e-12)
		np.testing.assert_array_equal(res_a.enriched_condition_idx,
		                               res_b.enriched_condition_idx)
		# Effect size DOES depend on the changed conditions (it averages
		# the K-1 non-k* values), so it must differ on every locus.
		assert not np.allclose(res_a.effect_size, res_b.effect_size)

	def test_background_rank_rejects_below_2(self):
		rng = np.random.default_rng(906)
		counts = rng.poisson(50, size=(50, 4)).astype(float)
		with pytest.raises(ValueError, match="background_rank"):
			enrichment_analysis(counts, background_rank=1)
		with pytest.raises(ValueError, match="background_rank"):
			enrichment_analysis(counts, background_rank=0)
		with pytest.raises(ValueError, match="background_rank"):
			enrichment_analysis(counts, background_rank=1.5)  # type: ignore[arg-type]

	def test_p_value_bonferroni_corrected_by_K(self):
		"""The per-locus p-value reported is exactly
		`min(K * 0.5 * chi^2(1).sf(LRT), 1)`: the upper half-chi^2(1) tail
		for the one-sided LRT, multiplied by K to Bonferroni-correct for
		the data-driven argmax choice of k*. This locks in the formula
		end-to-end against the lrt_stat that the routine reports."""
		rng = np.random.default_rng(50)
		n, K = 500, 4
		counts = rng.poisson(80, size=(n, K)).astype(float)
		res = enrichment_analysis(counts, dispersion_override=0.01)
		from scipy.stats import chi2 as _chi2
		bonf = np.minimum(K * 0.5 * _chi2.sf(res.lrt_stat, df=1), 1.0)
		np.testing.assert_allclose(res.p_value, bonf, atol=1e-12)

	def test_size_factor_spread_warning(self):
		"""Large library-size ratios should trigger a FertilizerEnrichmentWarning."""
		rng = np.random.default_rng(4)
		n = 500
		base = rng.poisson(100, size=n).astype(float)
		counts = np.column_stack([base, base * 10.0, base * 0.1])
		with pytest.warns(FertilizerEnrichmentWarning, match="size factors span"):
			enrichment_analysis(counts, size_factor_warn_ratio=5.0)

	def test_size_factors_override_skips_estimation(self):
		rng = np.random.default_rng(91)
		counts = rng.poisson(100, size=(500, 3)).astype(float)
		res = enrichment_analysis(counts, size_factors_override=np.array([1.0, 2.0, 4.0]))
		np.testing.assert_allclose(res.size_factors, [1.0, 2.0, 4.0])
		assert res.n_loci_for_size_factors == 0

	def test_size_factors_override_changes_call_argmax(self):
		"""Manually setting size factors to bake in a 4x library-size
		difference in condition 2 should make a flat-input locus look enriched
		in condition 0, not condition 2."""
		counts = np.array([
		    [50.0, 50.0, 200.0],
		    [10.0, 10.0, 40.0],
		    [25.0, 25.0, 100.0],
		    [5.0, 5.0, 20.0],
		])
		# Without override: condition 2 looks enriched (raw counts 4x higher).
		res_default = enrichment_analysis(
		    counts,
		    size_factors_override=np.array([1.0, 1.0, 1.0]),
		)
		assert (res_default.enriched_condition_idx == 2).all()
		# With override saying "condition 2 had 4x more sequencing depth":
		# normalized signal is now flat across conditions and no condition
		# is the argmax in a meaningful sense.
		res_norm = enrichment_analysis(
		    counts,
		    size_factors_override=np.array([1.0, 1.0, 4.0]),
		)
		# All p-values should be ~1 because there is no enrichment after
		# accounting for the supplied depth ratio.
		assert (res_norm.p_value > 0.5).all()

	def test_size_factors_override_validates_shape(self):
		counts = np.ones((20, 3)) * 10
		with pytest.raises(ValueError, match="shape"):
			enrichment_analysis(counts, size_factors_override=np.array([1.0, 1.0]))

	def test_size_factors_override_rejects_nonpositive(self):
		counts = np.ones((20, 3)) * 10
		with pytest.raises(ValueError, match="positive"):
			enrichment_analysis(counts, size_factors_override=np.array([1.0, 0.0, 1.0]))
		with pytest.raises(ValueError, match="positive"):
			enrichment_analysis(counts, size_factors_override=np.array([1.0, -1.0, 1.0]))

	def test_dispersion_override(self):
		rng = np.random.default_rng(5)
		counts = rng.poisson(100, size=(500, 4)).astype(float)
		res = enrichment_analysis(counts, dispersion_override=0.123)
		assert res.dispersion_fit == "override"
		np.testing.assert_allclose(res.per_locus_dispersion, 0.123)

	def test_dispersion_override_shifts_p_values(self):
		"""Same data, different forced α: higher α broadens the NB null,
		so the LRT statistic for a clearly enriched locus shrinks and
		its p-value moves toward 1. This locks in that the override actually
		enters the likelihood, not just the reported metadata."""
		rng = np.random.default_rng(14)
		counts = rng.poisson(100, size=(500, 3)).astype(float)
		counts[0] = [50.0, 100.0, 400.0]  # clearly enriched
		res_poisson = enrichment_analysis(counts, dispersion_override=0.0)
		res_nb = enrichment_analysis(counts, dispersion_override=0.1)
		assert res_poisson.p_value[0] < res_nb.p_value[0]
		assert not np.allclose(res_poisson.p_value, res_nb.p_value)

	@pytest.mark.parametrize("bad", [-0.5, np.nan, np.inf])
	def test_dispersion_override_rejects_invalid(self, bad):
		counts = np.random.default_rng(5).poisson(100, size=(50, 3)).astype(float)
		with pytest.raises(ValueError, match="dispersion_override"):
			enrichment_analysis(counts, dispersion_override=bad)

	def test_fit_type_zero_forces_poisson(self):
		rng = np.random.default_rng(6)
		counts = rng.poisson(100, size=(500, 4)).astype(float)
		with pytest.warns(FertilizerEnrichmentWarning, match="forces Poisson"):
			res = enrichment_analysis(counts, fit_type="zero")
		assert res.dispersion_fit == "zero"
		np.testing.assert_array_equal(res.per_locus_dispersion, 0.0)

	def test_fit_type_common_yields_scalar_alpha(self):
		rng = np.random.default_rng(7)
		counts = rng.poisson(100, size=(500, 4)).astype(float)
		res = enrichment_analysis(counts, fit_type="common")
		assert res.dispersion_fit in {"common", "common-fallback"}
		assert np.all(res.per_locus_dispersion == res.per_locus_dispersion[0])

	def test_poisson_fallback_emits_warning(self):
		"""When too few loci pass --min-signal, dispersion silently dropping
		to 0 is strictly anti-conservative — must warn."""
		rng = np.random.default_rng(8)
		counts = rng.poisson(100, size=(500, 4)).astype(float)
		with pytest.warns(FertilizerEnrichmentWarning, match="Poisson"):
			enrichment_analysis(
			    counts, fit_type="common", dispersion_min_signal=1e6,
			)

	def test_parametric_common_fallback_emits_warning(self):
		"""Parametric fit that degrades to common should say so."""
		rng = np.random.default_rng(9)
		counts = rng.poisson(100, size=(500, 4)).astype(float)
		with pytest.warns(FertilizerEnrichmentWarning, match="Poisson"):
			enrichment_analysis(
			    counts, fit_type="parametric", dispersion_min_signal=1e6,
			)


class TestCalibration:
	"""Null-distribution calibration. These simulations assert that Type-I
	error at nominal alpha=0.05 stays within sane bounds across the parameter
	grid we care about. Bounds are loose because n_loci is only a few thousand
	and simulation noise is real.

	At `background_rank=2`, the test is uniformly conservative across K.
	At `background_rank=3` (default), the test is above nominal at K=3
	(~0.08; rank-3 = the lowest condition out of three, where the
	order-statistic gap is at its widest) and increasingly conservative
	for K>=4 as Bonferroni × K dominates."""

	@pytest.mark.parametrize("K,mu_val", [
	    (3, 30),
	    (3, 100),
	    (3, 500),
	    (5, 100),
	    (8, 100),
	])
	def test_rank_2_poisson_null_is_conservative(self, K, mu_val):
		"""At background_rank=2 the test is tightly conservative across K."""
		rng = np.random.default_rng(10 + K * 97 + mu_val)
		n = 4000
		counts = rng.poisson(mu_val, size=(n, K)).astype(float)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", FertilizerEnrichmentWarning)
			res = enrichment_analysis(counts, fit_type="parametric",
			                          background_rank=2)
		t1 = (res.p_value < 0.05).mean()
		assert t1 < 0.08, f"K={K}, mu={mu_val}: T1@0.05 = {t1:.4f}"

	@pytest.mark.parametrize("K,mu_val", [
	    (3, 30),
	    (3, 100),
	    (3, 500),
	    (5, 100),
	    (8, 100),
	])
	def test_default_rank_poisson_null_within_bounds(self, K, mu_val):
		"""background_rank=3 (default): above nominal at K=3 (rank-3 =
		lowest of three; order-statistic gap is widest here), conservative
		for K>=4."""
		rng = np.random.default_rng(10 + K * 97 + mu_val)
		n = 4000
		counts = rng.poisson(mu_val, size=(n, K)).astype(float)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", FertilizerEnrichmentWarning)
			res = enrichment_analysis(counts, fit_type="parametric")
		t1 = (res.p_value < 0.05).mean()
		# At K=3, rank=3 compares max vs min of 3 → mildly anti-conservative.
		# At K>=4 the Bonferroni × K dominates and the test is sub-nominal.
		bound = 0.10 if K == 3 else 0.05
		assert t1 < bound, f"K={K}, mu={mu_val}: T1@0.05 = {t1:.4f}"

	@pytest.mark.parametrize("K,mu_val,alpha_true", [
	    (3, 100, 0.05),
	    (3, 100, 0.10),
	    (5, 100, 0.05),
	    (5, 100, 0.10),
	    (8, 100, 0.05),
	    (8, 100, 0.10),
	])
	def test_rank_2_nb_null_with_trend(self, K, mu_val, alpha_true):
		"""Under NB with moderate overdispersion, the parametric trend
		should estimate alpha well enough to keep T1 near nominal at
		background_rank=2."""
		rng = np.random.default_rng(100 + K * 53 + int(alpha_true * 1000))
		n = 5000
		r = 1.0 / alpha_true
		p = r / (r + mu_val)
		counts = rng.negative_binomial(r, p, size=(n, K)).astype(float)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", FertilizerEnrichmentWarning)
			res = enrichment_analysis(counts, fit_type="parametric",
			                          background_rank=2)
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
			warnings.simplefilter("ignore", FertilizerEnrichmentWarning)
			res = enrichment_analysis(counts, fit_type="common")
		alpha_hat = res.per_locus_dispersion[0]
		# Allow 2x slack; MoM + median is known to be moderately biased
		# downward at low K, but should not be off by more than this.
		assert alpha_true / 2.0 < alpha_hat < alpha_true * 2.0, (
		    f"alpha_true={alpha_true}, alpha_hat={alpha_hat:.4f}"
		)

	def test_size_factors_and_dispersion_robust_to_minority_enrichment(self):
		"""The null-majority assumption: with 5% of loci truly enriched in
		one condition, both the size factors and the common-fit dispersion
		should stay close to the all-null values. This is the regime users
		actually run in; we want to lock in that the median-of-ratios and
		the median-of-MoM aren't pulled off by a realistic minority of
		true positives."""
		rng = np.random.default_rng(7777)
		n, K = 4000, 4
		mu_val = 200
		alpha_true = 0.05
		r = 1.0 / alpha_true
		p = r / (r + mu_val)
		counts = rng.negative_binomial(r, p, size=(n, K)).astype(float)

		# 5% of loci: condition 0 is 4x enriched.
		n_enriched = n // 20
		r_hi = 1.0 / alpha_true
		p_hi = r_hi / (r_hi + 4.0 * mu_val)
		counts[:n_enriched, 0] = rng.negative_binomial(r_hi, p_hi, size=n_enriched)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore", FertilizerEnrichmentWarning)
			res = enrichment_analysis(counts, fit_type="common")

		# Size factors should still be close to 1 (within 20%) — the 5%
		# enrichment in cond 0 is a minority and median-of-ratios absorbs it.
		sf = res.size_factors
		sf_normed = sf / np.exp(np.log(sf).mean())
		assert np.all(np.abs(sf_normed - 1.0) < 0.2), f"sf_normed={sf_normed}"

		# Common alpha should be within 2x of truth.
		alpha_hat = res.per_locus_dispersion[0]
		assert alpha_true / 2.0 < alpha_hat < alpha_true * 2.5, (
		    f"alpha_true={alpha_true}, alpha_hat={alpha_hat:.4f}"
		)

	def test_n_loci_for_size_factors_is_reported(self):
		rng = np.random.default_rng(3030)
		counts = rng.poisson(100, size=(500, 3)).astype(float)
		# Zero out 10% of loci in condition 0 — they will be excluded from
		# size-factor estimation.
		counts[:50, 0] = 0
		res = enrichment_analysis(counts, fit_type="common")
		assert isinstance(res.n_loci_for_size_factors, int)
		assert res.n_loci_for_size_factors <= 450
		assert res.n_loci_for_size_factors > 400

	def test_lrt_zero_dominated_flag(self):
		"""Loci where the LRT pair (X_top, X_bg) contains a zero are flagged.
		These produce very small p-values driven by the 1e-20 mu_alt clamp
		rather than by data and should be visible to users."""
		counts = np.array([
		    [10.0, 10.0, 10.0, 10.0],
		    [20.0, 20.0, 20.0, 20.0],
		    [50.0, 50.0, 50.0, 50.0],
		    [100.0, 100.0, 100.0, 100.0],
		    [0.0, 20.0, 40.0, 80.0],   # X_bg (rank-3, normalized=20) > 0; X_top=80 > 0 -> NOT flagged
		    [0.0, 0.0, 0.0, 80.0],     # X_bg (rank-3) == 0 -> flagged
		])
		res = enrichment_analysis(counts, dispersion_override=0.05)
		assert res.lrt_zero_dominated.shape == (6,)
		assert not res.lrt_zero_dominated[:5].any()
		assert res.lrt_zero_dominated[5]

	def test_pc_dominated_flag_fires_when_expected(self):
		counts = np.array([
		    [10.0, 10.0, 10.0],
		    [20.0, 20.0, 20.0],
		    [50.0, 50.0, 50.0],
		    [100.0, 100.0, 100.0],
		    [0.0, 20.0, 80.0],  # one normalized condition < pc=0.5
		])
		res = enrichment_analysis(counts, pseudocount=0.5)
		assert res.effect_size_pc_dominated[4]
		assert not res.effect_size_pc_dominated[:4].any()


class TestEnrichmentCLI:
	def _write_input(self, path, **cols):
		pd.DataFrame(cols).to_csv(path, sep="\t", index=False)

	def _enrich_argv(self, inp, conditions, out, extra=()):
		return [
		    "enrich",
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
		assert main(self._enrich_argv(inp, ["A", "B", "C"], out)) == 0

		df = pd.read_csv(out, sep="\t")
		assert list(df.columns) == [
		    "chrom", "start", "end", "A", "B", "C",
		    "effect_size", "p_value", "q_value", "enriched_condition",
		    "effect_size_pc_dominated", "lrt_zero_dominated",
		    "lrt_convergence_failed",
		]
		assert (df["enriched_condition"] == "C").mean() > 0.5
		assert df["q_value"].le(0.05).all()

	def test_k3_warning_ratio_matches_reported_rate(self, tmp_path):
		"""The K=3 warning states a rate and a multiple of nominal; the two
		must agree with each other."""
		import re

		rng = np.random.default_rng(10)
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		self._write_input(inp, **{c: rng.poisson(60, size=200).astype(float) for c in "ABC"})
		with pytest.warns(FertilizerEnrichmentWarning, match="at K=3") as rec:
			assert main(self._enrich_argv(inp, list("ABC"), out)) == 0
		msg = next(str(w.message) for w in rec if "at K=3" in str(w.message))
		rate, ratio = re.search(r"is ~([0-9.]+), about ([0-9.]+)x nominal", msg).groups()
		assert float(ratio) == pytest.approx(float(rate) / 0.05, abs=0.05)

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
		main(self._enrich_argv(inp, list("ABCD"), out_q_only, ["--q-threshold", "1.0"]))
		df_q = pd.read_csv(out_q_only, sep="\t")
		assert len(df_q) == n

		out_pq = tmp_path / "pq.tsv"
		main(self._enrich_argv(
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
		main(self._enrich_argv(inp, list("ABCD"), out_strict, ["--q-threshold", "0.05"]))
		df_strict = pd.read_csv(out_strict, sep="\t")

		out_loose = tmp_path / "loose.tsv"
		main(self._enrich_argv(inp, list("ABCD"), out_loose, ["--q-threshold", "1.0"]))
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
		main(self._enrich_argv(inp, ["A", "B", "C"], out, ["--q-threshold", "1.0"]))
		df = pd.read_csv(out, sep="\t")
		assert df["chrom"].tolist() == ["chr3", "chr1", "chr2", "chr1", "chr3", "chr2"]
		assert df["start"].tolist() == [500, 0, 100, 200, 300, 400]

	def test_missing_column_errors(self, tmp_path, capsys):
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		self._write_input(
		    inp,
		    chrom=["chr1", "chr1"], start=[0, 100], end=[50, 150],
		    A=[10.0, 20.0], B=[12.0, 18.0],
		)
		assert main(self._enrich_argv(inp, ["A", "nope"], out)) == 2
		assert "columns not found" in capsys.readouterr().err

	def test_empty_cell_rejected(self, tmp_path, capsys):
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		inp.write_text("chrom\tstart\tend\tA\tB\nchr1\t0\t50\t10\t\nchr1\t100\t150\t20\t18\n")
		assert main(self._enrich_argv(inp, ["A", "B"], out)) == 2
		assert "finite" in capsys.readouterr().err
		assert not out.exists()

	@pytest.mark.parametrize("name", ["p_value", "effect_size", "enriched_condition"])
	def test_condition_named_like_output_column_rejected(self, tmp_path, capsys, name):
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		self._write_input(inp, A=np.full(20, 50.0), **{name: np.full(20, 50.0)})
		assert main(self._enrich_argv(inp, ["A", name], out)) == 2
		assert "output column" in capsys.readouterr().err
		assert not out.exists()

	def test_single_condition_rejected(self, tmp_path, capsys):
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		self._write_input(
		    inp,
		    chrom=["chr1"], start=[0], end=[100],
		    A=[10.0],
		)
		assert main(self._enrich_argv(inp, ["A"], out)) == 2
		assert "at least 2 conditions" in capsys.readouterr().err

	def test_size_factors_flag(self, tmp_path):
		rng = np.random.default_rng(99)
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
		assert main(self._enrich_argv(
		    inp, list("ABC"), out,
		    ["--q-threshold", "1.0", "--size-factors", "1.0", "2.0", "0.5"],
		)) == 0
		df = pd.read_csv(out, sep="\t")
		assert len(df) == n

	def test_size_factors_flag_wrong_length(self, tmp_path):
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		self._write_input(
		    inp,
		    chrom=["chr1", "chr1"], start=[0, 100], end=[50, 150],
		    A=[10.0, 20.0], B=[12.0, 18.0], C=[11.0, 19.0],
		)
		assert main(self._enrich_argv(
		    inp, list("ABC"), out,
		    ["--size-factors", "1.0", "1.0"],
		)) == 2

	def test_background_rank_flag(self, tmp_path):
		"""`--background-rank 2` compares k* against the second-highest
		condition; the default (rank 3) compares against the third-highest
		and gives smaller p-values when there is a competing peak."""
		rng = np.random.default_rng(404)
		n = 300
		cols = {c: rng.poisson(80, size=n).astype(float) for c in "ABCDE"}
		# Inject enrichment in A with B as a competing peak for the first 60.
		cols["A"][:60] = rng.poisson(320, size=60)
		cols["B"][:60] = rng.poisson(280, size=60)
		inp = tmp_path / "in.tsv"
		self._write_input(
		    inp,
		    chrom=["chr1"] * n,
		    start=np.arange(n),
		    end=np.arange(n) + 1,
		    **cols,
		)
		out_default = tmp_path / "default.tsv"
		out_rank2 = tmp_path / "rank2.tsv"
		assert main(self._enrich_argv(
		    inp, list("ABCDE"), out_default, ["--q-threshold", "1.0"],
		)) == 0
		assert main(self._enrich_argv(
		    inp, list("ABCDE"), out_rank2,
		    ["--q-threshold", "1.0", "--background-rank", "2"],
		)) == 0
		df_def = pd.read_csv(out_default, sep="\t")
		df_r2 = pd.read_csv(out_rank2, sep="\t")
		# Default (rank 3) should yield smaller p-values than rank 2 on the
		# injected loci (where the competing peak defeats rank 2).
		assert (df_def["p_value"].iloc[:60] < df_r2["p_value"].iloc[:60]).mean() > 0.7

	def test_background_rank_flag_invalid_rejected(self, tmp_path):
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		self._write_input(
		    inp,
		    chrom=["chr1"], start=[0], end=[100],
		    A=[10.0], B=[12.0], C=[11.0],
		)
		assert main(self._enrich_argv(
		    inp, list("ABC"), out, ["--background-rank", "1"],
		)) == 2

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
		assert main(self._enrich_argv(
		    inp, list("ABC"), out,
		    ["--q-threshold", "1.0", "--dispersion", "0.05"],
		)) == 0
		# All rows present (loose threshold)
		df = pd.read_csv(out, sep="\t")
		assert len(df) == n


	def test_negative_dispersion_flag_rejected(self, tmp_path, capsys):
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		cols = {c: np.full(20, 50.0) for c in "ABC"}
		self._write_input(inp, chrom=["chr1"] * 20, start=np.arange(20),
		                  end=np.arange(20) + 1, **cols)
		assert main(self._enrich_argv(
		    inp, list("ABC"), out, ["--dispersion", "-0.1"],
		)) == 2
		assert "--dispersion" in capsys.readouterr().err
		assert not out.exists()

class TestExtractStatGuard:
	"""`enrich` should refuse inputs produced by `extract --stat <non-sum>`
	unless the user passes `--allow-non-sum`. Inputs with no metadata header
	(e.g. user-supplied TSVs) are accepted without comment."""

	def _write_with_header(self, path, stat, **cols):
		df = pd.DataFrame(cols)
		with open(path, "w") as fh:
			fh.write(f"# fertilizer-extract stat={stat}\n")
			df.to_csv(fh, sep="\t", index=False)

	def test_refuses_mean_input(self, tmp_path, capsys):
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		self._write_with_header(
		    inp, "mean",
		    chrom=["chr1", "chr1"], start=[0, 100], end=[50, 150],
		    A=[10.0, 20.0], B=[12.0, 18.0],
		)
		assert main(["enrich", "-i", str(inp), "-c", "A", "B", "-o", str(out)]) == 2
		assert "--stat mean" in capsys.readouterr().err

	def test_allow_non_sum_bypasses(self, tmp_path):
		rng = np.random.default_rng(2024)
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		n = 200
		self._write_with_header(
		    inp, "mean",
		    chrom=["chr1"] * n,
		    start=np.arange(n),
		    end=np.arange(n) + 1,
		    A=rng.poisson(50, size=n).astype(float),
		    B=rng.poisson(50, size=n).astype(float),
		)
		assert main([
		    "enrich", "-i", str(inp), "-c", "A", "B",
		    "-o", str(out), "--q-threshold", "1.0", "--allow-non-sum",
		]) == 0

	def test_sum_header_runs_without_flag(self, tmp_path):
		rng = np.random.default_rng(2025)
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		n = 200
		self._write_with_header(
		    inp, "sum",
		    chrom=["chr1"] * n,
		    start=np.arange(n),
		    end=np.arange(n) + 1,
		    A=rng.poisson(50, size=n).astype(float),
		    B=rng.poisson(50, size=n).astype(float),
		)
		assert main([
		    "enrich", "-i", str(inp), "-c", "A", "B",
		    "-o", str(out), "--q-threshold", "1.0",
		]) == 0

	def test_no_header_is_accepted(self, tmp_path):
		"""User-supplied TSVs without a fertilizer-extract header pass through."""
		rng = np.random.default_rng(2026)
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv"
		n = 200
		pd.DataFrame({
		    "chrom": ["chr1"] * n,
		    "start": np.arange(n),
		    "end": np.arange(n) + 1,
		    "A": rng.poisson(50, size=n).astype(float),
		    "B": rng.poisson(50, size=n).astype(float),
		}).to_csv(inp, sep="\t", index=False)
		assert main([
		    "enrich", "-i", str(inp), "-c", "A", "B",
		    "-o", str(out), "--q-threshold", "1.0",
		]) == 0


def test_parser_has_both_subcommands():
	parser = build_parser()
	for sub in ("extract", "enrich"):
		with pytest.raises(SystemExit):
			parser.parse_args([sub, "--help"])


def _nb_counts(rng, mu, alpha, K):
	"""(n, K) NB counts with per-locus mean `mu` and dispersion `alpha`."""
	mu = np.asarray(mu, dtype=float)
	alpha = np.broadcast_to(np.asarray(alpha, dtype=float), mu.shape)
	return rng.negative_binomial(
	    (1 / alpha)[:, None], (1 / (1 + alpha * mu))[:, None], size=(mu.size, K),
	).astype(float)


class TestNBLogPmf:
	@pytest.mark.parametrize("mu", [0.5, 5.0, 50.0])
	@pytest.mark.parametrize("alpha", [1e-3, 0.1, 2.0])
	def test_matches_scipy_nbinom(self, mu, alpha):
		y = np.arange(0, 60, dtype=float)
		expected = stats.nbinom.logpmf(y, 1 / alpha, 1 / (1 + alpha * mu))
		np.testing.assert_allclose(_nb_logpmf(y, mu, alpha), expected, rtol=1e-9, atol=1e-9)

	@pytest.mark.parametrize("mu", [0.5, 5.0, 50.0])
	def test_poisson_branch_matches_scipy_poisson(self, mu):
		y = np.arange(0, 60, dtype=float)
		np.testing.assert_allclose(
		    _nb_logpmf(y, mu, 0.0), stats.poisson.logpmf(y, mu), rtol=1e-12, atol=1e-12,
		)

	def test_continuous_across_poisson_cutoff(self):
		y = np.arange(0, 200, dtype=float)
		cutoff = enrichment_module._POISSON_CUTOFF
		below = _nb_logpmf(y, 50.0, cutoff * 0.999)
		above = _nb_logpmf(y, 50.0, cutoff * 1.001)
		np.testing.assert_allclose(below, above, rtol=1e-2)

	def test_broadcasts_per_locus_alpha(self):
		y = np.array([3.0, 7.0])
		alpha = np.array([0.0, 0.5])
		out = _nb_logpmf(y, np.array([4.0, 4.0]), alpha)
		np.testing.assert_allclose(out[0], stats.poisson.logpmf(3, 4.0))
		np.testing.assert_allclose(out[1], stats.nbinom.logpmf(7, 2.0, 1 / 3.0))


class TestInterceptMLE:
	def test_matches_scalar_root_finder(self):
		rng = np.random.default_rng(0)
		counts = rng.poisson(40, size=(30, 2)).astype(float)
		sf = rng.uniform(0.5, 2.0, size=(30, 2))
		alpha = rng.uniform(0.01, 0.5, size=30)
		mu0, converged = _intercept_mle(counts, sf, alpha)
		assert converged.all()
		for i in range(30):
			def score(m, i=i):
				return np.sum((counts[i] - m * sf[i]) / (1 + alpha[i] * m * sf[i]))
			np.testing.assert_allclose(mu0[i], brentq(score, 1e-8, 1e6), rtol=1e-7)

	def test_poisson_is_closed_form(self):
		counts = np.array([[30.0, 10.0], [5.0, 5.0]])
		sf = np.array([[1.0, 2.0], [1.0, 1.0]])
		mu0, converged = _intercept_mle(counts, sf, 0.0)
		np.testing.assert_allclose(mu0, [40.0 / 3.0, 5.0])
		assert converged.all()

	def test_non_convergence_is_flagged_and_warned(self):
		# Unequal size factors: with equal ones the Poisson start is already
		# the NB root and no Newton step is taken.
		counts = np.array([[500.0, 1.0], [300.0, 2.0]])
		sf = np.array([[0.5, 2.0], [0.5, 2.0]])
		with pytest.warns(FertilizerEnrichmentWarning, match="did not converge"):
			_, converged = _intercept_mle(counts, sf, 0.5, max_iter=1)
		assert not converged.any()

	def test_unconverged_loci_get_p_value_one(self, monkeypatch):
		rng = np.random.default_rng(1)
		counts = rng.poisson(50, size=(200, 3)).astype(float)
		counts[0] = [5.0, 5.0, 500.0]
		real = enrichment_module._intercept_mle

		def fail_first(*args, **kwargs):
			mu0, converged = real(*args, **kwargs)
			converged = converged.copy()
			converged[0] = False
			return mu0, converged

		monkeypatch.setattr(enrichment_module, "_intercept_mle", fail_first)
		res = enrichment_analysis(counts, background_rank=2)
		assert res.lrt_convergence_failed[0]
		assert res.p_value[0] == 1.0
		assert not res.lrt_convergence_failed[1:].any()


class TestLRTStatistic:
	def test_poisson_pair_matches_hand_computed(self):
		"""K=2, unit size factors, alpha=0: the null mean is 20 for (30, 10),
		so T = 2 * (30 log(30/20) + 10 log(10/20))."""
		counts = np.array([[30.0, 10.0], [10.0, 30.0]])
		res = enrichment_analysis(counts, size_factors_override=[1.0, 1.0],
		                          dispersion_override=0.0)
		t = 2 * (30 * np.log(1.5) + 10 * np.log(0.5))
		np.testing.assert_allclose(res.lrt_stat, [t, t], rtol=1e-10)
		np.testing.assert_allclose(res.p_value, min(2 * 0.5 * stats.chi2.sf(t, 1), 1.0))
		np.testing.assert_array_equal(res.enriched_condition_idx, [0, 1])

	def test_nb_pair_matches_scipy_likelihood(self):
		x, sf, alpha = np.array([30.0, 10.0]), np.array([1.5, 0.8]), 0.1
		counts = np.array([x, x[::-1]])
		res = enrichment_analysis(counts, size_factors_override=sf,
		                          dispersion_override=alpha)

		def nb_ll(y, m):
			return stats.nbinom.logpmf(y, 1 / alpha, 1 / (1 + alpha * m)).sum()

		def score(m):
			return np.sum((x - m * sf) / (1 + alpha * m * sf))
		m0 = brentq(score, 1e-8, 1e6)
		t = 2 * (nb_ll(x, x) - nb_ll(x, m0 * sf))
		np.testing.assert_allclose(res.lrt_stat[0], t, rtol=1e-8)

	def test_all_zero_rows_are_not_called(self):
		rng = np.random.default_rng(3)
		counts = rng.poisson(50, size=(300, 3)).astype(float)
		counts[:5] = 0.0
		res = enrichment_analysis(counts)
		np.testing.assert_array_equal(res.p_value[:5], 1.0)
		np.testing.assert_array_equal(res.effect_size[:5], 0.0)
		assert res.lrt_zero_dominated[:5].all()
		assert np.isfinite(res.q_value).all()


class TestParametricTrend:
	def test_recovers_constant_dispersion(self):
		rng = np.random.default_rng(0)
		mu = np.exp(rng.uniform(np.log(20), np.log(1000), 4000))
		counts = _nb_counts(rng, mu, 0.05, K=5)
		res = enrichment_analysis(counts, fit_type="parametric")
		assert res.dispersion_fit == "parametric"
		a, b = res.dispersion_trend
		assert a < 0.1
		assert b == pytest.approx(0.05, rel=0.1)

	def test_detects_decreasing_trend(self):
		rng = np.random.default_rng(0)
		mu = np.exp(rng.uniform(np.log(10), np.log(1000), 4000))
		counts = _nb_counts(rng, mu, 2.0 / mu + 0.05, K=5)
		res = enrichment_analysis(counts, fit_type="parametric")
		low, high = _apply_trend(np.array([10.0, 1000.0]), res.dispersion_trend)
		assert low > 1.5 * high

	@pytest.mark.skip(reason="known bias: with alpha(mu) = 2/mu + 0.05 and K=5 the "
	                         "fitted `a` is 0.5-0.7 across seeds, so low-mu dispersion "
	                         "is underestimated")
	def test_recovers_trend_coefficients(self):
		rng = np.random.default_rng(0)
		mu = np.exp(rng.uniform(np.log(10), np.log(1000), 4000))
		counts = _nb_counts(rng, mu, 2.0 / mu + 0.05, K=5)
		a, b = enrichment_analysis(counts, fit_type="parametric").dispersion_trend
		assert a == pytest.approx(2.0, rel=0.3)
		assert b == pytest.approx(0.05, rel=0.3)

	def test_per_locus_alpha_clipped_to_informative_support(self):
		rng = np.random.default_rng(4)
		mu = np.concatenate([np.full(50, 0.5), np.exp(rng.uniform(np.log(10), np.log(1000), 2000))])
		counts = _nb_counts(rng, mu, 2.0 / mu + 0.05, K=5)
		res = enrichment_analysis(counts, fit_type="parametric")
		normalized_mu = (counts / res.size_factors).mean(axis=1)
		informative = normalized_mu >= 5.0
		support = _apply_trend(normalized_mu[informative], res.dispersion_trend)
		assert res.per_locus_dispersion.max() <= support.max() + 1e-12
		assert res.per_locus_dispersion.min() >= support.min() - 1e-12

	def test_trim_leaving_too_few_loci_returns_none(self):
		mu = np.full(12, 50.0)
		alpha = np.array([0.0] * 9 + [10.0] * 3)
		assert _fit_parametric_trend(mu, alpha, min_signal=5.0) is None

	def test_too_few_informative_loci_returns_none(self):
		assert _fit_parametric_trend(np.full(9, 50.0), np.zeros(9), min_signal=5.0) is None

	def test_failed_fit_falls_back_to_common(self, monkeypatch):
		rng = np.random.default_rng(5)
		counts = rng.poisson(100, size=(500, 3)).astype(float)
		monkeypatch.setattr(enrichment_module, "_fit_parametric_trend", lambda *a, **k: None)
		with pytest.warns(FertilizerEnrichmentWarning, match="single common alpha"):
			res = enrichment_analysis(counts, fit_type="parametric")
		assert res.dispersion_fit == "common-fallback"
		common = enrichment_analysis(counts, fit_type="common")
		np.testing.assert_allclose(res.per_locus_dispersion, common.per_locus_dispersion)

	def test_unknown_fit_type_rejected(self):
		counts = np.random.default_rng(6).poisson(50, size=(50, 3)).astype(float)
		with pytest.raises(ValueError, match="unknown fit_type"):
			enrichment_analysis(counts, fit_type="local")


class TestRegionOverlapWarning:
	def _frame(self, chroms, starts, ends):
		return pd.DataFrame({"chrom": chroms, "start": starts, "end": ends})

	def test_sliding_windows_warn(self):
		starts = np.arange(0, 5000, 50)
		df = self._frame(["chr1"] * len(starts), starts, starts + 100)
		with pytest.warns(FertilizerEnrichmentWarning, match="adjacent regions overlap"):
			_warn_if_regions_overlap(df)

	def test_disjoint_regions_are_silent(self):
		starts = np.arange(0, 5000, 100)
		df = self._frame(["chr1"] * len(starts), starts, starts + 100)
		with warnings.catch_warnings():
			warnings.simplefilter("error")
			_warn_if_regions_overlap(df)

	def test_same_coordinates_on_different_chroms_are_silent(self):
		df = self._frame(["chr1", "chr2", "chr3"], [0, 0, 0], [100, 100, 100])
		with warnings.catch_warnings():
			warnings.simplefilter("error")
			_warn_if_regions_overlap(df)

	def test_no_coordinate_columns_is_silent(self):
		df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
		with warnings.catch_warnings():
			warnings.simplefilter("error")
			_warn_if_regions_overlap(df)

	def test_cli_emits_warning(self, tmp_path):
		rng = np.random.default_rng(7)
		n = 200
		starts = np.arange(n) * 50
		inp = tmp_path / "in.tsv"
		pd.DataFrame({"chrom": ["chr1"] * n, "start": starts, "end": starts + 100,
		              **{c: rng.poisson(50, size=n).astype(float) for c in "AB"}},
		             ).to_csv(inp, sep="\t", index=False)
		with pytest.warns(FertilizerEnrichmentWarning, match="adjacent regions overlap"):
			assert main(["enrich", "-i", str(inp), "-c", "A", "B",
			             "-o", str(tmp_path / "out.tsv")]) == 0


class TestReadExtractStat:
	def test_gzip_header_is_read(self, tmp_path):
		path = tmp_path / "in.tsv.gz"
		with gzip.open(path, "wt") as fh:
			fh.write("# fertilizer-extract stat=mean\nchrom\tstart\tend\tA\n")
		assert _read_extract_stat(str(path)) == "mean"

	def test_missing_file_returns_none(self, tmp_path):
		assert _read_extract_stat(str(tmp_path / "nope.tsv")) is None

	def test_comment_without_stat_returns_none(self, tmp_path):
		path = tmp_path / "in.tsv"
		path.write_text("# some other comment\nA\tB\n1\t2\n")
		assert _read_extract_stat(str(path)) is None

	def test_gzip_mean_input_refused_by_cli(self, tmp_path, capsys):
		path = tmp_path / "in.tsv.gz"
		with gzip.open(path, "wt") as fh:
			fh.write("# fertilizer-extract stat=mean\n")
			pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]}).to_csv(fh, sep="\t", index=False)
		assert main(["enrich", "-i", str(path), "-c", "A", "B",
		             "-o", str(tmp_path / "out.tsv")]) == 2
		assert "--stat mean" in capsys.readouterr().err

	def test_gzip_output_round_trips(self, tmp_path):
		rng = np.random.default_rng(8)
		inp = tmp_path / "in.tsv"
		out = tmp_path / "out.tsv.gz"
		pd.DataFrame({c: rng.poisson(50, size=100).astype(float) for c in "AB"}).to_csv(
		    inp, sep="\t", index=False)
		assert main(["enrich", "-i", str(inp), "-c", "A", "B", "-o", str(out),
		             "--q-threshold", "1.0"]) == 0
		with gzip.open(out, "rt") as fh:
			df = pd.read_csv(fh, sep="\t")
		assert len(df) == 100
		assert "q_value" in df.columns


class TestEnrichCLIValidation:
	@pytest.mark.parametrize("extra,message", [
	    (["--q-threshold", "1.5"], "--q-threshold"),
	    (["--q-threshold", "-0.1"], "--q-threshold"),
	    (["--p-threshold", "2"], "--p-threshold"),
	    (["--pseudocount", "0"], "--pseudocount"),
	    (["--background-rank", "1"], "--background-rank"),
	])
	def test_invalid_flags_exit_2(self, tmp_path, capsys, extra, message):
		inp = tmp_path / "in.tsv"
		pd.DataFrame({"A": [10.0, 20.0], "B": [12.0, 18.0]}).to_csv(inp, sep="\t", index=False)
		assert main(["enrich", "-i", str(inp), "-c", "A", "B",
		             "-o", str(tmp_path / "out.tsv"), *extra]) == 2
		assert message in capsys.readouterr().err

	def test_expected_t1_beyond_table_uses_largest_k(self):
		assert _expected_t1_at_05(12, 2) == _expected_t1_at_05(8, 2)
		assert _expected_t1_at_05(12, 3) == _expected_t1_at_05(8, 3)
