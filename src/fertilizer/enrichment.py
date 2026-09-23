"""NB-GLM-based enrichment analysis across >=2 conditions.

This module implements a DESeq2-inspired negative-binomial GLM likelihood-
ratio test for the single-replicate-per-condition setting this package
targets. The algorithm follows the shape of DESeq2 (Love, Huber, Anders,
2014) — size factors, dispersion trend across loci, NB-GLM LRT — but is a
deliberate simplification, and is restricted to detecting **enrichment**
of one condition relative to the others (loci where one condition is
depleted are not called). See "Differences from DESeq2" below for the
specific places where this code and DESeq2 will disagree, and why.

Pipeline per locus i, across K conditions (one observation per condition):

1. Median-of-ratios size factors s_j across conditions, computed jointly
   using loci with positive signal in every condition.
2. Per-locus method-of-moments dispersion
       alpha_i^MoM = max((V_i - mu_i) / mu_i^2, 0)
   where mu_i and V_i are the mean and sample variance (ddof=1) of the
   size-factor-normalized signal across conditions.
3. A dispersion estimate shared across loci. Default is a single *common*
   alpha taken as the median of per-locus MoM estimates over informative
   loci (robust to the minority of truly-enriched loci pulling their
   own MoM estimate up). A parametric trend `alpha(mu) = a / mu + b`
   (DESeq2's functional form) is available via `fit_type="parametric"`;
   its fit robustly trims upper-tail MoM estimates before weighted least
   squares. The final per-locus alpha is applied directly, not shrunk via
   empirical Bayes as in DESeq2 — we replace.
4. Per locus, identify the top condition by size-factor-normalized signal:
   k* = argmax_j (X_ij / s_j), and a background condition k_bg = the
   condition at rank `background_rank` (default 3) in the sort by
   normalized signal — rank 1 is k*, rank 2 is the second-highest, rank r
   is the r-th largest. The one-sided LRT compares (k*, k_bg) only; the
   other K - 2 conditions enter both models as saturated nuisance
   parameters and cancel from the likelihood ratio. Comparing against a
   rank > 2 makes the test robust to "competing peaks": at the default of
   3, one second condition can be elevated without depressing the LRT.
   The design also makes the test ignore depletion patterns (where the
   conditions at ranks 1..k_bg all sit at the high level and look
   essentially equal under the test).
5. Null model on the pair: mu_{k*} = mu_{k_bg} = mu_top, fit by
   intercept-only NB MLE on the two observations (X_{k*}, X_{k_bg}) with
   size factors (s_{k*}, s_{k_bg}). Vectorized Newton's method on the
   score equation
       sum_{j in {k*, k_bg}} (X_j - mu_top s_j) / (1 + alpha mu_top s_j) = 0
   starting from the Poisson closed-form MLE.
6. Alternative on the pair: saturated. mu_{k*} = X_{k*}/s_{k*},
   mu_{k_bg} = X_{k_bg}/s_{k_bg}, constrained mu_{k*} > mu_{k_bg} (always
   satisfied by construction since k* is the argmax and k_bg sits at a
   lower rank).
7. LRT_i = 2 * (logL_alt - logL_null) over the pair, chi-bar-squared({0, 1})
   under H0 for fixed k*. We use the upper half-chi^2(1) tail (factor of
   1/2), then Bonferroni-correct by K to account for picking k* as the
   empirical argmax. The choice of k_bg is deterministic given the
   condition ordering and adds no extra Bonferroni cost. Final per-locus
   p-value is min(K * 0.5 * chi2_1_sf(LRT), 1).
8. Benjamini-Hochberg q-values across loci.

The effect size reported is
`log2((X_{i,k*}/s_{k*}) + pc) - log2(mean_{j!=k*}(X_{i,j}/s_j) + pc)`:
the log2 fold change of the enriched condition vs the mean of the others
on the size-factor-normalized scale. Always non-negative by construction
(k* is the argmax). Note that this is a summary for users; the test
itself is computed only from the (k*, k_bg) pair.

Differences from DESeq2 (non-exhaustive, but the ones that matter):

- **Per-locus dispersion estimator.** DESeq2 uses a Cox-Reid adjusted
  profile likelihood NB-GLM MLE per locus. We use method-of-moments. MoM
  is less efficient per locus (higher variance on a single estimate) but
  is consistent, has no convergence failures, and is perfectly adequate
  once we pool via a trend.
- **Shrinkage of per-locus dispersion toward the trend.** DESeq2 applies
  empirical-Bayes shrinkage with a log-normal prior fit to the residuals
  around the trend. We simply use the trend value as the final per-locus
  dispersion (equivalent to infinite shrinkage). This is robust in the
  small-N setting but cannot recover genuinely heterogeneous dispersion.
- **Dispersion outlier retention.** DESeq2 detects loci with MLE well
  above the trend and keeps their MLE rather than shrinking. We don't —
  every locus uses the trend.
- **Observation-level outlier detection.** DESeq2 uses Cook's distance to
  flag and optionally refit excluding outliers. We don't.
- **Log2 fold change shrinkage.** DESeq2 optionally shrinks LFC estimates
  via apeglm/ashr; we report the raw log2 fold change of the enriched
  condition vs the mean of the others as the effect size.
- **Arbitrary designs / two-sided LRT.** DESeq2's LRT supports `full` vs
  `reduced` formulas of arbitrary design matrices and is two-sided (it
  fires on both enrichment and depletion). We hard-code a 1-df one-sided
  LRT comparing the top condition against a background condition at a
  user-chosen rank (`background_rank`, default 3, which tolerates one
  competing peak), with Bonferroni x K for picking the top by argmax.
  Other K - 2 conditions enter both models as saturated nuisance and
  cancel. Loci where one condition is *depleted* relative to the others
  (the top conditions then all sit at the high level and look
  indistinguishable) are not called.
- **Independent filtering.** DESeq2 filters low-count loci out of
  multiple-testing correction to maximize power at a given alpha. We don't
  (but users can set `--min-signal` high and post-filter themselves).
- **Integer counts.** DESeq2 is designed for integer counts; the NB
  likelihood here uses `scipy.special.gammaln` and is numerically correct
  for any non-negative float input. This matches the common practice of
  passing fractional RSEM/salmon expected counts to DESeq2 via tximport,
  and is required here because bigWig-extracted signal is real-valued.

Background-rank knob:

- `background_rank=3` (default): k* compared against the 3rd-ranked
  condition. Tolerates one competing peak (rank 2 can be elevated
  without depressing the LRT) and is K-independent.
- `background_rank=2`: k* compared against the second-highest condition.
  Maximally powerful when only one condition is active at a locus, but
  fragile when a second condition is also elevated.
- Larger values tolerate more competing peaks at the cost of comparing
  against an increasingly-low background condition. Capped to K.

Dispersion model knobs:

- `fit_type="common"` (default): one alpha everywhere, taken as the median
  of per-locus MoM estimates over informative loci. Robust; ignores any
  mean-dispersion relationship.
- `fit_type="parametric"`: fits `alpha(mu) = a/mu + b` by robust weighted
  least squares on the per-locus MoM estimates. Use when dispersion clearly
  trends with mean signal and enough informative loci are available.
- `fit_type="zero"`: forces Poisson (alpha=0) at every locus. Diagnostic
  only — strictly anti-conservative if real overdispersion exists. A
  warning is emitted on every call.
- `dispersion_override=<float>`: forces a fixed alpha for every locus,
  bypassing `fit_type` and `dispersion_min_signal` entirely. Useful for
  sensitivity analyses (re-run at two values to see how much calls move).
"""

from __future__ import annotations

import argparse
import gzip
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import gammaln
from scipy.stats import chi2

__all__ = [
    "EnrichmentResult",
    "FertilizerEnrichmentWarning",
    "bh_qvalues",
    "enrichment_analysis",
    "run_enrich",
    "size_factors",
    "size_factors_with_n",
]


class FertilizerEnrichmentWarning(UserWarning):
    """Quality warnings emitted from the enrichment pipeline."""


@dataclass
class EnrichmentResult:
    size_factors: np.ndarray                  # (n_conditions,)
    n_loci_for_size_factors: int              # loci with positive signal in every condition
    dispersion_fit: str                       # label for how alpha was derived
    dispersion_trend: tuple[float, float]     # (a, b) s.t. alpha(mu) = a/mu + b
    per_locus_dispersion: np.ndarray          # (n_loci,) final alpha used
    effect_size: np.ndarray                   # (n_loci,) log2(enriched / mean-of-rest)
    lrt_stat: np.ndarray                      # (n_loci,) one-sided LRT test statistic
    p_value: np.ndarray                       # (n_loci,) Bonferroni-corrected
    q_value: np.ndarray                       # (n_loci,)
    enriched_condition_idx: np.ndarray        # (n_loci,) argmax(X / s); always
                                              # populated, so only meaningful for
                                              # loci that pass a significance
                                              # threshold (on a null locus it is
                                              # just the column highest under noise)
    effect_size_pc_dominated: np.ndarray      # (n_loci,) bool; true when some X_j/s_j < pc
    lrt_zero_dominated: np.ndarray            # (n_loci,) bool; true when X_top==0 or X_bg==0
    lrt_convergence_failed: np.ndarray        # (n_loci,) bool; true when null-fit NB MLE failed
    background_rank: int                      # k actually used (== min(requested, K))


def size_factors(counts: np.ndarray) -> np.ndarray:
    """DESeq2 median-of-ratios size factors.

    `counts` is an (n_loci, n_samples) array. Only loci with positive
    signal in every sample contribute. Raises ValueError if fewer than 2
    such loci exist. The number of contributing loci is also available
    via `size_factors_with_n` if you need to log it.
    """
    sf, _ = size_factors_with_n(counts)
    return sf


def size_factors_with_n(counts: np.ndarray) -> tuple[np.ndarray, int]:
    """Same as `size_factors` but also returns the number of loci with
    positive signal in every sample (i.e., the loci that contributed)."""
    counts = np.asarray(counts, dtype=np.float64)
    positive = (counts > 0).all(axis=1)
    n_positive = int(positive.sum())
    if n_positive < 2:
        raise ValueError(
            "at least 2 loci with positive signal in every sample are "
            "required to compute size factors"
        )
    log_counts = np.log(counts[positive])
    log_geom_mean = log_counts.mean(axis=1, keepdims=True)
    sf = np.exp(np.median(log_counts - log_geom_mean, axis=0))
    return sf, n_positive


def bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values for an array of p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    sorted_q = q[order]
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(sorted_q, 0.0, 1.0)
    return out


# NB log-pmf with Poisson fallback below this alpha, where r = 1/alpha
# becomes large enough that `gammaln(y+r) - gammaln(r)` loses precision
# (catastrophic cancellation). At alpha = 1e-4, r = 1e4, gammaln(1e4) is
# ~8.2e4 with ~1e-14 relative error, so differences of order y*log(r)~10
# are still ~10 decimal digits accurate. Poisson is numerically identical
# to NB in this regime.
_POISSON_CUTOFF = 1e-4


def _nb_logpmf(y: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """NB(mu, alpha) log-pmf with Var(Y) = mu + alpha * mu^2.

    Accepts float-valued `y` (region means are not integers). For
    alpha < _POISSON_CUTOFF, falls back to the Poisson log-pmf to avoid
    numerical cancellation in gammaln differences at very small alpha.
    Broadcasting applies.
    """
    mu = np.maximum(mu, 1e-300)
    poisson_ll = y * np.log(mu) - mu - gammaln(y + 1.0)

    alpha_safe = np.maximum(alpha, _POISSON_CUTOFF)
    r = 1.0 / alpha_safe
    nb_ll = (
        gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
        + y * np.log(mu / (mu + r))
        + r * np.log(r / (mu + r))
    )
    use_poisson = np.asarray(alpha) < _POISSON_CUTOFF
    return np.where(use_poisson, poisson_ll, nb_ll)


def _mom_dispersion(
    normalized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-locus method-of-moments dispersion estimate with median-bias
    correction applied per locus.

    Under the Gaussian approximation V_i ~ sigma_i^2 * chi^2(K-1)/(K-1) with
    sigma_i^2 = mu_i + alpha_i * mu_i^2, so median(V_i) = sigma_i^2 * c_K where
    c_K = median(chi^2(K-1)) / (K-1). Rearranging,
        alpha_i = (V_i / c_K - mu_i) / mu_i^2
    is an approximately median-unbiased per-locus estimator of alpha_i (the
    (c_K - 1)/mu bias term that contaminates (V - mu)/mu^2 is absorbed into
    V/c_K). Downstream pooling via the median of these per-locus values is
    then unbiased to leading order at every mu, not only at large mu.

    Returns (alpha_hat, mu, V). alpha_hat is UNCLIPPED (can be negative);
    downstream consumers clip at 0 where positivity is required. Un-clipped
    estimates are needed to avoid biasing the median-over-loci estimator
    upward when aggregating.
    """
    K = normalized.shape[1]
    mu = normalized.mean(axis=1)
    V = normalized.var(axis=1, ddof=1)
    c_K = _median_bias_correction(K)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(mu > 0, (V / c_K - mu) / mu ** 2, 0.0)
    return alpha, mu, V


def _median_bias_correction(K: int) -> float:
    """Scale factor for median(sample_variance/sigma^2) with K observations.

    Under a Gaussian-approximation null, (K-1)*V/sigma^2 ~ chi^2(K-1), so
    median(V)/sigma^2 = median(chi^2(K-1))/(K-1). The median of a chi^2
    is strictly less than its mean at small df, so median-of-MoM-alpha
    underestimates the true dispersion; dividing by this factor undoes
    the leading-order bias. Exact only at large mu; still helpful at
    moderate mu where dispersion_min_signal cuts off the low end.

    K=2 -> 0.455, K=3 -> 0.693, K=5 -> 0.839, K=8 -> 0.907.
    """
    if K < 2:
        return 1.0
    return float(chi2.median(df=K - 1) / (K - 1))


def _fit_parametric_trend(
    mu: np.ndarray,
    alpha_mom: np.ndarray,
    min_signal: float,
    trim_mad_k: float = 3.0,
) -> tuple[float, float] | None:
    """Fit alpha(mu) = a/mu + b by weighted least squares on robust-trimmed data.

    `alpha_mom` is expected to already carry the per-locus median-bias
    correction (see `_mom_dispersion`), so the residual (c_K - 1)/mu bias that
    would otherwise contaminate the `a` coefficient is absent; `a` is free to
    model genuine low-mu dispersion heterogeneity.

    MoM alpha for truly-enriched loci is biased upward (mean heterogeneity
    across conditions contaminates the sample variance), so an un-trimmed
    least-squares fit is pulled toward a few positive outliers when the null
    majority is at alpha ~ 0. We pre-trim loci whose MoM alpha exceeds
    median + trim_mad_k * MAD before fitting. Returns None if fewer than 10
    informative loci remain.

    Weights are proportional to sqrt(mu) — higher-mu loci have lower variance
    on their MoM alpha estimate. Non-negativity bounds on (a, b) via
    `least_squares`.
    """
    mask = np.isfinite(alpha_mom) & (mu >= min_signal) & (mu > 0)
    if mask.sum() < 10:
        return None
    mu_m = mu[mask]
    alpha_m = alpha_mom[mask]

    # Robust pre-trim: drop the upper tail of MoM alpha estimates, which is
    # where enriched loci land when the null majority is near alpha = 0.
    med = float(np.median(alpha_m))
    mad = float(np.median(np.abs(alpha_m - med)))
    cutoff = med + trim_mad_k * max(mad, 1e-6)
    keep = alpha_m <= cutoff
    if keep.sum() < 10:
        return None
    mu_k = mu_m[keep]
    alpha_k = alpha_m[keep]
    weights = np.sqrt(mu_k)

    def residuals(params: np.ndarray) -> np.ndarray:
        a, b = params
        return weights * (alpha_k - (a / mu_k + b))

    init = np.array([1.0, max(float(np.median(alpha_k)), 1e-4)])
    try:
        result = least_squares(
            residuals, x0=init,
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            max_nfev=500,
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
    if not result.success and result.status <= 0:
        return None
    return float(result.x[0]), float(result.x[1])


def _apply_trend(mu: np.ndarray, trend: tuple[float, float]) -> np.ndarray:
    """Evaluate alpha = a/mu + b at each locus mean."""
    a, b = trend
    safe_mu = np.where(mu > 0, mu, 1.0)
    alpha = np.where(mu > 0, a / safe_mu + b, b)
    return np.clip(alpha, 0.0, None)


def _intercept_mle(
    counts: np.ndarray,
    sf: np.ndarray,
    alpha: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Intercept-only NB GLM MLE per locus on the (k*, k_bg) sub-arrays.

    Solves sum_j (X_ij - mu_0 s_ij) / (1 + alpha_i mu_0 s_ij) = 0 for
    mu_0 per locus, via vectorized Newton's method starting from the
    Poisson closed-form MLE. The score function is monotone decreasing,
    so Newton's method converges quickly with no stepsize control.

    `counts` and `sf` are both (n_loci, 2) — one row per locus, holding
    the values for that locus's (k*, k_bg) pair with their respective size
    factors. `alpha` is (n_loci,) or scalar. Returns (mu_0, converged),
    both of shape (n_loci,). `converged[i]` is False for loci whose final
    Newton step did not fall below `tol`; downstream consumers should treat
    those loci's LRT statistic as unreliable.
    """
    counts = np.asarray(counts, dtype=np.float64)
    sf = np.asarray(sf, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 0:
        alpha = np.full(counts.shape[0], float(alpha))

    mu0 = counts.sum(axis=1) / sf.sum(axis=1)
    mu0 = np.maximum(mu0, 1e-20)

    needs_iter = alpha > _POISSON_CUTOFF
    if not needs_iter.any():
        # Poisson closed form is exact; everything converged trivially.
        return mu0, np.ones(counts.shape[0], dtype=bool)

    alpha_c = alpha[:, None]
    per_locus_change = np.full(counts.shape[0], np.inf)
    for _ in range(max_iter):
        mu_ij = mu0[:, None] * sf
        denom = 1.0 + alpha_c * mu_ij
        f = ((counts - mu_ij) / denom).sum(axis=1)
        # score derivative is always negative; clamp to a strict upper bound
        # of -1e-20 so division never blows up
        f_prime = -((sf * (1.0 + alpha_c * counts)) / denom ** 2).sum(axis=1)
        f_prime = np.minimum(f_prime, -1e-20)
        step = np.where(needs_iter, f / f_prime, 0.0)
        mu0_new = np.maximum(mu0 - step, 1e-20)
        per_locus_change = np.abs(mu0_new - mu0)
        mu0 = mu0_new
        if float(per_locus_change.max()) < tol:
            break

    # Loci that didn't need NB iteration are trivially converged; others must
    # have driven their Newton step below `tol`.
    converged = ~needs_iter | (per_locus_change < tol)
    n_failed = int((~converged).sum())
    if n_failed > 0:
        worst = int(np.argmax(per_locus_change))
        warnings.warn(
            f"intercept-only NB MLE did not converge for {n_failed} locus/loci "
            f"after {max_iter} iterations (worst locus index {worst}, "
            f"final step {float(per_locus_change[worst]):.3g}); p-values for "
            "these loci will be set to 1.0 and `lrt_convergence_failed=True` "
            "in the result.",
            FertilizerEnrichmentWarning, stacklevel=3,
        )
    return mu0, converged


def enrichment_analysis(
    counts: np.ndarray,
    pseudocount: float = 0.5,
    fit_type: str = "common",
    dispersion_min_signal: float = 5.0,
    dispersion_override: float | None = None,
    size_factor_warn_ratio: float = 5.0,
    size_factors_override: np.ndarray | None = None,
    background_rank: int = 3,
) -> EnrichmentResult:
    """NB-GLM enrichment LRT across >=2 conditions with a common dispersion trend.

    Per locus, identifies the top condition (k* = argmax of size-factor-
    normalized signal) and a background condition (k_bg = the condition at
    rank `background_rank` when conditions are sorted by normalized signal,
    where rank 1 is k*, rank 2 is the second-highest, etc.) and runs a 1-df
    one-sided LRT of `mu_{k*} = mu_{k_bg}` vs `mu_{k*} > mu_{k_bg}` on that
    pair. The other K - 2 conditions enter both models as saturated nuisance
    and cancel from the likelihood ratio. The p-value is Bonferroni-
    corrected by K for the data-driven argmax. Loci where a single
    condition is *depleted* relative to the others are not called: under
    depletion, the top conditions all sit at the high level and the LRT is
    near zero.

    `background_rank` controls robustness to "competing peaks". With the
    default of 3, the LRT compares k* against the 3rd-ranked condition,
    so a single second condition that is also active does not depress the
    test statistic. Higher values tolerate more competing peaks at the cost
    of comparing against an increasingly-low background. `background_rank=2`
    compares against the second-highest condition. `background_rank` is
    capped to K when larger (so the default works at K=2 without
    special-casing).

    Parameters
    ----------
    counts
        (n_loci, n_conditions) non-negative, finite float array. Typically the
        numeric columns of `fertilizer extract`'s output.
    pseudocount
        Added to normalized counts before the log2 transform used for
        computing `effect_size`. Does NOT affect the NB likelihood.
    fit_type
        One of "parametric" (fit alpha = a/mu + b), "common" (single
        alpha = median of per-locus MoM estimates over informative loci),
        or "zero" (force Poisson; diagnostic only).
    dispersion_min_signal
        Loci with mean normalized signal below this are excluded from
        dispersion trend fitting. They are still tested at whatever
        dispersion the trend predicts for their mean.
    dispersion_override
        If not None, use this fixed alpha for every locus and skip
        fitting. Useful for sensitivity analyses.
    size_factor_warn_ratio
        Emit a FertilizerEnrichmentWarning when max(sf) / min(sf) exceeds this.
        Large spreads often indicate a violated null-majority assumption.
    size_factors_override
        If not None, a length-K array of positive size factors used in place
        of the median-of-ratios estimate. Use this when you have an external
        normalization (RPM/RPKM, spike-in factors, etc.) you trust more than
        the package's null-majority median-of-ratios. The corresponding
        EnrichmentResult.n_loci_for_size_factors is reported as 0 to signal
        that no loci were used to derive it.
    background_rank
        Rank (1 = k*, 2 = second-highest, ...) of the condition used as the
        background in the LRT pair. Default 3 (compare k* against the 3rd
        condition, tolerating one competing peak). Must be >= 2. Capped to
        K when larger.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] < 2:
        raise ValueError(
            f"counts must be (n_loci, n_conditions) with n_conditions >= 2; "
            f"got shape {counts.shape}"
        )
    n_nonfinite = int((~np.isfinite(counts)).sum())
    if n_nonfinite > 0:
        raise ValueError(
            f"counts must be finite; found {n_nonfinite} NaN or infinite "
            "value(s). An empty cell in the input TSV is read as NaN."
        )
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    if pseudocount <= 0.0:
        raise ValueError(
            f"pseudocount must be > 0 (got {pseudocount}); pc=0 produces "
            "-inf in the effect-size log2 transform whenever a condition "
            "is exactly zero"
        )
    if not isinstance(background_rank, (int, np.integer)) or background_rank < 2:
        raise ValueError(
            f"background_rank must be an integer >= 2 (got {background_rank!r}); "
            "rank 2 compares k* against the second-highest condition, 3 against "
            "the third-highest (tolerating one competing peak), and so on."
        )
    n_loci, K = counts.shape
    effective_rank = min(int(background_rank), K)

    if size_factors_override is not None:
        sf = np.asarray(size_factors_override, dtype=np.float64)
        if sf.shape != (K,):
            raise ValueError(
                f"size_factors_override must have shape ({K},); got {sf.shape}"
            )
        if not np.all(sf > 0) or not np.all(np.isfinite(sf)):
            raise ValueError("size_factors_override must be all positive and finite")
        n_loci_sf = 0
    else:
        sf, n_loci_sf = size_factors_with_n(counts)
    sf_ratio = float(sf.max() / max(sf.min(), 1e-300))
    if sf_ratio > size_factor_warn_ratio:
        warnings.warn(
            f"size factors span {sf_ratio:.1f}x (max/min); this is large, "
            "and the null-majority assumption behind median-of-ratios may "
            "be violated - real enrichment signal will be absorbed into "
            "the size-factor estimate",
            FertilizerEnrichmentWarning, stacklevel=2,
        )

    normalized = counts / sf
    alpha_mom, mu, _ = _mom_dispersion(normalized)

    if dispersion_override is not None:
        alpha = np.full(n_loci, float(dispersion_override))
        trend = (0.0, float(dispersion_override))
        fit_label = "override"
    elif fit_type == "zero":
        warnings.warn(
            "fit_type='zero' forces Poisson (alpha=0). This is strictly "
            "anti-conservative if real overdispersion exists — p-values "
            "will be too small and q-values will under-estimate the FDR. "
            "Intended for diagnostics only.",
            FertilizerEnrichmentWarning, stacklevel=2,
        )
        alpha = np.zeros(n_loci)
        trend = (0.0, 0.0)
        fit_label = "zero"
    elif fit_type == "common":
        informative = (mu >= dispersion_min_signal) & np.isfinite(alpha_mom)
        if informative.sum() >= 10:
            common = max(float(np.median(alpha_mom[informative])), 0.0)
        else:
            warnings.warn(
                f"fewer than 10 loci passed --min-signal={dispersion_min_signal} "
                f"(got {int(informative.sum())}); cannot estimate dispersion — "
                "falling back to Poisson (alpha=0). This is strictly "
                "anti-conservative if real overdispersion exists. Lower "
                "--min-signal, supply more loci, or pass --dispersion explicitly.",
                FertilizerEnrichmentWarning, stacklevel=2,
            )
            common = 0.0
        alpha = np.full(n_loci, common)
        trend = (0.0, common)
        fit_label = "common"
    elif fit_type == "parametric":
        # alpha_mom already carries the per-locus c_K correction, so the
        # fitted (a, b) are on the right scale — no post-hoc rescaling.
        fitted = _fit_parametric_trend(mu, alpha_mom, dispersion_min_signal)
        if fitted is None:
            informative = (mu >= dispersion_min_signal) & np.isfinite(alpha_mom)
            if informative.sum() >= 10:
                common = max(float(np.median(alpha_mom[informative])), 0.0)
                warnings.warn(
                    "parametric dispersion fit failed; falling back to a "
                    f"single common alpha = {common:.4g}.",
                    FertilizerEnrichmentWarning, stacklevel=2,
                )
            else:
                warnings.warn(
                    "parametric dispersion fit failed and fewer than 10 loci "
                    f"passed --min-signal={dispersion_min_signal}; falling "
                    "back to Poisson (alpha=0). This is strictly "
                    "anti-conservative if real overdispersion exists.",
                    FertilizerEnrichmentWarning, stacklevel=2,
                )
                common = 0.0
            alpha = np.full(n_loci, common)
            trend = (0.0, common)
            fit_label = "common-fallback"
        else:
            trend = (float(fitted[0]), float(fitted[1]))
            alpha = _apply_trend(mu, trend)
            # Clip per-locus alpha to the trend's evaluated range on the
            # informative-loci support. Loci with mu well outside that range
            # get extrapolated values that can be wildly off (especially for
            # the a/mu term as mu -> 0); clipping bounds the damage without
            # changing well-supported estimates.
            informative_mu = mu[(mu >= dispersion_min_signal) & (mu > 0)]
            if informative_mu.size > 0:
                support_alpha = _apply_trend(informative_mu, trend)
                lo, hi = float(support_alpha.min()), float(support_alpha.max())
                alpha = np.clip(alpha, lo, hi)
            fit_label = "parametric"
    else:
        raise ValueError(f"unknown fit_type: {fit_type!r}")

    # Identify k* (argmax) and the background condition (rank `effective_rank`
    # in the sort by size-factor-normalized signal). The other K - 2 conditions
    # enter null and alt as saturated nuisance parameters that cancel from
    # the LRT. Sorting ascending with argsort, the j-th largest sits at
    # position -j, so rank-1 = order[:, -1] = k* and rank-r = order[:, -r].
    order = np.argsort(normalized, axis=1)
    row_arange = np.arange(n_loci)
    enriched_idx = order[:, -1]
    bg_idx = order[:, -effective_rank]

    x_top = counts[row_arange, enriched_idx]
    x_bg = counts[row_arange, bg_idx]
    s_top = sf[enriched_idx]
    s_bg = sf[bg_idx]
    # Flag loci where the LRT pair contains a zero. These produce very small
    # p-values driven by mu_alt ~ 0 (clipped to 1e-20) rather than by data,
    # and tend to dominate the top of any sparse-data output as spurious
    # hits (e.g. a region of poor mappability in some tracks).
    lrt_zero_dominated = (x_top == 0) | (x_bg == 0)

    # Null fit on the pair: shared mu via intercept-only NB MLE.
    pair_counts = np.column_stack([x_top, x_bg])
    pair_sf = np.column_stack([s_top, s_bg])
    mu_top, mle_converged = _intercept_mle(pair_counts, pair_sf, alpha)
    mu_null_top = np.maximum(mu_top * s_top, 1e-20)
    mu_null_bg = np.maximum(mu_top * s_bg, 1e-20)
    lrt_convergence_failed = ~mle_converged

    # Alternative fit on the pair: saturated. Constraint mu_{k*} > mu_{k_bg}
    # is satisfied by construction (k* is the argmax of X/s, k_bg is lower).
    mu_alt_top = np.maximum(x_top, 1e-20)
    mu_alt_bg = np.maximum(x_bg, 1e-20)

    ll_alt = (
        _nb_logpmf(x_top, mu_alt_top, alpha)
        + _nb_logpmf(x_bg, mu_alt_bg, alpha)
    )
    ll_null = (
        _nb_logpmf(x_top, mu_null_top, alpha)
        + _nb_logpmf(x_bg, mu_null_bg, alpha)
    )
    lrt_stat = np.clip(2.0 * (ll_alt - ll_null), 0.0, None)

    # Half chi^2(1) tail for the one-sided LRT, then Bonferroni x K to
    # correct for selecting k* as the empirical argmax over K conditions.
    # The K-2 nuisance conditions contribute no df and need no correction.
    p_one_sided = np.where(lrt_stat > 0, 0.5 * chi2.sf(lrt_stat, df=1), 1.0)
    p_value = np.minimum(K * p_one_sided, 1.0)
    # For loci where the null-fit NB MLE didn't converge, the LRT is
    # unreliable; surface a p-value of 1.0 so users can't accidentally
    # call these as hits.
    p_value = np.where(lrt_convergence_failed, 1.0, p_value)
    q_value = bh_qvalues(p_value)

    # Effect size: log2((X_{k*}/s_{k*}) + pc) - log2(mean_{j!=k*}(X_j/s_j) + pc).
    # This is a user-facing summary of "how high is the enriched condition vs
    # typical other conditions", not the quantity the LRT is computed from.
    col_idx = np.broadcast_to(np.arange(K), (n_loci, K))
    rest_mask = col_idx != enriched_idx[:, None]
    rest_normalized = normalized[rest_mask].reshape(n_loci, K - 1)
    mean_rest = rest_normalized.mean(axis=1)
    enriched_normalized = normalized[row_arange, enriched_idx]
    effect_size = (
        np.log2(enriched_normalized + pseudocount)
        - np.log2(mean_rest + pseudocount)
    )
    # Flag loci where any normalized signal is below the pseudocount; for
    # those loci the log2 ratio is dominated by `pc` rather than by data,
    # and the effect size should be interpreted as a lower bound only.
    effect_size_pc_dominated = (normalized < pseudocount).any(axis=1)

    return EnrichmentResult(
        size_factors=sf,
        n_loci_for_size_factors=n_loci_sf,
        dispersion_fit=fit_label,
        dispersion_trend=trend,
        per_locus_dispersion=alpha,
        effect_size=effect_size,
        lrt_stat=lrt_stat,
        p_value=p_value,
        q_value=q_value,
        enriched_condition_idx=enriched_idx,
        effect_size_pc_dominated=effect_size_pc_dominated,
        lrt_zero_dominated=lrt_zero_dominated,
        lrt_convergence_failed=lrt_convergence_failed,
        background_rank=effective_rank,
    )


_ENRICH_EPILOG = """\
Example:
  fertilizer enrich -i signals.tsv -c A B C -o enrichment.tsv

  # keep all loci (for QC / volcano plots):
  fertilizer enrich -i signals.tsv -c A B C -o all.tsv --q-threshold 1.0

  # external normalization (spike-in / pre-normalized tracks):
  fertilizer enrich -i signals.tsv -c A B C -o out.tsv --size-factors 1 1 1

See the README for full docs:
https://github.com/jmschrei/fertilizer#fertilizer-enrich--enrichment-analysis
"""


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the `fertilizer enrich` subcommand."""
    parser = subparsers.add_parser(
        "enrich",
        help="Find regions where one condition is enriched over the others.",
        epilog=_ENRICH_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Enrichment NB-GLM likelihood-ratio test across conditions, "
            "adapted from DESeq2 for the one-replicate-per-condition setting. "
            "Reads a TSV (typically the output of `fertilizer extract`), tests "
            "each locus for one condition being significantly enriched "
            "relative to the others, and writes a filtered TSV with effect "
            "size (log2 enriched-vs-rest), p-value, q-value, and the name of "
            "the enriched condition. Loci where a single condition is "
            "depleted relative to the others are not called. See the README "
            "for the specific ways this differs from DESeq2."
        ),
    )
    parser.add_argument("-i", "--input", required=True, metavar="TSV",
                        help="Input TSV.")
    parser.add_argument("-c", "--conditions", nargs="+", required=True, metavar="COL",
                        help="Two or more column names to compare.")
    parser.add_argument("-o", "--output", required=True, metavar="TSV",
                        help="Output TSV (filtered to loci passing the threshold).")
    parser.add_argument("--q-threshold", type=float, default=0.05,
                        help="Keep loci with q-value <= this (default 0.05; "
                             "use 1.0 to keep all rows).")
    parser.add_argument("--p-threshold", type=float, default=None,
                        help="Additionally require p-value <= this (default: off).")
    parser.add_argument("--pseudocount", type=float, default=0.5,
                        help="Pseudocount for the effect-size log2 transform "
                             "(default 0.5). Does not affect the LRT.")
    parser.add_argument("--fit-type", choices=["common", "parametric", "zero"],
                        default="common",
                        help="Dispersion model: single common alpha (default, "
                             "robust median of MoM estimates), parametric "
                             "alpha=a/mu+b trend, or Poisson.")
    parser.add_argument("--min-signal", type=float, default=5.0,
                        help="Minimum mean normalized signal for loci included "
                             "in dispersion trend fitting (default 5.0).")
    parser.add_argument("--dispersion", type=float, default=None,
                        help="Override the fitted dispersion with a fixed alpha "
                             "applied to every locus. Bypasses --fit-type and "
                             "--min-signal entirely; reported dispersion_fit "
                             "becomes 'override'. Useful for sensitivity "
                             "analyses — e.g., re-run with 0.05 and 0.10 to "
                             "see how much downstream calls depend on the "
                             "dispersion estimate.")
    parser.add_argument("--size-factors", nargs="+", type=float, default=None,
                        metavar="SF",
                        help="Externally-provided size factors, one positive "
                             "value per --conditions entry in the same order. "
                             "Bypasses median-of-ratios. Use when you have a "
                             "normalization you trust more (RPM/RPKM, "
                             "spike-in). Pass `1 1 1 ...` to disable "
                             "normalization entirely.")
    parser.add_argument(
        "--allow-non-sum", action="store_true",
        help="Bypass the check that the input was produced by `fertilizer "
             "extract --stat sum`. The NB-GLM likelihood assumes count-like "
             "data; `mean`/`max`/`min`/`std`/`coverage` are not counts and "
             "the reported p-values may be miscalibrated. Use only when you "
             "have empirically verified calibration on your data.",
    )
    parser.add_argument("--background-rank", type=int, default=3, metavar="K",
                        help="Rank (1 = enriched condition, 2 = "
                             "second-highest, ...) of the condition "
                             "compared against the enriched one in the LRT "
                             "pair. Default 3 (tolerates one competing "
                             "peak). Larger values tolerate more competing "
                             "peaks; 2 compares directly against the "
                             "second-highest condition. Capped to the "
                             "number of conditions when larger.")
    parser.set_defaults(func=run_enrich)
    return parser


def _warn_if_regions_overlap(df: pd.DataFrame) -> None:
    """Emit a FertilizerEnrichmentWarning when input regions are not
    disjoint. BH-FDR validity assumes independent (or PRDS) tests; densely
    overlapping windows violate this and q-values become optimistic.

    Only fires when chrom/start/end columns are present and the overlap
    fraction exceeds a small noise threshold. Silent for the common case
    of an unrelated TSV (no coord columns) or a clean non-overlapping
    region set.
    """
    cols = {"chrom", "start", "end"}
    if not cols.issubset(df.columns) or len(df) < 2:
        return
    sorted_df = df[["chrom", "start", "end"]].sort_values(
        ["chrom", "start", "end"]
    ).reset_index(drop=True)
    same_chrom = sorted_df["chrom"].values[1:] == sorted_df["chrom"].values[:-1]
    overlap = sorted_df["start"].values[1:] < sorted_df["end"].values[:-1]
    overlap_frac = float((same_chrom & overlap).sum()) / max(len(sorted_df) - 1, 1)
    if overlap_frac > 0.01:
        warnings.warn(
            f"{overlap_frac:.1%} of adjacent regions overlap; BH-FDR assumes "
            "independent (or positively dependent) tests, and densely "
            "overlapping windows violate this — reported q-values will be "
            "optimistic. Thin to non-overlapping regions if possible.",
            FertilizerEnrichmentWarning, stacklevel=2,
        )


def _read_extract_stat(path: str) -> str | None:
    """Return the `stat=` value from a `fertilizer extract` metadata header,
    or None if the input lacks one (e.g. user-supplied TSV). Transparently
    handles `.gz` inputs."""
    opener = gzip.open if str(path).endswith(".gz") else open
    try:
        with opener(path, "rt") as fh:
            first = fh.readline()
    except OSError:
        return None
    if not first.startswith("#"):
        return None
    # Expected form: "# fertilizer-extract stat=<value>\n"
    parts = first.lstrip("#").strip().split()
    for tok in parts:
        if tok.startswith("stat="):
            return tok.split("=", 1)[1]
    return None


def run_enrich(args: argparse.Namespace) -> int:
    if len(args.conditions) < 2:
        raise ValueError("need at least 2 conditions to run enrichment analysis")
    if not 0.0 <= args.q_threshold <= 1.0:
        raise ValueError(f"--q-threshold must be in [0, 1], got {args.q_threshold}")
    if args.p_threshold is not None and not 0.0 <= args.p_threshold <= 1.0:
        raise ValueError(f"--p-threshold must be in [0, 1], got {args.p_threshold}")
    if args.pseudocount <= 0.0:
        raise ValueError(
            f"--pseudocount must be > 0 (got {args.pseudocount}); pc=0 "
            "produces -inf effect sizes on zero-valued conditions"
        )
    if args.background_rank < 2:
        raise ValueError(
            f"--background-rank must be >= 2 (got {args.background_rank}); "
            "rank 2 compares against the second-highest condition, 3 against "
            "the third-highest (tolerating one competing peak), and so on."
        )

    extract_stat = _read_extract_stat(args.input)
    if extract_stat is not None and extract_stat != "sum" and not args.allow_non_sum:
        raise ValueError(
            f"input was produced by `fertilizer extract --stat {extract_stat}`, "
            "which aggregates bigWig signal in a way that is NOT count-like; "
            "the NB-GLM likelihood used by `enrich` assumes count-like data. "
            "Re-run extract with `--stat sum`, or pass `--allow-non-sum` "
            "to bypass this check at your own risk (p-values may be miscalibrated)."
        )

    df = pd.read_csv(args.input, sep="\t", dtype={"chrom": str}, comment="#")
    _warn_if_regions_overlap(df)
    missing = [c for c in args.conditions if c not in df.columns]
    if missing:
        raise ValueError(f"columns not found in {args.input}: {missing}")

    counts = df[list(args.conditions)].to_numpy(dtype=np.float64)
    sf_override = None
    if args.size_factors is not None:
        if len(args.size_factors) != len(args.conditions):
            raise ValueError(
                f"--size-factors has {len(args.size_factors)} values but "
                f"--conditions has {len(args.conditions)}; they must match"
            )
        sf_override = np.asarray(args.size_factors, dtype=np.float64)
    result = enrichment_analysis(
        counts,
        pseudocount=args.pseudocount,
        fit_type=args.fit_type,
        dispersion_min_signal=args.min_signal,
        dispersion_override=args.dispersion,
        size_factors_override=sf_override,
        background_rank=args.background_rank,
    )

    df["effect_size"] = result.effect_size
    df["p_value"] = result.p_value
    df["q_value"] = result.q_value
    df["enriched_condition"] = np.asarray(args.conditions)[result.enriched_condition_idx]
    # Boolean flag columns are always emitted (all-False if nothing tripped)
    # so downstream parsers can rely on a stable schema.
    df["effect_size_pc_dominated"] = result.effect_size_pc_dominated
    df["lrt_zero_dominated"] = result.lrt_zero_dominated
    df["lrt_convergence_failed"] = result.lrt_convergence_failed

    mask = df["q_value"] <= args.q_threshold
    if args.p_threshold is not None:
        mask &= df["p_value"] <= args.p_threshold
    out = df.loc[mask].reset_index(drop=True)

    # Emit diagnostics before writing so they're visible even if the write fails.
    for cond, sf_val in zip(args.conditions, result.size_factors, strict=True):
        print(f"size factor {cond}: {sf_val:.4f}", file=sys.stderr)
    if sf_override is not None:
        print("size factors: supplied via --size-factors (median-of-ratios bypassed)",
              file=sys.stderr)
    else:
        print(
            f"size factors estimated from {result.n_loci_for_size_factors} / "
            f"{len(df)} loci (positive signal in every condition)",
            file=sys.stderr,
        )
    a, b = result.dispersion_trend
    print(
        f"dispersion fit: {result.dispersion_fit} "
        f"(alpha(mu) = {a:.4g}/mu + {b:.4g})",
        file=sys.stderr,
    )
    K = len(args.conditions)
    rank_note = ""
    if args.background_rank > K:
        rank_note = (
            f" (requested {args.background_rank}, capped to K)"
        )
    print(
        f"background rank: {result.background_rank}{rank_note} "
        f"(k* compared against the rank-{result.background_rank} condition; "
        f"higher = more robust to competing peaks)",
        file=sys.stderr,
    )
    expected_t1 = _expected_t1_at_05(K, result.background_rank)
    print(
        f"conservativeness: 1-df chi-bar-squared null with Bonferroni x K={K} for "
        f"the data-driven argmax. At K={K}, rank={result.background_rank}, "
        f"empirical T1@alpha=0.05 ~= {expected_t1:.3f}.",
        file=sys.stderr,
    )
    # At K=3, background_rank=3, the empirical Type-I rate at nominal alpha=0.05
    # is approximately 2x nominal (rank-3 is the lowest of three conditions and
    # the order-statistic gap is at its widest there). Filtering at q <= 0.05
    # gives users roughly the FDR they'd get from q <= 0.10 — worth a warning,
    # not just a quiet stderr line.
    if K == 3 and result.background_rank == 3 and expected_t1 > 0.05:
        warnings.warn(
            f"at K=3 with default --background-rank=3, the empirical Type-I "
            f"rate at nominal alpha=0.05 is ~{expected_t1:.3f}, roughly 2x "
            "nominal. The q-values you'd typically filter at (e.g. 0.05) "
            "correspond to a higher effective FDR. To get more conservative "
            "calibration at K=3, either pass `--background-rank 2` (compares "
            "against the runner-up; uniformly conservative across K) or "
            "tighten `--q-threshold` (e.g. to 0.025).",
            FertilizerEnrichmentWarning, stacklevel=2,
        )
    print(
        f"kept {len(out)} / {len(df)} loci "
        f"(q <= {args.q_threshold}"
        + (f", p <= {args.p_threshold}" if args.p_threshold is not None else "")
        + ")",
        file=sys.stderr,
    )

    # pandas infers .gz compression from the filename suffix.
    out.to_csv(args.output, sep="\t", index=False)
    return 0


def _expected_t1_at_05(K: int, rank: int) -> float:
    """Rough effective Type-I rate at nominal alpha=0.05 under the Poisson
    null, derived from simulation. Used only for the conservativeness
    diagnostic in CLI stderr output. Two tables: one for rank=2 (uniformly
    conservative); one for rank>=3 (approximately nominal at K=3 — where
    the rank-3 condition is the lowest of three and the order-statistic
    gap is at its widest — and increasingly conservative as K grows). For
    rank > 3 the rank=3 table is used as a (slightly pessimistic)
    approximation; the qualitative story is the same."""
    pair_table = {2: 0.05, 3: 0.008, 4: 0.003, 5: 0.001, 6: 0.0005, 7: 0.0003, 8: 0.0002}
    default = {2: 0.05, 3: 0.08, 4: 0.015, 5: 0.003, 6: 0.002, 7: 0.001, 8: 0.0005}
    table = pair_table if rank == 2 else default
    if K in table:
        return table[K]
    return table[8] if K > 8 else table[2]
