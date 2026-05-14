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
4. Per locus, identify the top two conditions by size-factor-normalized
   signal: k* = argmax_j (X_ij / s_j) and k_2 = argmax over j != k*. The
   one-sided LRT compares these two conditions only; the other K - 2
   conditions enter both models as saturated nuisance parameters and
   cancel from the likelihood ratio. This design makes the test reject
   genuine enrichment of k* above the rest of the distribution while
   ignoring depletion patterns (where the top two conditions are both at
   the high level and look essentially equal under the test).
5. Null model on the top pair: mu_{k*} = mu_{k_2} = mu_top, fit by
   intercept-only NB MLE on the two observations (X_{k*}, X_{k_2}) with
   size factors (s_{k*}, s_{k_2}). Vectorized Newton's method on the
   score equation
       sum_{j in {k*, k_2}} (X_j - mu_top s_j) / (1 + alpha mu_top s_j) = 0
   starting from the Poisson closed-form MLE.
6. Alternative on the top pair: saturated. mu_{k*} = X_{k*}/s_{k*},
   mu_{k_2} = X_{k_2}/s_{k_2}, constrained mu_{k*} > mu_{k_2} (always
   satisfied by construction since k* is the argmax).
7. LRT_i = 2 * (logL_alt - logL_null) over the top two conditions,
   chi-bar-squared({0, 1}) under H0 for fixed k*. We use the upper
   half-chi^2(1) tail (factor of 1/2), then Bonferroni-correct by K to
   account for picking k* as the empirical argmax. Final per-locus
   p-value is min(K * 0.5 * chi2_1_sf(LRT), 1).
8. Benjamini-Hochberg q-values across loci.

The effect size reported is
`log2((X_{i,k*}/s_{k*}) + pc) - log2(mean_{j!=k*}(X_{i,j}/s_j) + pc)`:
the log2 fold change of the enriched condition vs the mean of the others
on the size-factor-normalized scale. Always non-negative by construction
(k* is the argmax). Note that this is a summary for users; the test
itself is computed only from the top pair.

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
  LRT comparing the top two conditions (largest and second-largest
  normalized signal) per locus, with Bonferroni x K for picking the top
  by argmax. Other K - 2 conditions enter both models as saturated
  nuisance and cancel. Loci where one condition is *depleted* relative
  to the others (the top two conditions then both sit at the high level
  and look indistinguishable) are not called.
- **Independent filtering.** DESeq2 filters low-count loci out of
  multiple-testing correction to maximize power at a given alpha. We don't
  (but users can set `--min-signal` high and post-filter themselves).
- **Integer counts.** DESeq2 is designed for integer counts; the NB
  likelihood here uses `scipy.special.gammaln` and is numerically correct
  for any non-negative float input. This matches the common practice of
  passing fractional RSEM/salmon expected counts to DESeq2 via tximport,
  and is required here because bigWig-extracted signal is real-valued.

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
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import gammaln
from scipy.stats import chi2


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
    enriched_condition_idx: np.ndarray        # (n_loci,) argmax(X / s)
    effect_size_pc_dominated: np.ndarray      # (n_loci,) bool; true when some X_j/s_j < pc


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
) -> np.ndarray:
    """Intercept-only NB GLM MLE per locus on the top-pair sub-arrays.

    Solves sum_j (X_ij - mu_0 s_ij) / (1 + alpha_i mu_0 s_ij) = 0 for
    mu_0 per locus, via vectorized Newton's method starting from the
    Poisson closed-form MLE. The score function is monotone decreasing,
    so Newton's method converges quickly with no stepsize control.

    `counts` and `sf` are both (n_loci, 2) — one row per locus, holding
    the values for that locus's top two conditions (k*, k_2) with their
    respective size factors. `alpha` is (n_loci,) or scalar. Returns
    mu_0 of shape (n_loci,).
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
        return mu0

    alpha_c = alpha[:, None]
    converged = False
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
        max_change = float(np.max(np.abs(mu0_new - mu0)))
        mu0 = mu0_new
        if max_change < tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"intercept-only NB MLE did not converge for at least one locus "
            f"after {max_iter} iterations (max change {max_change:.3g}); "
            "p-values for those loci may be slightly miscalibrated. "
            "Consider passing --dispersion or fewer extreme counts.",
            FertilizerEnrichmentWarning, stacklevel=3,
        )
    return mu0


def enrichment_analysis(
    counts: np.ndarray,
    pseudocount: float = 0.5,
    fit_type: str = "common",
    dispersion_min_signal: float = 5.0,
    dispersion_override: float | None = None,
    size_factor_warn_ratio: float = 5.0,
    size_factors_override: np.ndarray | None = None,
) -> EnrichmentResult:
    """NB-GLM enrichment LRT across >=2 conditions with a common dispersion trend.

    Per locus, identifies the top two conditions by size-factor-normalized
    signal (k* = argmax, k_2 = second-argmax) and runs a 1-df one-sided
    LRT of `mu_{k*} = mu_{k_2}` vs `mu_{k*} > mu_{k_2}` on that pair.
    The other K - 2 conditions enter both models as saturated nuisance
    and cancel from the likelihood ratio. The p-value is Bonferroni-
    corrected by K for the data-driven argmax. Loci where a single
    condition is *depleted* relative to the others are not called: under
    depletion, k* and k_2 both sit at the high level and the LRT is
    near zero.

    Parameters
    ----------
    counts
        (n_loci, n_conditions) non-negative float array. Typically the
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
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] < 2:
        raise ValueError(
            f"counts must be (n_loci, n_conditions) with n_conditions >= 2; "
            f"got shape {counts.shape}"
        )
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    if pseudocount <= 0.0:
        raise ValueError(
            f"pseudocount must be > 0 (got {pseudocount}); pc=0 produces "
            "-inf in the effect-size log2 transform whenever a condition "
            "is exactly zero"
        )
    n_loci, K = counts.shape

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

    # Identify the top-two conditions per locus by size-factor-normalized
    # signal. The K - 2 conditions outside this pair enter null and alt as
    # saturated nuisance parameters that cancel from the LRT.
    enriched_idx = normalized.argmax(axis=1)
    normalized_masked = normalized.copy()
    row_arange = np.arange(n_loci)
    normalized_masked[row_arange, enriched_idx] = -np.inf
    second_idx = normalized_masked.argmax(axis=1)

    x_top = counts[row_arange, enriched_idx]
    x_two = counts[row_arange, second_idx]
    s_top = sf[enriched_idx]
    s_two = sf[second_idx]

    # Null fit on the top pair: shared mu_top via intercept-only NB MLE.
    top_pair_counts = np.column_stack([x_top, x_two])
    top_pair_sf = np.column_stack([s_top, s_two])
    mu_top = _intercept_mle(top_pair_counts, top_pair_sf, alpha)
    mu_null_top = np.maximum(mu_top * s_top, 1e-20)
    mu_null_two = np.maximum(mu_top * s_two, 1e-20)

    # Alternative fit on the top pair: saturated. Constraint mu_{k*} >
    # mu_{k_2} is satisfied by construction (k* is the argmax of X/s).
    mu_alt_top = np.maximum(x_top, 1e-20)
    mu_alt_two = np.maximum(x_two, 1e-20)

    ll_alt = (
        _nb_logpmf(x_top, mu_alt_top, alpha)
        + _nb_logpmf(x_two, mu_alt_two, alpha)
    )
    ll_null = (
        _nb_logpmf(x_top, mu_null_top, alpha)
        + _nb_logpmf(x_two, mu_null_two, alpha)
    )
    lrt_stat = np.clip(2.0 * (ll_alt - ll_null), 0.0, None)

    # Half chi^2(1) tail for the one-sided LRT, then Bonferroni x K to
    # correct for selecting k* as the empirical argmax over K conditions.
    # The K-2 nuisance conditions contribute no df and need no correction.
    p_one_sided = np.where(lrt_stat > 0, 0.5 * chi2.sf(lrt_stat, df=1), 1.0)
    p_value = np.minimum(K * p_one_sided, 1.0)
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
    )


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the `fertilizer enrich` subcommand."""
    parser = subparsers.add_parser(
        "enrich",
        help="Find regions where one condition is enriched over the others.",
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
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
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

    df = pd.read_csv(args.input, sep="\t", dtype={"chrom": str})
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
    )

    out = df.copy()
    out["effect_size"] = result.effect_size
    out["p_value"] = result.p_value
    out["q_value"] = result.q_value
    out["enriched_condition"] = np.asarray(args.conditions)[result.enriched_condition_idx]
    # Only emit the pc-dominated flag column when at least one locus is
    # flagged — keeps the common case (all data well above pc) clean.
    if result.effect_size_pc_dominated.any():
        out["effect_size_pc_dominated"] = result.effect_size_pc_dominated

    mask = out["q_value"] <= args.q_threshold
    if args.p_threshold is not None:
        mask &= out["p_value"] <= args.p_threshold
    out = out.loc[mask]

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
    print(
        f"conservativeness: 1-df chi-bar-squared null with Bonferroni x K={K} for "
        f"the data-driven argmax. At K={K} the test runs roughly "
        f"{_expected_t1_at_05(K):.0%} of nominal at alpha=0.05.",
        file=sys.stderr,
    )
    print(
        f"kept {len(out)} / {len(df)} loci "
        f"(q <= {args.q_threshold}"
        + (f", p <= {args.p_threshold}" if args.p_threshold is not None else "")
        + ")",
        file=sys.stderr,
    )

    out.to_csv(args.output, sep="\t", index=False)
    return 0


def _expected_t1_at_05(K: int) -> float:
    """Rough effective Type-I rate at nominal alpha=0.05 under the Poisson
    null, derived from the simulations documented in the README. Used only
    for the conservativeness diagnostic in CLI stderr output."""
    table = {2: 0.05, 3: 0.008, 4: 0.003, 5: 0.001, 6: 0.0005, 7: 0.0003, 8: 0.0002}
    if K in table:
        return table[K]
    return table[8] if K > 8 else table[2]
