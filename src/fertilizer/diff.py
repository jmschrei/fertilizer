"""NB-GLM-based differential activity analysis across >=2 conditions.

This module implements a DESeq2-inspired negative-binomial GLM likelihood-
ratio test for the single-replicate-per-condition setting this package
targets. The algorithm follows the shape of DESeq2 (Love, Huber, Anders,
2014) — size factors, dispersion trend across loci, NB-GLM LRT — but is a
deliberate simplification. See "Differences from DESeq2" below for the
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
   loci (robust to the minority of truly-differential loci pulling their
   own MoM estimate up). A parametric trend `alpha(mu) = a / mu + b`
   (DESeq2's functional form) is available via `fit_type="parametric"`;
   its fit robustly trims upper-tail MoM estimates before weighted least
   squares. The final per-locus alpha is applied directly, not shrunk via
   empirical Bayes as in DESeq2 — we replace.
4. Null-model NB GLM fit per locus by vectorized Newton's method on the
   intercept-only score equation
       sum_j (X_ij - mu_0 * s_j) / (1 + alpha_i * mu_0 * s_j) = 0
   starting from the Poisson closed-form MLE.
5. The full model is saturated (mu_ij = X_ij, one parameter per condition).
   LRT statistic T_i = 2 * (logL_full - logL_null) ~ chi-squared with
   df = K - 1 under the null.
6. Benjamini-Hochberg q-values across loci.

The effect size reported is `max_j |log2((X_ij/s_j) + pc) - log2(mean_i + pc)|`,
where `mean_i` is the per-locus grand mean across all conditions on the
size-factor-normalized scale. This is the ANOVA-like analogue of DESeq2's
`log2FoldChange` for two-group comparisons.

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
  via apeglm/ashr; the effect size we report is the raw maximum.
- **Arbitrary designs.** DESeq2's LRT supports `full` vs `reduced` formulas
  of arbitrary design matrices. We hard-code the all-conditions-equal null
  vs per-condition-mean full.
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


class FertilizerDiffWarning(UserWarning):
    """Quality warnings emitted from the diff pipeline."""


@dataclass
class DiffResult:
    size_factors: np.ndarray                  # (n_conditions,)
    dispersion_fit: str                       # label for how alpha was derived
    dispersion_trend: tuple[float, float]     # (a, b) s.t. alpha(mu) = a/mu + b
    per_locus_dispersion: np.ndarray          # (n_loci,) final alpha used
    effect_size: np.ndarray                   # (n_loci,) max |log2FC vs grand mean|
    lrt_stat: np.ndarray                      # (n_loci,) LRT test statistic
    p_value: np.ndarray                       # (n_loci,)
    q_value: np.ndarray                       # (n_loci,)
    max_condition_idx: np.ndarray             # (n_loci,)


def size_factors(counts: np.ndarray) -> np.ndarray:
    """DESeq2 median-of-ratios size factors.

    `counts` is an (n_loci, n_samples) array. Only loci with positive
    signal in every sample contribute. Raises ValueError if fewer than 2
    such loci exist.
    """
    counts = np.asarray(counts, dtype=np.float64)
    positive = (counts > 0).all(axis=1)
    if positive.sum() < 2:
        raise ValueError(
            "at least 2 loci with positive signal in every sample are "
            "required to compute size factors"
        )
    log_counts = np.log(counts[positive])
    log_geom_mean = log_counts.mean(axis=1, keepdims=True)
    return np.exp(np.median(log_counts - log_geom_mean, axis=0))


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

    MoM alpha for truly-differential loci is biased upward (mean heterogeneity
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
    # where differential loci land when the null majority is near alpha = 0.
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
    except Exception:
        return None
    return float(result.x[0]), float(result.x[1])


def _apply_trend(mu: np.ndarray, trend: tuple[float, float]) -> np.ndarray:
    """Evaluate alpha = a/mu + b at each locus mean."""
    a, b = trend
    safe_mu = np.where(mu > 0, mu, 1.0)
    alpha = np.where(mu > 0, a / safe_mu + b, b)
    return np.clip(alpha, 0.0, None)


def _null_mle(
    counts: np.ndarray,
    sf: np.ndarray,
    alpha: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Intercept-only NB GLM MLE per locus.

    Solves sum_j (X_ij - mu_0 s_j) / (1 + alpha_i mu_0 s_j) = 0 for
    mu_0 per locus, via vectorized Newton's method starting from the
    Poisson closed-form MLE. The score function is monotone decreasing,
    so Newton's method converges quickly with no stepsize control.
    Returns mu_0 of shape (n_loci,).
    """
    counts = np.asarray(counts, dtype=np.float64)
    sf = np.asarray(sf, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 0:
        alpha = np.full(counts.shape[0], float(alpha))

    mu0 = counts.sum(axis=1) / sf.sum()
    mu0 = np.maximum(mu0, 1e-20)

    needs_iter = alpha > _POISSON_CUTOFF
    if not needs_iter.any():
        return mu0

    alpha_c = alpha[:, None]
    s_row = sf[None, :]
    for _ in range(max_iter):
        mu_ij = mu0[:, None] * s_row
        denom = 1.0 + alpha_c * mu_ij
        f = ((counts - mu_ij) / denom).sum(axis=1)
        # score derivative is always negative; clamp to a strict upper bound
        # of -1e-20 so division never blows up
        f_prime = -((s_row * (1.0 + alpha_c * counts)) / denom ** 2).sum(axis=1)
        f_prime = np.minimum(f_prime, -1e-20)
        step = np.where(needs_iter, f / f_prime, 0.0)
        mu0_new = np.maximum(mu0 - step, 1e-20)
        if np.max(np.abs(mu0_new - mu0)) < tol:
            mu0 = mu0_new
            break
        mu0 = mu0_new

    return mu0


def differential_analysis(
    counts: np.ndarray,
    pseudocount: float = 0.5,
    fit_type: str = "common",
    dispersion_min_signal: float = 5.0,
    dispersion_override: float | None = None,
    size_factor_warn_ratio: float = 5.0,
) -> DiffResult:
    """NB-GLM LRT across >=2 conditions with a common dispersion trend.

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
        Emit a FertilizerDiffWarning when max(sf) / min(sf) exceeds this.
        Large spreads often indicate a violated null-majority assumption.
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

    if K == 2:
        warnings.warn(
            "K=2: only one observation per condition means per-locus "
            "variance is unobserved (1 df for the sample variance), and "
            "the LRT's χ²(1) reference distribution is poorly calibrated "
            "at this sample size. The NB likelihood adds little over a "
            "raw ratio test here. Prefer ranking loci by effect_size "
            "(|log2FC vs grand mean|) over trusting q-values; add more "
            "conditions if calibrated inference matters.",
            FertilizerDiffWarning, stacklevel=2,
        )

    sf = size_factors(counts)
    sf_ratio = float(sf.max() / max(sf.min(), 1e-300))
    if sf_ratio > size_factor_warn_ratio:
        warnings.warn(
            f"size factors span {sf_ratio:.1f}x (max/min); this is large, "
            "and the null-majority assumption behind median-of-ratios may "
            "be violated - real differential signal will be absorbed into "
            "the size-factor estimate",
            FertilizerDiffWarning, stacklevel=2,
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
            FertilizerDiffWarning, stacklevel=2,
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
                FertilizerDiffWarning, stacklevel=2,
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
                    FertilizerDiffWarning, stacklevel=2,
                )
            else:
                warnings.warn(
                    "parametric dispersion fit failed and fewer than 10 loci "
                    f"passed --min-signal={dispersion_min_signal}; falling "
                    "back to Poisson (alpha=0). This is strictly "
                    "anti-conservative if real overdispersion exists.",
                    FertilizerDiffWarning, stacklevel=2,
                )
                common = 0.0
            alpha = np.full(n_loci, common)
            trend = (0.0, common)
            fit_label = "common-fallback"
        else:
            trend = (float(fitted[0]), float(fitted[1]))
            alpha = _apply_trend(mu, trend)
            fit_label = "parametric"
    else:
        raise ValueError(f"unknown fit_type: {fit_type!r}")

    mu0 = _null_mle(counts, sf, alpha)
    mu_null_ij = mu0[:, None] * sf[None, :]

    # Full-model fit is saturated; floor at a small positive to keep NB
    # log-pmf finite when an observation is exactly zero. For y=0, the
    # log-pmf contribution is `r * log(r/(mu+r))`, which is 0 at mu=0 --
    # flooring does not affect this case meaningfully.
    mu_full_ij = np.maximum(counts, 1e-20)

    alpha_b = alpha[:, None]
    ll_full = _nb_logpmf(counts, mu_full_ij, alpha_b).sum(axis=1)
    ll_null = _nb_logpmf(counts, mu_null_ij, alpha_b).sum(axis=1)
    lrt_stat = np.clip(2.0 * (ll_full - ll_null), 0.0, None)
    p_value = chi2.sf(lrt_stat, df=K - 1)
    q_value = bh_qvalues(p_value)

    log2_norm = np.log2(normalized + pseudocount)
    log2_grand = np.log2(mu + pseudocount)
    effect_size = np.max(np.abs(log2_norm - log2_grand[:, None]), axis=1)
    max_condition_idx = normalized.argmax(axis=1)

    return DiffResult(
        size_factors=sf,
        dispersion_fit=fit_label,
        dispersion_trend=trend,
        per_locus_dispersion=alpha,
        effect_size=effect_size,
        lrt_stat=lrt_stat,
        p_value=p_value,
        q_value=q_value,
        max_condition_idx=max_condition_idx,
    )


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the `fertilizer diff` subcommand."""
    parser = subparsers.add_parser(
        "diff",
        help="Find differentially active regions across two or more conditions.",
        description=(
            "NB-GLM likelihood-ratio test across conditions, adapted from "
            "DESeq2 for the one-replicate-per-condition setting. Reads a "
            "TSV (typically the output of `fertilizer extract`), tests each "
            "locus, and writes a filtered TSV with effect size, p-value, "
            "q-value, and the name of the condition carrying the highest "
            "signal. See the README for the specific ways this differs "
            "from DESeq2."
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
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    if len(args.conditions) < 2:
        raise ValueError("need at least 2 conditions to run differential analysis")
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
    result = differential_analysis(
        counts,
        pseudocount=args.pseudocount,
        fit_type=args.fit_type,
        dispersion_min_signal=args.min_signal,
        dispersion_override=args.dispersion,
    )

    out = df.copy()
    out["effect_size"] = result.effect_size
    out["p_value"] = result.p_value
    out["q_value"] = result.q_value
    out["max_condition"] = np.asarray(args.conditions)[result.max_condition_idx]

    mask = out["q_value"] <= args.q_threshold
    if args.p_threshold is not None:
        mask &= out["p_value"] <= args.p_threshold
    out = out.loc[mask]

    # Emit diagnostics before writing so they're visible even if the write fails.
    for cond, sf_val in zip(args.conditions, result.size_factors):
        print(f"size factor {cond}: {sf_val:.4f}", file=sys.stderr)
    a, b = result.dispersion_trend
    print(
        f"dispersion fit: {result.dispersion_fit} "
        f"(alpha(mu) = {a:.4g}/mu + {b:.4g})",
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
