# How the test works

Input: an (n regions × K conditions) matrix X of summed signal, one value per
region per condition, no replicates. The design follows DESeq2 (Love, Huber &
Anders 2014) and is simplified for that setting.

1. **Size factors.** DESeq2 median-of-ratios over regions with positive signal
   in every condition: s_j = median_i( X_ij / geometric-mean_j(X_i·) ).
   Normalized signal is X_ij / s_j. Replaced by `--size-factors` when given.
2. **Per-region dispersion, method of moments.** With μ_i and V_i the mean and
   sample variance of region i's normalized values across the K conditions,
   α_i = (V_i / c_K − μ_i) / μ_i², where c_K = median(χ²_{K−1})/(K−1)
   corrects the median's small-sample bias (0.455 at K = 2, 0.693 at K = 3,
   0.907 at K = 8).
3. **Shared dispersion.** There is no within-condition variance, so α is pooled
   *across regions*, assuming most regions are null. `common`: the median α_i
   over regions with μ_i ≥ `--min-signal`, clipped at 0. `parametric`: fits
   α(μ) = a/μ + b by weighted least squares after trimming the upper tail
   (median + 3·MAD), clipped to the fitted range. Every region uses the pooled
   value; there is no per-region shrinkage. Fewer than 10 regions above
   `--min-signal` → α = 0 (Poisson) with a warning.
4. **Pick the pair.** Sort a region's normalized values. k\* = the top
   condition, k_bg = the condition at rank `--background-rank` r (default 3;
   capped to K). The other K − 2 conditions are saturated nuisance parameters
   and cancel from the likelihood ratio.
5. **LRT on the pair.** Null: one shared mean for k\* and k_bg, fit by Newton's
   method on the NB intercept-only score equation. Alternative: each at its own
   observed value (μ_k\* > μ_bg holds by construction). T = 2(ℓ_alt − ℓ_null).
   Under the boundary-constrained null T ~ ½χ²₀ + ½χ²₁ (Self & Liang 1987), so
   p = ½·P(χ²₁ > T).
6. **Multiple testing.** p × K (Bonferroni for choosing k\* by argmax), capped at
   1; then Benjamini–Hochberg across regions.
7. **Effect size.** log2 of the top normalized value over the mean of the other
   K − 1, each plus `--pseudocount`. Descriptive only; not used by the test.

The NB log-likelihood uses `gammaln`, so non-integer sums are handled exactly.

## Consequences that users ask about

- **Enrichment only.** When one condition is depleted, the top r conditions sit
  at the same high level and T ≈ 0. Depletion is never called.
- **The rank sets the specificity question.** A call at rank r says k\* is above
  the r-th highest condition, so up to r − 2 other conditions may be as high as
  k\*. Rank 2: unique to one condition. Rank 3: top of at most two.
- **Calibration depends on K and r.** The argmax selection and the Bonferroni
  correction do not cancel exactly. Measured rates: `references/choosing-parameters.md`.
- **Normalization and dispersion come from the regions supplied.** Change the
  region set and both change.

## Differences from DESeq2

| DESeq2 | fertilizer |
|---|---|
| per-gene dispersion from replicates (Cox–Reid profile likelihood), shrunk toward a trend | method-of-moments across conditions, pooled across regions; no shrinkage, no dispersion outliers |
| arbitrary design, two-sided Wald or LRT | fixed one-sided test of top vs rank-r condition |
| Cook's distance outlier handling | none |
| independent filtering before BH | none; BH over every row |
| LFC shrinkage (apeglm/ashr) | raw log2 ratio with pseudocount |
| integer counts | any non-negative float |
