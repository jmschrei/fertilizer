# Choosing `--background-rank`, K and dispersion settings

Numbers below are from seeded simulations with `enrichment_analysis`: NB null,
μ = 50, α = 0.05, 20,000 regions (calibration) or 5,000 (power). They describe
the test's behavior, not any real dataset.

## Background rank

Ask the user: **if a region is high in two conditions, is it a hit?** Then pick:

| Wanted | Rank |
|---|---|
| active in exactly one condition | `3` plus the runner-up margin (`references/recipes.md` §3), or `2` (fewer calls) |
| top of a group of at most two (default) | `3` |
| top of a group of at most m | `m + 1` |

Rank must stay **≤ K − 1** once K ≥ 4. Rank = K tests the top against the
*lowest* condition, and the selection makes that anti-conservative.

Fraction of null regions with `p_value ≤ 0.05`:

| K | r = 2 | r = 3 | r = 4 | r = 5 | r = 6 | r = 7 | r = 8 |
|---|---|---|---|---|---|---|---|
| 2 | 0.053 | (capped to 2) | | | | | |
| 3 | 0.006 | **0.085** | | | | | |
| 4 | 0.001 | 0.012 | **0.114** | | | | |
| 5 | 0.000 | 0.003 | 0.020 | **0.146** | | | |
| 6 | 0.000 | 0.000 | 0.004 | 0.027 | **0.167** | | |
| 8 | 0.000 | 0.000 | 0.000 | 0.002 | 0.009 | 0.040 | **0.201** |

Bold is r = K for K ≥ 3. Everything else is at or under nominal, and more conservative the
further r is below K. The rates use the reported `p_value` (already × K).

Realized FDR among calls (1,000 of 20,000 regions 4× up in one condition):

| K, rank | FDR at q ≤ 0.05 | FDR at q ≤ 0.025 |
|---|---|---|
| 3, 3 (default) | 0.056 | 0.026 |
| 4, 3 | 0.001 | 0.000 |
| 4, 4 | 0.094 | 0.044 |
| 5, 4 | 0.000 | 0.000 |
| 5, 5 | 0.146 | 0.078 |

At this overdispersion (α = 0.05, μ = 50), 2× effects were almost never called
at any K or rank; expect calls to be strong effects.

Power at K = 4 (250 regions with one condition 4× up, 250 with two conditions
4× up, 4,500 null; calls at q ≤ 0.05):

| Rank | unique regions called | shared regions called | null calls |
|---|---|---|---|
| 2 | 46 / 250 | 0 / 250 | 0 |
| 3 | 176 / 250 | 175 / 250 | 0 |
| 4 (= K) | 229 / 250 | 241 / 250 | 45 |

Rank 2 loses most of the unique regions too, because the top is compared
against the maximum of the remaining K − 1. To ask "unique to one" with more
power, run rank 3 and filter on the runner-up afterwards
(`references/recipes.md` §3).

## K = 3

At K = 3 the default rank 3 *is* rank K, so every run prints

```
FertilizerEnrichmentWarning: at K=3 with default --background-rank=3, the empirical Type-I rate at nominal alpha=0.05 is ~0.080, roughly 2x nominal.
```

(stderr and the warning print 0.080 from a built-in table; the simulation
above measured 0.085). Rank = K is tolerated here only because rank 2 is the
sole alternative. Realized FDR at q ≤ 0.05 was 0.056, close to nominal, so the
output is usable. Choose one and say which:

- shared peaks are acceptable → keep rank 3; `--q-threshold 0.025` brings the
  realized FDR to 0.026;
- unique peaks only → rank 3 with `--q-threshold 0.025`, then the runner-up
  margin of `references/recipes.md` §3. `--background-rank 2` also answers it
  but is conservative (null rate 0.006): with 1,000 regions 4× up in one
  condition it called 467, where rank 3 called 810 before any margin filter.

At K ≥ 4, rank = K gets no warning at all; the only guard is choosing r ≤ K − 1.

## K = 2

Rank caps to 2 and the test is ~nominal (0.053). This is the pairwise
"A higher than B" test. It is two-sided in effect: `enriched_condition` says
which direction won.

## Many conditions

At K = 8 the default rank 3 calls almost nothing under the null (0.000); no
power was measured at K = 8, but the null rate says the threshold is far
stricter than nominal. Before raising the rank, decide whether the question
changes: rank 7 at K = 8 calls a region high in six conditions as "enriched in"
whichever of the six is top. Options:

1. Merge biologically equivalent conditions (sum their columns) to lower K.
2. Raise rank to match the specificity the user wants, keeping r ≤ K − 1.

## Null majority and size factors

Median-of-ratios and pooled dispersion both assume most regions are null. With
K = 4 and one condition 4× up in a fraction of regions:

| Fraction enriched | size factor of that condition | true calls |
|---|---|---|
| 5% | 1.02 | 169 / 250 |
| 20% | 1.08 | 403 / 1000 |
| 50% | 1.65 | 0 / 2500 |
| 90% | 2.78 | 0 / 4500 |

`enrich` warns `size factors span Nx (max/min)` above 5×. Below that the damage
is already done, as the 50% row shows. When the biology implies a global shift
(an EZH2 knockdown and H3K27me3, a transcription shutdown), or the region set
is candidates only, pass `--size-factors` from spike-ins or library sizes
(total mapped reads divided by their geometric mean), or supply a background
region set.

## Dispersion settings

| Setting | When |
|---|---|
| `--fit-type common` (default) | always, unless below |
| `--fit-type parametric` | dispersion visibly trends with mean and there are many regions above `--min-signal` |
| `--dispersion A` | sensitivity analysis (`references/recipes.md` §6), or an external α estimate |
| `--fit-type zero` | never for results. Poisson ignores overdispersion; anti-conservative |
| `--min-signal` | lower it only when the fit falls back to Poisson and region sums are genuinely small |

A larger α means fewer calls (demo: 11 at the fitted α = 0.056, 9 at
`--dispersion 0.1`).
