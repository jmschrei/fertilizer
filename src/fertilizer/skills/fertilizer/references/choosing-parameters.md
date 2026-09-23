# Choosing `--background-rank`, K and dispersion settings

Numbers below are from seeded simulations with `enrichment_analysis`: NB null,
μ = 50, α = 0.05, 20,000 regions (calibration) or 5,000 (power). They describe
the test's behavior, not any real dataset.

## Background rank

The rank r does two things. It sets which regions *can* be called: the top
condition is tested against the r-th, so up to r − 2 others may be as high as
the top. And it sets power: the further r is below K, the more conservative
the test. The runner-up margin filter (`references/recipes.md` §3) separates
the two: choose r for power, then enforce the specificity the user asked for.

**Highest rank measured at or under nominal:** K − 1 for K ≤ 8, K − 2 at K = 12
and K = 20. r = K is always anti-conservative, and nothing warns at K ≥ 4.

Ask the user: **if a region is high in two conditions, is it a hit?** Then:

| Question | Rank | Then |
|---|---|---|
| unique to one condition | highest safe rank | margin, top vs runner-up |
| shared by a specific pair | highest safe rank | runner-up in the pair, margin second vs third |
| top of a group of at most m, no margin | m + 1 | — |

At K = 4 the default 3 is already the highest safe rank. At K = 3 see §K = 3.

Fraction of null regions with `p_value ≤ 0.05` (the reported, × K, p-value):

| K | r = 2 | r = 3 | r = 4 | r = 5 | r = 6 | r = 7 | r = 8 |
|---|---|---|---|---|---|---|---|
| 2 | 0.053 | (capped to 2) | | | | | |
| 3 | 0.006 | **0.085** | | | | | |
| 4 | 0.001 | 0.012 | **0.114** | | | | |
| 5 | 0.000 | 0.003 | 0.020 | **0.146** | | | |
| 6 | 0.000 | 0.000 | 0.004 | 0.027 | **0.167** | | |
| 8 | 0.000 | 0.000 | 0.000 | 0.002 | 0.009 | 0.040 | **0.201** |

| K | r = 3 | r = K/2 | r = K − 2 | r = K − 1 | r = K |
|---|---|---|---|---|---|
| 12 | 0.000 | 0.000 | 0.019 | 0.067 | 0.263 |
| 20 | 0.000 | 0.000 | 0.040 | 0.113 | 0.350 |

Bold is r = K.

Realized FDR among calls (1,000 of 20,000 regions 4× up in one condition):

| K, rank | FDR at q ≤ 0.05 | FDR at q ≤ 0.025 |
|---|---|---|
| 3, 3 (default) | 0.056 | 0.026 |
| 4, 3 | 0.001 | 0.000 |
| 4, 4 | 0.094 | 0.044 |
| 5, 4 | 0.000 | 0.000 |
| 5, 5 | 0.146 | 0.078 |

Power at K = 4 (250 regions 4× up in one condition, 250 with two up, 4,500
null; calls at q ≤ 0.05):

| r | unique called | shared called | null calls |
|---|---|---|---|
| 2 | 46 | 0 | 0 |
| 3 | 176 | 175 | 0 |
| 4 (= K) | 229 | 241 | 45 |

Power at larger K (250 unique, 250 with three up, 4,500 null; one dataset per
K), before and after a runner-up margin of ≥ 2× (`references/recipes.md` §3):

| K | r | unique / shared / null called | after margin |
|---|---|---|---|
| 8 | 3 | 0 / 0 / 0 | 0 / 0 / 0 |
| 8 | 6 | 188 / 230 / 0 | 183 / 1 / 0 |
| 8 | 7 (= K − 1) | 219 / 244 / 1 | 204 / 1 / 0 |
| 12 | 3 | 0 / 0 / 0 | 0 / 0 / 0 |
| 12 | 10 (= K − 2) | 211 / 246 / 2 | 188 / 1 / 0 |
| 20 | 3 | 0 / 0 / 0 | 0 / 0 / 0 |
| 20 | 18 (= K − 2) | 224 / 250 / 1 | 191 / 4 / 0 |

At this overdispersion, 2× effects were almost never called at any K or rank;
expect calls to be strong effects.

## K = 3

At K = 3 the default rank 3 *is* rank K, so every run prints

```
FertilizerEnrichmentWarning: at K=3 with default --background-rank=3, the empirical Type-I rate at nominal alpha=0.05 is ~0.080, about 1.6x nominal.
```

(0.080 comes from a built-in table; the simulation measured 0.085.) Realized
FDR at q ≤ 0.05 was 0.056, so the output is usable. Choose one and say which:

- shared peaks are acceptable → rank 3 with `--q-threshold 0.025` (realized
  FDR 0.026);
- unique peaks only → the same, then the runner-up margin. `--background-rank 2`
  also answers it but is conservative (null rate 0.006): with 1,000 regions 4×
  up in one condition it called 467, where rank 3 called 810 before the margin.

A single all-zero track is the background at rank K, so at K = 3 it makes every
region it zeroes a significant call (2,500 of 2,500 in simulation). Check each
track (`references/extract.md` §Coordinates and locus problems).

## K = 2

Rank caps to 2 and the test is ~nominal (0.053). This is the pairwise
"A higher than B" test; `enriched_condition` says which direction won.

## Many conditions

Pseudobulk bigWigs from single-cell clusters are the usual case. At K ≥ 8 the
default rank 3 has almost no power (table above); use the highest safe rank
plus the margin filter. The stderr `conservativeness` line is wrong at these
ranks (`references/enrich.md` §Stderr); ignore it. Also:

1. Merge biologically equivalent conditions (sum their columns) to lower K.
2. Small clusters give small region sums; see `references/inputs.md` §Which
   bigWigs for when that stops the test working.

## Null majority and size factors

Median-of-ratios and pooled dispersion both assume most regions are null. With
K = 4 and one condition 4× up in a fraction of regions:

| Fraction enriched | size factor of that condition | true calls |
|---|---|---|
| 5% | 1.02 | 169 / 250 |
| 20% | 1.08 | 403 / 1000 |
| 50% | 1.65 | 0 / 2500 |
| 90% | 2.78 | 0 / 4500 |

`enrich` warns `size factors span Nx (max/min)` only above 5×, so the 50% case
passes silently. Fixes: a background region set; for a global shift in the
mark (an EZH2 knockdown and H3K27me3), spike-in `--size-factors`
(`references/recipes.md` §8).

## Dispersion settings

| Setting | When |
|---|---|
| `--fit-type common` (default) | always, unless below |
| `--fit-type parametric` | dispersion visibly trends with mean and many regions are above `--min-signal` |
| `--dispersion A` | sensitivity analysis (`references/recipes.md` §6), or an external α estimate |
| `--fit-type zero` | never for results. Poisson ignores overdispersion; anti-conservative |
| `--min-signal` | lower it only when the fit falls back to Poisson and region sums are genuinely small |

A larger α means fewer calls (demo: 11 at the fitted α = 0.056, 9 at
`--dispersion 0.1`).
