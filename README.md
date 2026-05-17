# fertilizer

**What this does (concretely):** `fertilizer` takes one bigWig per condition and a set of BED regions, computes a summary statistic per region per bigWig, and calls regions where one condition has significantly higher signal than the others. Output is a TSV with effect size, p-value, q-value, and the name of the enriched condition. The statistical model is a DESeq2-inspired negative-binomial GLM likelihood-ratio test adapted to the one-replicate-per-condition setting.

**Why "fertilizer".** The **Fertile Ground Hypothesis** is that genomes are full of "almost-regulatory" regions — sequences that do not do anything on their own, but can be minimally edited to achieve subtle and precise activity. Many near-motifs, for example, sit one or two substitutions away from binding a transcription factor and recruiting its downstream regulatory activity. `fertilizer` helps identify the fertile ground in a genome that is most useful for your design task by flagging regions where signal in one condition stands out from the others.

> **Status.** v0.1.0 — API is unstable until 1.0. Please report issues. Most
> of this package (including the test suite) was drafted with Claude
> assistance; the statistical methodology and calibration are exercised by
> the simulation tests in `tests/test_enrichment.py` and we are continuing
> to validate against external benchmarks.

## Installation

`fertilizer` is installable with [uv](https://docs.astral.sh/uv/). The PyPI
distribution is named `fertilizer-genomics` (the `fertilizer` name was taken),
but the importable module is still `fertilizer`.

```bash
# install from PyPI
uv pip install fertilizer-genomics

# or install directly from GitHub
uv pip install git+https://github.com/jmschrei/fertilizer.git

# or install from a local clone
git clone https://github.com/jmschrei/fertilizer.git
cd fertilizer
uv pip install -e .
```

> `pyBigWig` needs `libcurl` and `libssl` headers at install time on Linux.
> If pip fails to build it, install them first:
> `sudo apt-get install libcurl4-openssl-dev libssl-dev zlib1g-dev` (Debian/Ubuntu)
> or `brew install curl openssl` (macOS).

For a development environment with test and lint tooling:

```bash
uv venv
uv pip install -e ".[dev]"
pytest tests/
```

## Quickstart — try it on a tiny demo

```bash
python examples/make_demo_data.py
fertilizer extract -w examples/A.bw examples/B.bw examples/C.bw \
    -b examples/regions.bed -o examples/signals.tsv -s sum
fertilizer enrich -i examples/signals.tsv -c A B C \
    -o examples/enrichment.tsv --q-threshold 0.05
```

`examples/C.bw` has signal enriched at 10 randomly chosen regions; the
resulting `enrichment.tsv` should contain ~10 rows, each with
`enriched_condition == C`. See `examples/README.md` for a walkthrough.

## When NOT to use this

`fertilizer` is designed for the **one-bigWig-per-condition** setting with
many loci, most of which are not differentially enriched. It is the wrong
tool when:

- **You have replicates per condition.** Prefer DESeq2 / edgeR / csaw —
  they estimate per-locus dispersion from within-condition variance and
  give better power. `fertilizer` collapses biological + technical
  variability into a single across-condition α.
- **Most loci genuinely do change between conditions.** Median-of-ratios
  size factors and the across-loci dispersion estimator both assume that
  the null *majority* anchors normalization and variance. Pass a
  background-matched / genome-wide region set, not a candidate-only set.
- **Your experiment causes a global shift in the mark** (e.g. an EZH2
  knockdown that collapses H3K27me3 genome-wide). The null-majority
  assumption is violated by construction; use spike-in normalization
  and pass it via `--size-factors`.
- **Your bigWig aggregates are not count-like.** `enrich` refuses input
  produced by `extract --stat mean / max / min / std / coverage`
  (override at your own risk with `--allow-non-sum`). The NB-GLM
  assumes the variance-mean relationship of count data.
- **Your regions overlap densely** (sliding/tiling windows). BH controls
  FDR under independence or PRDS; overlapping windows violate this and
  q-values will be optimistic. Thin to non-overlapping regions, or use a
  method that models autocorrelation explicitly.

## Usage

`fertilizer` exposes two subcommands. Typical workflow:

```bash
fertilizer extract -w A.bw B.bw C.bw -b regions.bed -o signals.tsv -s sum
fertilizer enrich  -i signals.tsv -c A B C -o enrichment.tsv
```

> The NB-GLM in `enrich` assumes **count-like** input. Use `extract --stat sum`
> (the total signal over each region) — `mean`/`max`/`min`/`std`/`coverage`
> are not counts and `enrich` will refuse them unless `--allow-non-sum` is
> passed. `extract` writes a metadata header (`# fertilizer-extract stat=...`)
> that `enrich` reads to enforce this.

### `fertilizer extract` — signal aggregation

Compute a per-region summary statistic (mean by default; `-s` chooses among `mean`/`max`/`min`/`sum`/`std`/`coverage`) for each bigWig over each region in the concatenated BED input. One row per region, one column per bigWig. Input row order is preserved.

| flag | description |
| --- | --- |
| `-w`, `--bigwigs` | one or more bigWig signal tracks |
| `-b`, `--beds` | one or more BED region files. Columns 1-3 are required (`chrom`/`start`/`end`); columns 4-6 are passed through as `name`/`score`/`strand`; any further columns are passed through as `bed_col_<i>` (BED12 and narrowPeak disagree on the meaning of columns 7+, so generic names are used to avoid mislabeling). `#` comment lines are skipped. |
| `-o`, `--output` | path to the output TSV |
| `-s`, `--stat` | per-region summary statistic: `mean` (default), `max`, `min`, `sum`, `std`, `coverage`. Maps directly to pyBigWig's `stats(type=...)`. **Use `sum` if the output will be passed to `fertilizer enrich`** — the NB-GLM assumes count-like input. `extract` writes a `# fertilizer-extract stat=...` header line so `enrich` can verify this. |
| `-n`, `--names` | optional explicit column names, one per `--bigwigs` entry. Overrides the default of using each bigWig's filename stem. Useful when two paths share a basename (e.g. `RNAseq/A.bw` and `ATACseq/A.bw`). |
| `-j`, `--n-jobs` | parallel workers (default `-1`, all cores) |

**Coordinates are 0-based half-open**, matching the standard BED/UCSC bigWig convention. A region `chr1 100 200` covers bases 100..199 inclusive (length 100). If your input is a 1-based file (UCSC table dumps, some BED-like exports), subtract 1 from `start` before running `extract`.

Zeros never mean "missing" — the output is always numeric, never `NaN`. A region whose summary statistic genuinely resolves to zero (empty bigWig, uncovered span) is reported as `0.0` silently. A region with a locus-level problem (unknown chromosome, coordinates past the end of the chromosome, zero-length interval, negative start) is also reported as `0.0` but triggers a single `FertilizerWarning` — one warning per distinct issue type per run, regardless of how many rows or bigWigs were affected.

Example `signals.tsv`:

```
chrom   start   end     A       B       C
chr1    0       500     2.0     7.0     3.1
chr1    500     1000    6.0     7.0     5.8
chr1    1500    2000    4.0     7.0     4.4
chr2    0       100     0.0     0.0     0.0
```

Column names come from each bigWig's filename stem (override with `-n/--names`), so passing two bigWigs with the same basename (even from different directories) without `--names` is rejected up front. `extract` also emits a `FertilizerWarning` if more than 95% of the output cells are exactly zero — almost always a wrong path or a chromosome-naming mismatch (e.g. `chr1` in the BED but `1` in the bigWig).

### `fertilizer enrich` — enrichment analysis

Identify loci where one condition is **enriched** relative to the others. Takes a TSV whose columns include (at minimum) the non-negative numeric columns named via `-c` — the output of `fertilizer extract`, which also carries `chrom`/`start`/`end`, is the canonical input and is passed through verbatim — names the columns to compare, and writes a **filtered** TSV containing only loci that pass the significance threshold, with effect size, p-value, q-value, and the name of the enriched condition. Loci where one condition is *depleted* relative to the others are **not** called. Any columns present in the input beyond the tested conditions are preserved in the output.

The test is a one-sided negative-binomial GLM likelihood-ratio test per locus on `k*` and a **background condition at a user-chosen rank**, inspired by [DESeq2](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8) (Love, Huber, Anders, 2014) and adapted to the **one-replicate-per-condition** setting this package targets. Steps:

1. **Size factors.** DESeq2's median-of-ratios, computed jointly across all conditions from loci with positive signal in every condition.
2. **Dispersion.** With no within-condition replicates, per-locus variance cannot be estimated from repeated observations. A dispersion parameter `α` is estimated *across* loci under the null-majority assumption. Default (`--fit-type common`) uses the median of per-locus method-of-moments estimates, scaled by a closed-form correction for the small-df median-of-χ² bias; an alternative parametric trend `α(μ) = a/μ + b` (`--fit-type parametric`) is available, fit by robust (MAD-trimmed) weighted least squares and subject to the same scale correction.
3. **Pairwise NB-GLM LRT + Bonferroni + BH.** Per locus, identify `k*` = condition with the highest size-factor-normalized signal and `k_bg` = condition at rank `--background-rank` (default 3, so the third-highest condition by normalized signal; 1 = `k*`, 2 = runner-up). The null (`μ_{k*} = μ_{k_bg}`) is fit by intercept-only NB MLE on this pair; the alternative is saturated on this pair with the constraint `μ_{k*} > μ_{k_bg}` (always satisfied by construction). The remaining K−2 conditions enter both models as saturated nuisance and cancel from the likelihood ratio. Under the null with `k*` fixed, the statistic `T = 2·(ℓ_alt − ℓ_null)` follows the chi-bar-squared `½·χ²(0) + ½·χ²(1)` distribution — a standard result for LRTs on a boundary constraint (Self & Liang, *JASA* 1987; Silvapulle & Sen, *Constrained Statistical Inference*, 2005). The per-locus one-sided p-value is multiplied by K (Bonferroni) to correct for selecting `k*` by argmax; the choice of `k_bg` is deterministic given the ordering and adds no extra Bonferroni cost. Benjamini–Hochberg q-values then control the FDR across loci.

The rank knob controls robustness to "competing peaks" — loci where more than one condition is active. At `--background-rank 2`, the test compares `k*` to the runner-up: any second condition that is also elevated shrinks the gap and the LRT collapses to ~0. The default of 3 compares against the third-ranked condition, so a single competing peak does not depress the test; values >3 tolerate more competing peaks at the cost of comparing `k*` against an increasingly-low background. `--background-rank` is silently capped to K when larger, so the default just works at K = 2. Restricting the test to a single pair of conditions is what keeps it *enrichment-only*: a depletion pattern (one condition far below an otherwise-uniform set) has its top conditions all at the high level, so the test does not reject — exactly the behavior we want when looking for "fertile ground" loci.

> **FDR assumes independent (or positively dependent) loci.** BH controls FDR under independence or positive regression dependency (PRDS). BigWig signal on adjacent windows is correlated — overlapping tiling windows, shared peaks, or bin sizes smaller than the underlying signal's autocorrelation length all introduce dependence. For typical "one row per peak / per gene" inputs this is fine. For dense sliding-window inputs (e.g. 100 bp windows stepped every 50 bp), q-values will be optimistic; prefer non-overlapping windows or thin by autocorrelation length before trusting the FDR.

Columns appended to the output:

| column | meaning |
| --- | --- |
| `effect_size` | log2((X_{k\*}/s_{k\*}) + pc) − log2(mean_{j≠k\*}(X_j/s_j) + pc), i.e. the log2 fold change of the enriched condition vs the mean of the other K−1 conditions on the size-factor-normalized scale (computed as a log-difference; equivalent to a log2 ratio when pc is small). Always non-negative by construction. This is a user-facing summary; the test statistic itself is computed only from the `(k*, k_bg)` pair. |
| `p_value` | Bonferroni-corrected one-sided LRT p-value: `min(K · ½ · χ²(1).sf(T), 1)` |
| `q_value` | Benjamini–Hochberg q-value |
| `enriched_condition` | name of the condition column with the highest normalized signal (k\*). Always populated (it is just `argmax(X/s)`), so it is only meaningful for rows that pass a significance threshold — on a row with `q_value ≈ 1` it is whichever column happened to be highest under noise, not a call. |
| `effect_size_pc_dominated` | `True` when some `X_j/s_j < pseudocount` for that locus, meaning the log2 effect size is dominated by `pc` rather than data. Treat the effect size as a lower bound. The LRT itself is unaffected. |
| `lrt_zero_dominated` | `True` when `X_{k*} == 0` or `X_{k_bg} == 0`, i.e. the LRT pair contains a zero. The reported p-value for these loci is driven by the internal `mu_alt` floor (1e-20) rather than by data, and they tend to dominate the top of sparse-data output as spurious hits. Treat with skepticism — common causes are regions of poor mappability or chromosome-naming mismatches in some tracks. |
| `lrt_convergence_failed` | `True` when the intercept-only NB MLE for the null fit did not converge for that locus. Its `p_value` has been set to 1.0; the column is present so users can audit how many loci hit this case. |

Size factors, the number of loci that contributed to the size-factor estimate, the estimated dispersion fit, its trend coefficients, the test's effective conservativeness at the current K, and the number of kept/total loci are printed to stderr. The dispersion-fit label is one of:

| label | meaning |
| --- | --- |
| `common` | `--fit-type common` succeeded (single α = median of per-locus MoM estimates) |
| `parametric` | `--fit-type parametric` succeeded (fitted `α(μ) = a/μ + b`) |
| `common-fallback` | `--fit-type parametric` was requested but failed; fell back to the common fit |
| `override` | `--dispersion` was supplied; the fixed α was used at every locus |
| `zero` | `--fit-type zero` was supplied; Poisson was forced at every locus |

Size-factor spread, Poisson fallbacks, `--fit-type zero`, and `common-fallback` all emit a `FertilizerEnrichmentWarning` on stderr.

**Most users should keep the defaults.** `--fit-type common` with the default `--min-signal 5.0` and `--pseudocount 0.5` works well across most datasets. Touch the dispersion knobs only if (a) you have a strong prior that dispersion trends with mean signal (try `--fit-type parametric`), (b) you have an external estimate of α (pass `--dispersion`), or (c) you want a sensitivity analysis (`--dispersion 0.05` and `--dispersion 0.10`).

| CLI flag | effect |
| --- | --- |
| `-i`, `--input` | input TSV |
| `-c`, `--conditions` | two or more column names to compare |
| `-o`, `--output` | output TSV, filtered to loci passing the threshold, with extra columns appended |
| `--q-threshold` | keep loci with q ≤ this (default `0.05`; set to `1.0` to keep all rows) |
| `--p-threshold` | additionally keep only loci with raw p ≤ this (default: off) |
| `--fit-type` | dispersion model: `common` (default, median of MoM estimates, bias-corrected), `parametric` (fits `α(μ) = a/μ + b`), or `zero` (forces Poisson — diagnostic only, strictly anti-conservative if real overdispersion exists) |
| `--min-signal` | minimum mean normalized signal for loci included in dispersion estimation (default `5.0`) |
| `--dispersion` | override the fitted α with a fixed value applied to every locus; bypasses `--fit-type` and `--min-signal` entirely and sets `dispersion_fit = override`. Useful for sensitivity analyses (e.g. re-run at 0.05 and 0.10 to see how much calls depend on α) |
| `--size-factors` | externally-supplied size factors, one positive value per `-c` entry in the same order. Bypasses median-of-ratios. Use when you have an external normalization you trust more (RPM/RPKM, spike-in). Pass `1 1 1 ...` to disable normalization entirely. |
| `--background-rank` | rank of the condition compared against `k*` in the LRT pair (default `3` — tolerates one competing peak; `2` compares against the runner-up; larger values tolerate more competing peaks). Capped to K when larger; the default therefore works at K = 2 without special-casing. |
| `--pseudocount` | pseudocount for the effect-size log2 transform only; does not affect the LRT. Must be > 0 (default `0.5`) |
| `--allow-non-sum` | bypass the check that the input was produced by `fertilizer extract --stat sum`. The NB-GLM assumes count-like input; `mean`/`max`/`min`/`std`/`coverage` are not counts, so p-values may be miscalibrated. Use only after empirically verifying calibration on your data. |

**Differences from DESeq2** (non-exhaustive):

- **Enrichment-only, rank-pair LRT.** DESeq2's LRT is two-sided across an arbitrary `full` vs `reduced` design and fires on both enrichment and depletion. We hard-code a 1-df one-sided LRT comparing `k*` (the argmax) against the condition at a user-chosen rank (`--background-rank`, default 3, which tolerates one competing peak), with Bonferroni × K for picking `k*` as the empirical argmax. Loci where one condition is *depleted* relative to the rest are not called by construction.
- **Per-locus dispersion MLE.** DESeq2 uses Cox-Reid adjusted profile likelihood; we use method-of-moments. MoM is less efficient per locus but is consistent and does not fail to converge.
- **Shrinkage.** DESeq2 shrinks per-locus dispersion toward the trend via a log-normal empirical-Bayes prior and retains dispersion outliers. We use the trend value directly for every locus (equivalent to infinite shrinkage, no outlier retention) — robust for the small-K setting, but cannot capture genuinely heterogeneous per-locus dispersion.
- **Bias correction.** Because median(χ²(K−1))/(K−1) < 1 for small K (0.69 at K=3, 0.84 at K=5, 0.91 at K=8), median-of-MoM underestimates α. We apply the closed-form correction.
- **Log2 fold change shrinkage.** DESeq2 optionally shrinks LFC estimates (apeglm / ashr); we report a raw log2 fold change of the enriched condition vs the mean of the others as the effect size.
- **Observation-level outliers.** DESeq2 uses Cook's distance to flag and optionally refit without outliers. We don't.
- **Independent filtering.** DESeq2 filters low-signal loci out of multiple-testing correction to maximize power at a given FDR. We don't — use `--min-signal` (dispersion-only) or pre-filter the input TSV if you want this.
- **Integer counts.** DESeq2 is designed for integer RNA-seq counts; the NB likelihood here is evaluated with `scipy.special.gammaln` and is numerically correct for any non-negative float input. (This matches the common practice of passing fractional RSEM/salmon expected counts to DESeq2 via `tximport`, and is required here because bigWig region means are real-valued.)

**Known calibration behavior.** The Bonferroni × K correction for the data-driven argmax makes this test **conservative under the null** at K ≥ 4, increasingly so as K grows. Empirical Type-I rates at nominal α = 0.05 on Poisson nulls under the default `--background-rank 3`: K = 2 → ≈0.05 (rank capped to 2; `k*` vs runner-up); K = 3 → ≈0.07–0.09 (rank 3 is the lowest of three conditions, the widest order-statistic gap, so the test runs approximately nominal here); K = 4 → ≈0.015; K ≥ 5 → well under 0.005. With `--background-rank 2` the test is uniformly conservative across all K: K = 3 → ≈0.008, K ≥ 5 → well under 0.001. The test suite verifies Type-I error at α = 0.05 stays under 0.10 at K = 3 and under 0.05 for K ≥ 4 under the default, and under 0.08 across the same grid at `--background-rank 2`. Power is preserved against strong effects (≥80% at 2× fold change, ≥99% at 3× fold change across K ∈ {2…8} in simulations). At low μ (< ~5) the delta-method and the median-bias correction both degrade, and low-μ loci are excluded from dispersion estimation via `--min-signal`. Depletion-only patterns (one condition low, the rest uniform) are simulated in the test suite and confirmed *not* to be called.

> **Supply a mix of positive and negative loci.** The size-factor and dispersion steps both assume that *most* loci are **not** enriched — the "null majority" is what anchors the normalization and the variance estimate. If you run `enrich` on a set of regions pre-filtered to be those you expect to change, you will get sub-optimal results: the estimated library-size differences will absorb real biological differences, and the dispersion estimate will be inflated by the true positives. For best results pass a genome-wide or background-matched set of loci containing both putative-enriched and expected-stable regions.

Example (three conditions):

```bash
fertilizer enrich -i signals.tsv -c A B C -o enrichment.tsv --q-threshold 0.05
```

Output `enrichment.tsv` (filtered, numbers illustrative; `effect_size_pc_dominated`, `lrt_zero_dominated`, `lrt_convergence_failed` columns omitted from the example for brevity):

```
chrom   start   end     A     B     C       effect_size     p_value     q_value     enriched_condition
chr1    0       1000    2.0   2.1   8.4     2.07            0.0004      0.0032      C
chr3    1200    2200    15.2  3.6   4.1     2.08            0.0011      0.0060      A
chr7    900     1900    8.8   8.9   30.1    1.78            0.0018      0.0090      C
```

## Python API

Both subcommands are thin wrappers around library functions. The same work can be done from Python:

```python
import numpy as np
import pandas as pd

from fertilizer.extract import bigwig_region_means, load_regions, FertilizerWarning
from fertilizer.enrichment import (
    enrichment_analysis,
    size_factors,
    bh_qvalues,
    EnrichmentResult,
    FertilizerEnrichmentWarning,
)

# --- extract: region means for one bigWig -------------------------
regions = load_regions(["regions.bed"])                  # chrom/start/end
values, issues = bigwig_region_means(regions, "A.bw")    # np.ndarray, set[str]
# `issues` is a subset of {"missing_chrom", "out_of_bounds", "invalid_region"}

# --- enrich: enrichment NB-GLM LRT on a (n_loci, n_conditions) array
counts = pd.read_csv("signals.tsv", sep="\t", comment="#")[["A", "B", "C"]].to_numpy(float)
res: EnrichmentResult = enrichment_analysis(counts, fit_type="common")
# Per-locus arrays (length n_loci):
#   res.p_value, res.q_value, res.effect_size, res.lrt_stat,
#   res.per_locus_dispersion, res.enriched_condition_idx,
#   res.effect_size_pc_dominated, res.lrt_zero_dominated,
#   res.lrt_convergence_failed
# Per-condition array (length n_conditions):
#   res.size_factors
# Scalars:
#   res.n_loci_for_size_factors (int), res.dispersion_fit (str),
#   res.dispersion_trend (tuple[float, float]), res.background_rank (int)
```

`FertilizerWarning` (locus-level issues from `extract`) and `FertilizerEnrichmentWarning` (size-factor spread, Poisson fallbacks from `enrich`) are both `UserWarning` subclasses — catch them with `warnings.catch_warnings()` or filter them with `warnings.simplefilter(..., FertilizerWarning)`.

## Project layout

```
fertilizer/
├── pyproject.toml             # package metadata + uv/hatchling build config
├── README.md
├── LICENSE
├── src/
│   └── fertilizer/
│       ├── __init__.py
│       ├── cli.py             # top-level argparse dispatcher
│       ├── extract.py         # signal aggregation + `extract` subcommand
│       └── enrichment.py      # enrichment analysis + `enrich` subcommand
├── examples/
│   ├── make_demo_data.py     # generates a synthetic end-to-end demo dataset
│   └── README.md             # walkthrough of `extract` + `enrich` on demo data
└── tests/
    ├── test_extract.py
    └── test_enrichment.py
```

## Troubleshooting / FAQ

**`pyBigWig` won't install.** It needs system `libcurl` and `libssl` headers.
On Debian/Ubuntu: `sudo apt-get install libcurl4-openssl-dev libssl-dev zlib1g-dev`.
On macOS: `brew install curl openssl`.

**`extract` output is all zeros.** Most often a chromosome-naming mismatch
between the BED and the bigWig (`chr1` vs `1`). `extract` warns when more
than 95% of cells are exactly zero — re-check the inputs. To inspect a
bigWig's chromosome names: `python -c "import pyBigWig; print(pyBigWig.open('A.bw').chroms())"`.

**`enrich` gives `q_value` near 1 for everything.** Four common causes,
in order of likelihood:
1. K is large (≥ 5) and the Bonferroni × K argmax correction makes the test
   very conservative. The stderr output prints the effective Type-I rate
   at α=0.05 for your K and rank; if it is far below nominal, low power
   is by construction. Lower K (combine biologically equivalent conditions)
   or accept that strong effects only will be called.
2. Your enriched loci have a **competing peak** (a second condition also
   elevated). The default `--background-rank 3` tolerates one such peak,
   but if you have two or more, raise it (e.g. `--background-rank 4`).
   `--background-rank 2` is the most fragile choice in this regard.
3. Pre-filtering. If you passed a region set already enriched for the
   conditions you care about, the null-majority assumption is violated;
   the size factors absorb real differences and dispersion is inflated.
   Pass a genome-wide / background-matched region set instead.
4. The dispersion estimate fell back to Poisson (`alpha=0`) because too
   few loci passed `--min-signal`. Stderr will say so. Lower `--min-signal`
   or supply more loci.

**I have replicates per condition.** This package targets the
one-replicate-per-condition setting. With replicates, prefer DESeq2 or
edgeR — they estimate per-locus dispersion from within-condition variance
and give better power. `fertilizer` is the tool to reach for when you
have one bigWig per condition (a common setup for ChIP-seq, ATAC-seq, and
many predictive models) and want enrichment calls without making up
fake replicates.

**Which conditions should I use as "background"?** None — the test is
symmetric and compares each condition against the others. Just supply
all of your conditions as `-c`.

**Can I pre-normalize my data?** Yes. Either (a) pass `--size-factors V1 V2 ...`
matched to your `-c` entries to inject an external normalization (RPM/RPKM,
spike-in, etc.), or (b) pass `--size-factors 1 1 1 ...` to disable
normalization entirely if your inputs are already on the same scale.

**`effect_size_pc_dominated == True` for some loci.** Those loci have at
least one condition with normalized signal below `--pseudocount`, so the
log2 ratio is dominated by the pseudocount rather than data — treat the
effect size as a lower bound only. The p-value/q-value are unaffected
(they don't use the pseudocount).

## Citation

If you use `fertilizer` in published work, please cite:

> Schreiber, J. *fertilizer: per-region enrichment from single-replicate
> bigWig signal across conditions.* https://github.com/jmschrei/fertilizer,
> v0.1.0 (2026).

## License

MIT — see [LICENSE](LICENSE).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
