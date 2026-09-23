# Python API

The top-level package exports only `__version__`; import from the submodules.

## Run the CLI from Python

The simplest route to the exact CLI behavior, including the multi-threaded
extract and all warnings:

```python
from fertilizer.cli import main

rc = main(["extract", "-w", "A.bw", "B.bw", "C.bw", "-b", "regions.bed",
	"-o", "signals.tsv", "-s", "sum", "-j", "4"])
rc = main(["enrich", "-i", "signals.tsv", "-c", "A", "B", "C", "-o", "enrichment.tsv"])
rc = main(["extract", "-a", "A.bam", "B.cram", "-b", "regions.bed", "-o", "counts.tsv", "-j", "8"])
```

`main` returns 0 on success and 2 on a user-input error, printing
`fertilizer: error: ...` to stderr instead of raising. **Check the return
code**; a failed call does not raise.

## `enrichment_analysis`

```python
import pandas as pd
from fertilizer.enrichment import enrichment_analysis

conds = ["A", "B", "C"]
df = pd.read_csv("signals.tsv", sep="\t", comment="#", dtype={"chrom": str})
res = enrichment_analysis(df[conds].to_numpy(float))
```

```python
enrichment_analysis(
	counts,                        # (n_regions, K) non-negative, finite float, K >= 2
	pseudocount=0.5,               # effect size only; > 0
	fit_type="common",             # "common" | "parametric" | "zero"
	dispersion_min_signal=5.0,     # CLI --min-signal
	dispersion_override=None,      # CLI --dispersion; finite and >= 0
	size_factor_warn_ratio=5.0,    # warn when max(sf)/min(sf) exceeds this
	size_factors_override=None,    # CLI --size-factors; length-K array
	background_rank=3,             # int >= 2; capped to K
) -> EnrichmentResult
```

Returns every region. It does **not** apply `--q-threshold`, check the
`stat=sum` header, check for overlapping regions, or print the stderr
diagnostics; those live in the CLI wrapper, as does the K = 3 calibration
warning. Report instead: `res.size_factors`, `res.n_loci_for_size_factors`,
`res.dispersion_fit`, `res.dispersion_trend`, `res.background_rank`, and the
number of rows passing your threshold. The CLI's overlap check, which warns
above 1%:

```python
r = df.sort_values(["chrom", "start", "end"])
same = r["chrom"].to_numpy()[1:] == r["chrom"].to_numpy()[:-1]
print("overlapping adjacent pairs:", (same & (r["start"].to_numpy()[1:] < r["end"].to_numpy()[:-1])).mean())
```

| `EnrichmentResult` field | Shape / type |
|---|---|
| `p_value`, `q_value`, `effect_size`, `lrt_stat`, `per_locus_dispersion` | `(n,)` float |
| `enriched_condition_idx` | `(n,)` int, index into the columns passed |
| `effect_size_pc_dominated`, `lrt_zero_dominated`, `lrt_convergence_failed` | `(n,)` bool |
| `size_factors` | `(K,)` |
| `n_loci_for_size_factors` | int; 0 when `size_factors_override` was given |
| `dispersion_fit` | `"common"`, `"parametric"`, `"common-fallback"`, `"override"`, `"zero"` |
| `dispersion_trend` | `(a, b)` with α(μ) = a/μ + b |
| `background_rank` | the rank actually used, after capping |

Rebuild the CLI table:

```python
import numpy as np

df["effect_size"] = res.effect_size
df["p_value"] = res.p_value
df["q_value"] = res.q_value
df["enriched_condition"] = np.asarray(conds)[res.enriched_condition_idx]
df["effect_size_pc_dominated"] = res.effect_size_pc_dominated
df["lrt_zero_dominated"] = res.lrt_zero_dominated
df["lrt_convergence_failed"] = res.lrt_convergence_failed
hits = df[df["q_value"] <= 0.05]
```

NaN or infinite values raise `ValueError: counts must be finite; ...`, and
negative values raise `ValueError: counts must be non-negative`.

## Smaller pieces

```python
from fertilizer.enrichment import size_factors, size_factors_with_n, bh_qvalues
from fertilizer.extract import load_regions, bigwig_region_means

sf = size_factors(counts)                  # (K,); ValueError if < 2 all-positive rows
sf, n_used = size_factors_with_n(counts)
q = bh_qvalues(p_values)

regions = load_regions(["a.bed", "b.bed"])        # DataFrame: chrom, start, end, [name, score, strand, bed_col_6, ...]
values, issues = bigwig_region_means(regions, "A.bw", stat="sum")
# values: (n,) float64; issues: subset of {"missing_chrom", "out_of_bounds", "invalid_region"}
```

`bigwig_region_means` defaults to `stat="mean"`: pass `stat="sum"`. It is
serial, opens one bigWig, and emits no warnings; inspect `issues` yourself.

Counting BAM/CRAM and fragment files, serially and without the CLI's locus
checks or warnings:

```python
from fertilizer.counting import BarcodeGroups, count_bam, count_fragments, count_issues

arrays = (regions["chrom"].to_numpy(), regions["start"].to_numpy(), regions["end"].to_numpy())
c = count_bam("A.bam", *arrays, pos_shift=4, neg_shift=-5, min_mapq=30)   # (n, 1) int64
groups = BarcodeGroups.from_table("cells.tsv", group_column="cluster")
f, seen = count_fragments("fragments.tsv.gz", *arrays, groups=groups)    # (n, len(groups.names))
issues, bad = count_issues(*arrays, None, seen)                           # fragments: no lengths
```

`count_bam` skips duplicate, secondary, supplementary and QC-fail reads by
default (`skip_flags=0` counts them all) and streams the whole file; the CLI
additionally splits indexed files across processes.

## Warnings

```python
import warnings
from fertilizer.extract import FertilizerWarning
from fertilizer.enrichment import FertilizerEnrichmentWarning

with warnings.catch_warnings(record=True) as caught:
	warnings.simplefilter("always")
	res = enrichment_analysis(counts)
for w in caught:
	if issubclass(w.category, FertilizerEnrichmentWarning):
		print(w.message)
```

Both are `UserWarning` subclasses. `FertilizerWarning` comes from extract
(locus problems, >95% zeros); `FertilizerEnrichmentWarning` from enrich (size
factor spread, Poisson fallback, `fit_type="zero"`, non-convergence).
