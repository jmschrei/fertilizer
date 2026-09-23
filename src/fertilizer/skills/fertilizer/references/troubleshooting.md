# Troubleshooting

## `enrich` refuses the input

```
fertilizer: error: input was produced by `fertilizer extract --stat mean`, which aggregates bigWig signal in a way that is NOT count-like; ...
```

`extract` defaults to `-s mean`. Re-run it with `-s sum`. Do not pass
`--allow-non-sum`.

## `extract` output is (nearly) all zeros

Warnings: `N% of region-by-bigWig cells are exactly zero`, often with
`chromosome '1' (and possibly others) referenced in BED but missing from bigWig`.
`extract` still exits 0 and writes the file. Causes, most likely first:

1. Chromosome naming: BED `1` vs bigWig `chr1`, or the reverse. The warning
   lists the bigWig's names. Rename the BED with the `awk` in
   `references/inputs.md` §Chromosome naming (a bare `sed 's/^/chr/'` also
   prefixes `track` lines and turns `MT` into `chrMT`).
2. Wrong or empty bigWig.
3. Different assemblies: expect the `out_of_bounds` warning too.

Commands to compare names: `references/inputs.md` §Chromosome naming. The
per-track zero fraction: `references/extract.md` §Coordinates and locus
problems.

## `at least 2 loci with positive signal in every sample are required to compute size factors`

Median-of-ratios needs regions with signal in every condition. Seen after an
all-zero extract (above), with a track that is empty over the region set, or
with very sparse data. Fix the input; if one track is legitimately empty,
remove it from `-c` or pass `--size-factors`.

## `q_value` near 1 everywhere, or `kept 0 / N loci`

Check in order:

1. **Region sums too small.** Typical sums below ~10, often with
   `alpha(mu) = 0/mu + 0` on the `dispersion fit` line, carry too little
   evidence to call anything. Check the track type or cluster size
   (`references/inputs.md` §Which bigWigs).
2. **Size factors.** One far from the others, or a `size factors span Nx`
   warning, means the null-majority assumption failed: a pre-filtered region
   set or a global shift (`references/choosing-parameters.md` §Null majority).
3. **External `--size-factors` far from 1.** Their overall scale changes the
   dispersion fit; rescale to geometric mean 1 (`references/recipes.md` §8).
4. **K and rank.** At K ≥ 8 the default rank 3 has almost no power; raise it
   to the highest safe rank and filter with the margin
   (`references/choosing-parameters.md`).
5. **Competing peaks.** At rank 2, any second active condition removes the
   call. Raise the rank if shared activity is acceptable.
6. **No signal.** Plot a few known condition-specific regions from the TSV to
   confirm the difference is there.

## `fewer than 10 loci passed --min-signal=5.0 ... falling back to Poisson (alpha=0)`

Poisson ignores overdispersion, so if the data are overdispersed the p-values
that follow are too small. Lower `--min-signal` only when the sums are small
for a known reason (short regions, low depth). Otherwise supply more regions,
or fix the track scale. `--dispersion A` with an external estimate is the other option.

## Most top hits have `lrt_zero_dominated = True`

The tested pair contains an exact zero, so the p-value is not from data. Drop
these rows. The flag needs all but the top r − 1 tracks to be zero: at rank = K
(K = 3 at the default) one bad bigWig flags every region it zeroes; otherwise
the rows point at sparse or unmappable regions, or several bad tracks. A single
broken track at larger K causes no flag at all; the per-track zero fraction
(`references/extract.md` §Coordinates and locus problems) finds it at any K,
and a low `size factors estimated from N / M loci` count on stderr hints at it.
Where the flagged rows come from:

```python
import pandas as pd
conds = ["liver", "heart", "brain", "kidney"]
d = pd.read_csv("all.tsv", sep="\t", dtype={"chrom": str})    # enrich --q-threshold 1.0
f = d[d["lrt_zero_dominated"]]
print(len(f), "/", len(d), "flagged")
print((f[conds] == 0).sum())                   # zeros concentrated in one track → that bigWig
print(f["chrom"].value_counts().head())        # concentrated on some chroms → their names
```

## `N% of adjacent regions overlap`

q-values will be optimistic. Merge or thin the regions and re-extract
(`references/inputs.md` §Region set).

## `intercept-only NB MLE did not converge for N locus/loci`

Those rows get `p_value = 1` and `lrt_convergence_failed = True`. A handful is
harmless; many suggest extreme values (check for a mis-scaled track).

## The K = 3 "about 1.6x nominal" warning

Printed on every K = 3 run at the default rank. The output is usable (realized
FDR 0.056 at q ≤ 0.05 in simulation); what to change depends on whether shared
peaks count: `references/choosing-parameters.md` §K = 3.

## `pyBigWig` fails to build

System headers: `references/quickstart.md` §Install.

## Tracebacks instead of `fertilizer: error:`

The CLI converts `ValueError`, `FileNotFoundError` and `FileExistsError` to
exit code 2. Any other exception surfaces as a traceback; report it at
https://github.com/jmschrei/fertilizer/issues with the command and input sizes.
