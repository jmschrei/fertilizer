# `fertilizer enrich`

```bash
fertilizer enrich -i signals.tsv -c liver heart brain kidney \
	-o enrichment.tsv 2> enrich.log
```

| Flag | Default | Effect |
|---|---|---|
| `-i`, `--input` | required | TSV with a header row; `.gz` read transparently |
| `-c`, `--conditions` | required | two or more column names to compare |
| `-o`, `--output` | required | output TSV; gzipped when the name ends `.gz` |
| `--q-threshold` | `0.05` | keep rows with `q_value ≤` this. `1.0` keeps every row |
| `--p-threshold` | off | additionally require `p_value ≤` this |
| `--background-rank` | `3` | rank of the condition the top one is tested against; capped to K. See `references/choosing-parameters.md` |
| `--fit-type` | `common` | `common`: one α for all regions. `parametric`: α(μ) = a/μ + b. `zero`: Poisson, diagnostic only, anti-conservative |
| `--min-signal` | `5.0` | regions with mean normalized signal below this are left out of dispersion fitting (still tested) |
| `--dispersion` | fitted | fixed α ≥ 0 for every region (0 is Poisson); bypasses `--fit-type` and `--min-signal` |
| `--size-factors` | median-of-ratios | one positive value per `-c` entry, same order; fertilizer divides by them. Rescale to geometric mean 1 (`references/recipes.md` §8). `1 1 1 ...` disables normalization |
| `--pseudocount` | `0.5` | effect-size log2 only; the test ignores it. Must be > 0 |
| `--allow-non-sum` | off | skip the `stat=sum` header check. Leave it off |

Keep the defaults unless one of `references/choosing-parameters.md`'s cases
applies. Pick `--background-rank` deliberately; it defines the question.

## Stderr

Everything below goes to stderr only. Save it (`2> enrich.log`) and report the
size factors, the dispersion label and the kept count to the user.

```
size factor A: 1.0073
size factor B: 0.9743
size factor C: 1.0222
size factors estimated from 200 / 200 loci (positive signal in every condition)
dispersion fit: common (alpha(mu) = 0/mu + 0.05593)
background rank: 3 (k* compared against the rank-3 condition; higher = more robust to competing peaks)
conservativeness: ... At K=3, rank=3, empirical T1@alpha=0.05 ~= 0.080.
kept 11 / 200 loci (q <= 0.05)
```

| Dispersion label | Means |
|---|---|
| `common` | one α = bias-corrected median of per-region estimates |
| `parametric` | trend `a/mu + b` fitted |
| `common-fallback` | parametric requested but failed; used `common` (warned) |
| `override` | `--dispersion` given |
| `zero` | `--fit-type zero` |

`alpha(mu) = 0/mu + 0` under `common` has two causes: the Poisson fallback
(fewer than 10 regions passed `--min-signal`; a warning says so), or a median
per-region estimate ≤ 0, meaning no more variance than Poisson (seen with
scaled-down tracks; no warning). See `references/troubleshooting.md`.

**The `conservativeness` line is only reliable for ranks 2 and 3.** It reads a
built-in table with rows for rank 2 and rank 3 only, and uses the rank-3 row
for any higher rank: at K = 4, rank 4 it printed `0.015` where the measured
rate is 0.114, and nothing warns. Use the tables in
`references/choosing-parameters.md`.

## Output columns

The input columns pass through unchanged (including BED `name`), and these are
appended:

| Column | Meaning |
|---|---|
| `effect_size` | log2((X_top/s_top) + pc) − log2(mean of the other K−1 normalized values + pc). ≥ 0 by construction. Computed against the mean of *all* others, not the tested background |
| `p_value` | one-sided LRT p-value × K (Bonferroni for picking the top by argmax), capped at 1 |
| `q_value` | Benjamini–Hochberg over all rows |
| `enriched_condition` | argmax of normalized signal. Filled on every row; a call only where `q_value` passes |
| `effect_size_pc_dominated` | some normalized value < pseudocount: `effect_size` is a lower bound. Test unaffected |
| `lrt_zero_dominated` | the background value (or the whole row) is exactly 0; the p-value comes from an internal floor, not data. **Drop these** |
| `lrt_convergence_failed` | null fit did not converge; `p_value` forced to 1 |

The three flag columns are always present. The output has no `#` header line.
Rows keep input order: sort by `q_value` yourself.

**`lrt_zero_dominated` rows sort to the top once sorted by `q_value`.** The
background is the rank-r value, so it is zero when all but the top r − 1
tracks read zero: one at rank = K (so a single broken track at K = 3 flags
and calls every region it zeroes), two at K = 4, rank 3. Such a region
(unmappable, or on a chromosome some bigWigs lack) gets a tiny p-value against
the zero. A row where every track is zero also carries the flag, with
`p_value = 1`. When many rows carry the flag, check chr naming and the extract
warnings before anything else.

## Errors (exit code 2, message on stderr)

| Message | Fix |
|---|---|
| `input was produced by \`fertilizer extract --stat mean\`...` | re-run extract with `-s sum` |
| `columns not found in signals.tsv: ['D']` | `-c` must match header names exactly; check `-n` from extract |
| `at least 2 loci with positive signal in every sample are required to compute size factors` | all-zero or near-empty tracks; `references/troubleshooting.md` |
| `--size-factors has 2 values but --conditions has 3` | one value per `-c` entry |
| `--background-rank must be >= 2` | rank 1 is the top condition itself |
| `--pseudocount must be > 0` | |
