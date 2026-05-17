# fertilizer demo

Tiny end-to-end example exercising both subcommands on synthetic data.

## 1. Generate the input files

```bash
python examples/make_demo_data.py
```

This writes:
- `examples/regions.bed` — 200 non-overlapping 200 bp windows on `chr1`
- `examples/A.bw`, `examples/B.bw`, `examples/C.bw` — three bigWigs with
  Poisson-distributed signal at mean 20. `C` has 10 randomly chosen
  windows boosted to mean 100, simulating a condition-specific enrichment.

## 2. Aggregate signal over the regions

```bash
fertilizer extract \
    -w examples/A.bw examples/B.bw examples/C.bw \
    -b examples/regions.bed \
    -o examples/signals.tsv \
    -s sum
```

`signals.tsv` will have columns `chrom start end A B C`. `-s sum` is
required here because the downstream `enrich` NB-GLM assumes count-like
input — `extract` writes the chosen stat into a metadata header and
`enrich` refuses non-`sum` input unless `--allow-non-sum` is passed.

## 3. Call enrichment

```bash
fertilizer enrich \
    -i examples/signals.tsv \
    -c A B C \
    -o examples/enrichment.tsv \
    --q-threshold 0.05
```

Stderr will print the size factors, the dispersion fit, the conservativeness
of the test at K=3, and the number of loci that passed the threshold.

`enrichment.tsv` should contain ~10 rows with small q-values. With the
default seed in `make_demo_data.py`, almost all are `enriched_condition == C`
(matching the 10 boosted regions). The occasional non-C row is a false
positive from the Poisson-noise background — expected at q ≤ 0.05 on 200
loci, and a useful reminder that the FDR threshold is statistical, not
absolute.

## Tearing down

```bash
rm examples/A.bw examples/B.bw examples/C.bw \
   examples/regions.bed examples/signals.tsv examples/enrichment.tsv
```
