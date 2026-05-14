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
    -o examples/signals.tsv
```

`signals.tsv` will have columns `chrom start end A B C`.

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

`enrichment.tsv` should contain ~10 rows, each with `enriched_condition == C`
and small q-values. If your run is non-deterministic (different RNG seed in
`make_demo_data.py`) you may see slightly different counts.

## Tearing down

```bash
rm examples/A.bw examples/B.bw examples/C.bw \
   examples/regions.bed examples/signals.tsv examples/enrichment.tsv
```
