---
name: fertilizer
description: Call genomic regions where one condition's signal is significantly enriched over the others, from one bigWig, BAM or 10x fragment file per condition, with the `fertilizer` CLI (`fertilizer extract`, `fertilizer enrich`) or its Python API (`fertilizer.enrichment.enrichment_analysis`). Use when asked to find condition-specific, cell type-specific or differential regions/peaks/enhancers from bigWigs without replicates; to aggregate bigWigs, or count BAM reads or fragment insertions, over BED regions (including per-cluster pseudobulks of one fragment file); to pick "fertile ground" starting regions for regulatory DNA design; to choose `--background-rank`, `--size-factors` or dispersion settings; to interpret `effect_size`, `q_value`, `enriched_condition` or the `lrt_zero_dominated` flag; or to debug all-zero extract output, "q-values all near 1", the `--stat sum` refusal, or chromosome-naming warnings. Router skill — read the relevant file under `references/` before writing any fertilizer command or code.
---

# fertilizer

`fertilizer` takes **one bigWig, BAM or fragment file per condition** (or one
fragment file split by a barcode-to-group table) and a BED of regions, sums each
track's signal or counts reads/insertions over each region (`extract`), and runs a one-sided
negative-binomial likelihood-ratio test per region asking whether the
top-ranked condition is significantly above the others (`enrich`). Output is a
TSV with `effect_size` (log2), `p_value`, `q_value` and `enriched_condition`.
PyPI name `fertilizer-genomics`, import name `fertilizer`, CLI `fertilizer`.

## Wrong tool when

| Situation | Use instead |
|---|---|
| Replicates per condition | DESeq2 / edgeR / csaw on the replicate counts (`references/inputs.md` §Replicates) |
| You want regions *depleted* in one condition | not tested by `enrich`; pairwise workaround in `references/recipes.md` |
| Most regions genuinely change, or a global shift in the mark | only with external `--size-factors` (spike-in); `references/choosing-parameters.md` |
| Tracks are `-log10` p-value, log-scale, z-score or otherwise non-count | count the BAMs or fragment files directly with `extract -a`/`-f` (`references/inputs.md`). p-value tracks run without error and give meaningless calls |
| Dense overlapping / sliding windows | thin to non-overlapping windows first |

## Rules that hold everywhere

- **`extract -s sum`**, always, when bigWig output feeds `enrich`. The default
  stat is `mean`, and `enrich` refuses it. BAM (`-a`) and fragment (`-f`) input
  is always counted and needs no `-s`.
- **Pass `-j N` to `extract`.** The default `-1` takes every core.
- **Pass a background-matched region set** (union of all conditions' peaks,
  cCREs, genome-wide windows), never one pre-filtered to expected hits.
- **Capture stderr from `enrich`.** Size factors, dispersion fit and kept/total
  counts are printed there only.
- **`enrichment.tsv` is filtered to `q ≤ 0.05` and kept in input order**, not
  sorted. Use `--q-threshold 1.0` to keep every row.
- **Drop `lrt_zero_dominated` and `lrt_convergence_failed` rows** before
  treating calls as real.
- **`enriched_condition` is always filled in** (argmax), so it is a call only on
  rows that pass the threshold.
- **Choose `--background-rank` deliberately.** Ask whether a region active in
  two conditions counts as a hit. At K ≥ 8 the default 3 has almost no power;
  never go above the highest safe rank (K − 1 up to K = 8, K − 2 beyond), which
  is anti-conservative (`references/choosing-parameters.md`).

## Task → reference

| If the task is… | Read |
|---|---|
| install, run the demo, or the minimal two-command pipeline | `references/quickstart.md` |
| choosing bigWigs, BAMs or fragment files and the region set; strands, replicates, chr naming, external count matrices | `references/inputs.md` |
| `fertilizer extract` flags, stats, BAM/fragment counting, Tn5 shifts, barcode groups, coordinates, warnings, output format | `references/extract.md` |
| `fertilizer enrich` flags, stderr diagnostics, output columns | `references/enrich.md` |
| how the test works; how it differs from DESeq2 | `references/method.md` |
| choosing `--background-rank`, K, dispersion settings; calibration and power | `references/choosing-parameters.md` |
| a specific question: regions specific to X, top N per condition, A vs B, depleted in X, sensitivity analysis, design templates | `references/recipes.md` |
| doing it from Python; reproducing the CLI table; catching warnings | `references/python-api.md` |

## Question → recipe

| User asks | Do |
|---|---|
| "Which regions are specific to condition X?" | ask whether shared-with-one-other counts; yes → `references/recipes.md` §1, no → §3 |
| "Top N regions per condition" / "rank every region" / plots | `--q-threshold 1.0`, drop flagged rows, group and rank — `references/recipes.md` §2 |
| "Just give me a region × bigWig signal matrix" | `extract` alone — `references/extract.md` |
| "Only in X" / "shared by A and B" | highest safe rank plus a runner-up margin — `references/recipes.md` §3 |
| "Higher in A than B" | K = 2 run on just A and B — `references/recipes.md` §4 |
| "Lower in X than everywhere else" | intersect K = 2 runs — `references/recipes.md` §5 |
| "How robust are these calls?" | `--dispersion` sweep — `references/recipes.md` §6 |
| "Starting regions for ledidi / regulatory design" | `references/recipes.md` §7 |
| "I have a fragments file and cluster labels" | `extract -f fragments.tsv.gz -g cells.tsv --group-column <col>` — `references/extract.md` §BAM and fragment input |
| "I have 8+ conditions" / pseudobulk bigWigs per single-cell cluster | `references/choosing-parameters.md` §Many conditions |
| "My normalization / spike-in factors" | `--size-factors` — `references/recipes.md` §8 |

## Symptom → reference

| If you hit… | Read |
|---|---|
| `input was produced by \`fertilizer extract --stat mean\`` | `references/troubleshooting.md` — re-run with `-s sum` |
| `... do(es) not apply to --bams input`, `--stat applies to bigWig input only`, `--groups table ...`, `could not open BAM/SAM/CRAM`, `could not read ... REF_PATH` | `references/troubleshooting.md` §extract errors |
| `counts must be finite; found N NaN or infinite value(s)` | `references/troubleshooting.md` — empty cells in the input TSV |
| `N% of region-by-bigWig (or -column) cells are exactly zero` / `chromosome '1' ... missing from bigWig` (or BAM, fragment file) | `references/troubleshooting.md` — chr naming or assembly |
| `at least 2 loci with positive signal in every sample are required` | `references/troubleshooting.md` |
| `q_value` near 1 everywhere, or zero rows kept | `references/troubleshooting.md` |
| K = 3 warning "about 1.6x nominal" | `references/choosing-parameters.md` §K = 3 |
| `size factors span Nx` warning | `references/choosing-parameters.md` §Null majority |
| `fewer than 10 loci passed --min-signal` / Poisson fallback | `references/troubleshooting.md` |
| `N% of adjacent regions overlap` | `references/inputs.md` §Region set |
| `duplicate column names ... would collide` | `references/extract.md` — `-n/--names` |
| many top hits carry `lrt_zero_dominated = True` | `references/enrich.md` §Output columns |
| `already exists. Re-run with --force` from `install-skill` | `references/quickstart.md` §Install |
