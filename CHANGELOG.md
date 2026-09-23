# Changelog

All notable changes to `fertilizer` will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to semantic versioning (the API is unstable until 1.0).

## [Unreleased]

### Added

- `extract -a` reads CRAM as well as BAM/SAM. Only the fields the counts use
  are decoded (htslib `required_fields`), so no reference FASTA is needed. An
  indexed CRAM (`.crai`) is parallelized the same way as an indexed BAM.
- `extract -f` splits a single BGZF-compressed or uncompressed fragment file
  into byte ranges counted in parallel across `-j` workers, so one large file
  is no longer limited to one core. Ranges start at BGZF block boundaries
  (found by walking the block headers; no index needed) and every line is
  counted by exactly one range. Plain gzip is still read as one stream.
- `extract -a` splits an indexed BAM or CRAM into chromosome pieces (about
  four per `-j` worker) instead of one task per chromosome, so a
  single-chromosome file uses every worker and scaling no longer stops at the
  largest chromosome. Pieces are sized by the reads the index records per
  chromosome, empty chromosomes are skipped, and each read is counted by the
  piece where it starts.
- `extract` counts reads from BAM/SAM files (`-a/--bams`) and fragment ends
  from 10x fragment files (`-f/--fragments`), in addition to summarizing
  bigWigs. BAMs count each read's 5' end, skipping unmapped, duplicate,
  secondary, supplementary and QC-fail reads and MAPQ < 30 by default
  (`--min-mapq`, `--include-flagged`); fragment files count both ends of each
  fragment once. `-ps/--pos-shift` and `-ns/--neg-shift` shift read and
  fragment coordinates exactly as bam2bw does (`-ps 4 -ns -5` for Tn5).
  `-g/--groups` with `--barcode-column`/`--group-column` splits fragment files
  into one column per barcode group (pseudobulk). The output header records
  `stat=count`, which `enrich` accepts. `pysam` is now a dependency.
- `fertilizer install-skill` copies a bundled Claude Code skill into
  `~/.claude/skills/fertilizer` (`-d` to change the directory, `--symlink`,
  `-f/--force`). The skill covers inputs, both subcommands, the method,
  parameter choice with measured calibration, question-to-command recipes, the
  Python API and troubleshooting. Re-run `fertilizer install-skill --force`
  after upgrading to pick up corrections.
- `extract` writes a `# fertilizer-extract stat=<value>` metadata header
  so `enrich` can verify the upstream aggregation is count-like.
- `enrich` refuses input produced by `extract --stat <non-sum>` unless
  the new `--allow-non-sum` flag is passed.
- `lrt_zero_dominated` and `lrt_convergence_failed` columns in the
  `enrich` output, alongside the existing `effect_size_pc_dominated`.
  All three flag columns are now always emitted for schema stability.
- Per-locus convergence tracking in `_intercept_mle`: loci whose null-fit
  NB MLE didn't converge have their `p_value` set to 1.0 and are flagged
  via `lrt_convergence_failed`.
- BED column passthrough: `load_regions` keeps columns 4-6 as
  `name`/`score`/`strand` and any further columns as `bed_col_<i>`, so
  peak names survive the `extract → enrich` pipeline.
- Overlap detection at the start of `enrich`; emits a warning when
  >1% of adjacent input regions overlap (BH validity caveat).
- `FertilizerEnrichmentWarning` at K=3 with default `--background-rank 3`
  about the empirical Type-I rate being above nominal (~0.08).
- CLI: example invocations in subparser epilogs; clean error+exit-2 on
  user-input errors instead of raw Python tracebacks.

### Changed

- `enrich` reports `p_value = 1` and `lrt_convergence_failed = True` for
  loci whose null-fit NB MLE did not converge (previously: warning only).
- Top-level CLI description now describes the actual pipeline rather
  than referencing "regulatory design".
- README documents `.gz` input and output, how each `--stat` treats
  partially covered regions, that `lrt_stat` and `per_locus_dispersion` are
  Python-only, the `-j` threading model, and how to catch or silence both
  warning categories.
- README reordered so the quickstart demo immediately follows
  installation. Added a "When NOT to use this" section.

### Fixed

- `extract` now computes every statistic from the full-resolution bigWig data
  (`exact=True`). pyBigWig otherwise answers from a zoom level when the region
  is wide enough, and its zoom-level `sum` is wrong by orders of magnitude, so
  `extract -s sum` gave wrong values for regions of a few hundred bp or more on
  any bigWig with zoom levels (pyBigWig writes them by default; check
  `pyBigWig.open(path).header()["nLevels"]`). `mean`, `min`, `coverage` and
  `std` were approximate. Re-run `extract` on output made with 0.1.0.
- `extract` no longer caches open bigWig handles across calls. A bigWig
  rewritten at the same path in the same process (for example, from a notebook)
  was read from the stale handle and returned the old values.
- `enrichment_analysis` (and so `enrich`) raises `ValueError` on NaN or
  infinite counts. Previously a NaN, such as an empty cell in the input TSV,
  was accepted silently: that locus got the NaN column as its enriched
  condition, a NaN effect size and p = 1.
- `enrich --dispersion` and `enrichment_analysis(dispersion_override=...)`
  reject negative, NaN and infinite values. A negative value previously
  produced p-values from an invalid likelihood, and NaN set every p-value
  to 1, both without a warning.
- `extract` rejects track names (from `--names` or filename stems) that match
  a BED column in the input, such as `score` or `chrom`. Previously the
  track's values silently replaced that column. `enrich` likewise rejects
  condition columns named like one of its output columns (`p_value`,
  `effect_size`, ...), which were overwritten in the output.
- `extract` skips UCSC `track` and `browser` lines at the top of a BED file
  instead of failing with pandas' `ParserError: Error tokenizing data`.
- The K = 3 calibration text no longer contradicts itself. The README and
  docstrings called the default rank "approximately nominal" at K = 3 while the
  warning said "roughly 2x nominal"; both now say above nominal, and the warning
  derives its multiple from the rate it reports (0.080, about 1.6x).
- `cli.main` catches `ValueError` / `FileNotFoundError` and emits
  `fertilizer: error: <msg>` to stderr with exit code 2, instead of
  surfacing a Python traceback for user-input errors.
- Renamed module-level `run` to `run_extract` / `run_enrich` to avoid
  the cross-module name collision.

## [0.1.0]

- Initial release.
