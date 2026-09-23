# Changelog

All notable changes to `fertilizer` will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to semantic versioning (the API is unstable until 1.0).

## [Unreleased]

### Added

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
  about the empirical Type-I rate being ~2× nominal.
- CLI: example invocations in subparser epilogs; clean error+exit-2 on
  user-input errors instead of raw Python tracebacks.

### Changed

- `enrich` reports `p_value = 1` and `lrt_convergence_failed = True` for
  loci whose null-fit NB MLE did not converge (previously: warning only).
- Top-level CLI description now describes the actual pipeline rather
  than referencing "regulatory design".
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
- `cli.main` catches `ValueError` / `FileNotFoundError` and emits
  `fertilizer: error: <msg>` to stderr with exit code 2, instead of
  surfacing a Python traceback for user-input errors.
- Renamed module-level `run` to `run_extract` / `run_enrich` to avoid
  the cross-module name collision.

## [0.1.0]

- Initial release.
