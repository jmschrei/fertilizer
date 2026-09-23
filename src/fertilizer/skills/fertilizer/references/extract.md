# `fertilizer extract`

Sums (or otherwise summarizes) each bigWig over each BED region and writes one
row per region, one column per bigWig.

```bash
fertilizer extract -w liver.bw heart.bw brain.bw -b background.bed \
	-o signals.tsv -s sum -j 8 2> extract.log
```

Warnings go to stderr; keep them (`2> extract.log`) and read them before
`enrich`.

| Flag | Default | Effect |
|---|---|---|
| `-w`, `--bigwigs` | required | one or more bigWigs |
| `-b`, `--beds` | required | one or more BED files, concatenated in order |
| `-o`, `--output` | required | output TSV; gzipped when the name ends `.gz` |
| `-s`, `--stat` | **`mean`** | `mean`, `max`, `min`, `sum`, `std`, `coverage` (pyBigWig `stats(type=..., exact=True)`). **Use `sum` for `enrich`** |
| `-n`, `--names` | filename stems | column names, one per `-w` entry, same order |
| `-j`, `--n-jobs` | `-1` = every core | worker threads. Always set it; on a shared machine take at most half the free cores |

## Output

```
# fertilizer-extract stat=sum
chrom	start	end	A	B	C
chr1	127	327	4000.0	5400.0	5200.0
```

- Line 1 is the metadata header `enrich` reads to enforce `-s sum`. Read the
  file in pandas with `comment="#"` (or `skiprows=1`).
- Rows are in input order (concatenated BEDs), not sorted.
- BED columns 4–6 pass through as `name`, `score`, `strand`; columns 7+ as
  `bed_col_6`, `bed_col_7`, ... (0-based index). narrowPeak `signalValue` is
  therefore `bed_col_6`. Give every BED the same number of columns, or the
  short ones get NaN in the extra columns.
- Values are never NaN. A region with no coverage gives 0. In a partly covered
  region, uncovered bases are skipped, not counted as 0: `mean`, `min`, `max`
  and `std` describe the covered bases only, `sum` adds only covered bases,
  and `coverage` is the covered fraction.
- fertilizer 0.1.0 read statistics from bigWig zoom levels, which made `-s sum`
  wrong by orders of magnitude for regions of a few hundred bp or more whenever
  the bigWig has zoom levels (`header()["nLevels"] > 0`). Re-run `extract` on
  any `signals.tsv` made with 0.1.0.

## Column names

The stem of each path: `liver.bw` → `liver`, `ENCFF123ABC.bigWig` →
`ENCFF123ABC`, `x.+.bw` → `x.+`. Two paths with the same stem are rejected:

```
fertilizer: error: duplicate column names from bigWig filename stems would collide: ['A']
```

A name that matches a BED column present in the input (`chrom`, `start`,
`end`, `name`, `score`, `strand`, `bed_col_<i>`) is rejected too.

Pass `-n` whenever stems are accessions, contain strand characters, or
collide. `-n` may also reuse one bigWig under two names.

## Coordinates and locus problems

BED is 0-based half-open: `chr1 100 200` covers bases 100–199. Subtract 1 from
the start of a 1-based file before extracting.

A region with a problem gets `0.0` and one `FertilizerWarning` per issue type
per bigWig, naming the first chromosome it hit:

| Issue | Warning says | Usual cause |
|---|---|---|
| chromosome absent from bigWig | `chromosome '1' (and possibly others) referenced in BED but missing from bigWig ...` and lists the bigWig's chroms | `chr1` vs `1` naming |
| end past chromosome length | `some regions extend beyond the chromosome length` | BED and bigWig on different assemblies |
| `start < 0` or `start >= end` | `some regions have non-positive length or a negative start` | malformed BED |

Independently, more than 95% exact zeros in the output gives
`N% of region-by-bigWig cells are exactly zero`. The file is still written and
the exit code is 0. The 95% test is over the whole matrix, so one bad track
among many passes it; check each column:

```python
import pandas as pd
df = pd.read_csv("signals.tsv", sep="\t", comment="#", dtype={"chrom": str})
tracks = [c for c in df.columns if c not in ("chrom", "start", "end", "name", "score", "strand")
	and not c.startswith("bed_col_")]
print((df[tracks] == 0).mean().sort_values(ascending=False))    # fraction zero per track
print("regions positive in every track:", int((df[tracks] > 0).all(axis=1).sum()))
```

## Choosing `-s`

| Stat | For |
|---|---|
| `sum` | `enrich`. Also the right input for any count-based comparison |
| `mean` | per-base average for plots or ranking; `enrich` refuses it |
| `max`, `min`, `std`, `coverage` | QC; `coverage` is the fraction of bases with data. `enrich` refuses all four |

`enrich --allow-non-sum` bypasses the refusal. Do not use it to make the error
go away; re-run `extract -s sum` instead, which is cheap.

Python equivalent (one bigWig at a time, serial): `references/python-api.md`.
