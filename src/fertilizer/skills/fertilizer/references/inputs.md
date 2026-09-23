# Inputs: bigWigs, BAMs, fragment files and the region set

## Which input type

If the BAMs or 10x fragment files are available, count them (`extract -a` or
`-f`) rather than summing bigWigs: the values are true read or insertion counts,
which is what `enrich` models, and scale (below) stops being a question. A
single scATAC fragment file plus a barcode-to-cluster table gives one column per
cluster with `-g` (`references/extract.md` §BAM and fragment input). Use
bigWigs when they are all you have, or for predicted tracks.

## Which bigWigs

One bigWig per condition, same assembly, same chromosome naming, same assay,
**same output type**. Mixing a fold-change track with a p-value or coverage
track makes the within-region comparison measure the file type, not the
biology.

`enrich` models each region's summed signal as negative-binomial counts, so the
sums must behave like counts: non-negative, and variance growing with the mean.

| Track type | Use? |
|---|---|
| read or fragment coverage (`bamCoverage --normalizeUsing None`), CPM/RPM | yes |
| ENCODE fold-change over control | runs; a ratio to control is not a count and calibration on it has not been validated, so treat q-values as a ranking |
| predicted signal from a sequence model (Cherimoya, ChromBPNet, ...) | runs; calibration on predictions has not been validated, so treat q-values as a ranking |
| `-log10` p-value tracks | no. Non-negative, so they run without error and give meaningless calls. ENCODE ships these next to fold-change tracks; check the output type |
| log-transformed, z-scored, or any track with negative values | no. Negative values make `enrich` raise `counts must be non-negative` |

**Scale matters.** `-s sum` returns the sum of per-base values over the region,
so a flat track at 20 over a 200 bp region sums to 4000; a coverage bigWig sums
to reads × fragment length. In a seeded simulation (K = 4, 5,000 regions, 5%
at 4×):

| sums multiplied by | true calls / 250 | false calls |
|---|---|---|
| 0.01 | 0 | 0 |
| 1 | 170 | 0 |
| 100 | 124 | 0 |

Scaling up inflates the fitted dispersion and costs some power; the null rate
rises slightly (K = 3, rank 3, Poisson null: 0.086 at ×1, 0.101 at ×200).
Scaling down so that most regions' mean normalized sum falls below
`--min-signal` (5.0) breaks dispersion estimation and calls nothing. If typical
region sums are below ~10, check the track type (or, for pseudobulk, the
cluster size) before anything else:

```python
import pandas as pd
df = pd.read_csv("signals.tsv", sep="\t", comment="#")
print(df.drop(columns=["chrom", "start", "end"]).describe().loc[["50%", "max"]])
```

## Region set

**This choice decides whether the output means anything.** Size factors and
dispersion are both estimated from the regions themselves, under the
assumption that most are *not* enriched. A set pre-filtered to expected hits
makes the normalization absorb the real differences, and the test calls fewer
regions, down to none (measured in `references/choosing-parameters.md` §Null majority).

| Region set | Build |
|---|---|
| union of all conditions' peaks, merged | `zcat -f *.narrowPeak* \| cut -f1-3 \| sort -k1,1 -k2,2n \| bedtools merge > background.bed` |
| ENCODE SCREEN cCREs for the assembly | the registry BED from https://screen.wenglab.org/downloads; no peak calling needed |
| genome-wide non-overlapping windows | `bedtools makewindows -g genome.sizes -w 500 > windows.bed` |
| promoters / gene bodies | fine if it is the full set, not a hand-picked subset |

`bedtools merge` chains overlapping peaks into wide regions, which dilutes a
narrow peak. For fixed-width regions, center a window on each summit
(narrowPeak column 10) before merging:

```bash
zcat -f *.narrowPeak* | awk 'BEGIN{OFS="\t"} {s=$2+$10-250; if (s<0) s=0; print $1, s, s+500}' \
	| sort -k1,1 -k2,2n | bedtools merge | awk 'BEGIN{OFS="\t"} {print $0, "region" NR}' > background.bed
```

The last `awk` adds a `name` column, which `extract` and `enrich` pass through.
Drop contigs the bigWigs lack (`chrUn_*`, `*_random`, alt haplotypes) and
blacklisted regions (`bedtools subtract -a background.bed -b blacklist.bed`)
before extracting.

**Non-overlapping.** BH q-values assume independent (or positively dependent)
tests. `enrich` warns `N% of adjacent regions overlap ... reported q-values will
be optimistic` when more than 1% of sorted adjacent pairs overlap. Merge peaks
(`bedtools merge`) or use a window step equal to the window size.

Regions may differ in length; each test compares conditions within one
region.

BED details (0-based half-open, columns passed through): `references/extract.md`.

## Chromosome naming and assembly

BED `chr1` against a bigWig with `1` (Ensembl), or the reverse, gives zeros and
a warning, not an error. Check the BED and **every** bigWig before extracting:
one mismatched track among many stays under the 95%-zeros warning. At K = 3 it
then produces a spurious call in every region it zeroes; at larger K it
produces no flag and no call, so nothing downstream shows it:

```bash
cut -f1 background.bed | sort -u | head
for bw in *.bw; do
	python -c "import sys, pyBigWig; print(sys.argv[1], list(pyBigWig.open(sys.argv[1]).chroms())[:5])" "$bw"
done
```

Ensembl → UCSC names for a BED (`MT` → `chrM`; skips `track`/`browser`/`#`
lines and drops unplaced scaffolds such as `KI270728.1`, whose UCSC names
differ):

```bash
awk 'BEGIN{OFS="\t"} /^(track|browser|#)/{next} $1 ~ /^(KI|GL)/{next} {$1 = ($1=="MT") ? "chrM" : "chr"$1; print}' \
	regions.ensembl.bed > regions.ucsc.bed
```

A BED on hg19 against an hg38 bigWig gives wrong values everywhere, and the
`out_of_bounds` warning only where a region runs past a chromosome end. Confirm
the assembly from file provenance.

## Stranded tracks

Each `-c` column is one condition. Passing `x.+.bw` and `x.-.bw` as two
conditions tests strand against strand. Extract both strands, then add them
into one column per condition:

```python
import pandas as pd

df = pd.read_csv("signals.tsv", sep="\t", comment="#", dtype={"chrom": str})
for c in ["liver", "heart"]:
	df[c] = df.pop(f"{c}_plus") + df.pop(f"{c}_minus")
df.to_csv("signals_merged.tsv", sep="\t", index=False)
```

with `extract ... -n liver_plus liver_minus heart_plus heart_minus`. `+`
propagates NaN; `DataFrame.sum(axis=1)` treats NaN as 0 unless given
`min_count`. Writing through pandas drops the `# fertilizer-extract stat=sum`
header; `enrich` then skips the stat check, which is correct because sums of
sums are still sums.

## Replicates

`fertilizer` has no notion of replicates. With replicates, DESeq2 or edgeR on
the replicate counts estimates per-region dispersion from within-condition
variance and is the better test. If the user still wants the one-vs-rest
framing, sum the replicates into one column per condition; the replicate
variance is then discarded:

```python
for c in ["CTRL", "TRT"]:
	reps = [f"{c}_rep1", f"{c}_rep2"]
	df[c] = df[reps].sum(axis=1, min_count=len(reps))    # NaN if any replicate is missing
	df = df.drop(columns=reps)
```

Never pass replicates as separate `-c` conditions: the test would call regions
where one replicate is high.

## A count matrix from another tool

`enrich` reads any tab-separated file with a header row and numeric columns
named by `-c`. Lines starting with `#` are skipped as comments (anywhere in a
line, `#` truncates it). A first line `# ... stat=<x>` with `x != sum` triggers
the refusal; any other `#` line is ignored.

| Source | Note |
|---|---|
| featureCounts | works; its `# Program:featureCounts` line is skipped. Pass the sample columns to `-c` |
| deepTools `multiBigwigSummary BED-file --outRawCounts` | do not use. Its values are per-base means, not sums, and its header line starts with `#'chr'`, so it is skipped and the first data row becomes the header. Run `fertilizer extract -s sum` on the same bigWigs instead |
| hand-built TSV | blanks become NaN, which `enrich` does not reject: that region silently gets `p_value = 1`. Fill or drop NaN first |

Coordinate columns must be named `chrom`, `start`, `end` for the overlap check
to run; other names are passed through untouched.
