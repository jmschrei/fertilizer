# Quickstart

## Install

```bash
uv pip install fertilizer-genomics                       # PyPI
uv pip install git+https://github.com/jmschrei/fertilizer.git
fertilizer --version
```

`pysam` (BAM/CRAM input) ships wheels for Linux and macOS. `pyBigWig` builds
against `libcurl`/`libssl`. If the install fails building it:
`sudo apt-get install libcurl4-openssl-dev libssl-dev zlib1g-dev` (Debian/Ubuntu)
or `brew install curl openssl` (macOS).

Install this skill into Claude Code:

```bash
fertilizer install-skill            # copies into ~/.claude/skills/fertilizer
fertilizer install-skill --force    # after upgrading fertilizer: replaces the old copy
```

| Flag | Effect |
|---|---|
| `-d`, `--directory DIR` | skills directory to install into (default `~/.claude/skills`) |
| `--symlink` | symlink the packaged copy instead of copying; follows an editable install |
| `-f`, `--force` | delete an existing `<DIR>/fertilizer` and reinstall |

Without `--force`, an existing install exits 2 with
`fertilizer: error: ... already exists. Re-run with --force to overwrite it.`

## Demo (synthetic data, runs in seconds)

`examples/make_demo_data.py` is in the GitHub repository, not the wheel.

```bash
git clone https://github.com/jmschrei/fertilizer.git && cd fertilizer
python examples/make_demo_data.py      # 200 regions on chr1; C boosted at 10 of them
fertilizer extract -w examples/A.bw examples/B.bw examples/C.bw \
	-b examples/regions.bed -o examples/signals.tsv -s sum -j 4
fertilizer enrich -i examples/signals.tsv -c A B C -o examples/enrichment.tsv
```

Expected: stderr ends `kept 11 / 200 loci (q <= 0.05)`; ten rows have
`enriched_condition == C` (the boosted regions) and one is `B`, a false
positive at this FDR. A K = 3 calibration warning is also printed; see
`references/choosing-parameters.md` §K = 3.

## The pipeline on real data

```bash
# 1. region-by-track matrix of summed signal
fertilizer extract -w liver.bw heart.bw brain.bw kidney.bw \
	-b background.bed -o signals.tsv -s sum -j 8 2> extract.log

# 2. per-region enrichment test; stderr carries the diagnostics
fertilizer enrich -i signals.tsv -c liver heart brain kidney \
	-o enrichment.tsv 2> enrich.log
```

From BAMs, CRAMs or fragment files, step 1 counts instead of summing (no `-s`):

```bash
fertilizer extract -a liver.bam heart.bam brain.bam kidney.bam -b background.bed -o signals.tsv -j 8
fertilizer extract -f fragments.tsv.gz -g cells.tsv --group-column cluster -b background.bed -o signals.tsv -j 8
```

`-c` takes the column names `extract` wrote: each bigWig's filename stem
(`liver.bw` → `liver`) unless `-n` overrode them. Before running, settle:

1. Which region set (`references/inputs.md` §Region set).
2. Whether a region shared by two conditions is a hit (`--background-rank`,
   `references/choosing-parameters.md`).
3. Whether the user wants every row (`--q-threshold 1.0`) or only calls.

Then read the result:

```python
import pandas as pd

df = pd.read_csv("enrichment.tsv", sep="\t", dtype={"chrom": str})
df = df[~df["lrt_zero_dominated"] & ~df["lrt_convergence_failed"]]
df = df.sort_values("q_value")
print(df["enriched_condition"].value_counts())
```
