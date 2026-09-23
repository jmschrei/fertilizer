# Recipes: user question → commands

Every recipe assumes `signals.tsv` from `extract -s sum` (or counts from `extract -a`/`-f`) and a background-matched
region set (`references/inputs.md`). The examples have K = 4, where the default
`--background-rank 3` is also the highest safe rank. At K ≥ 5 pick the rank
from `references/choosing-parameters.md` first. Load results with:

```python
import pandas as pd

def load(path):
	df = pd.read_csv(path, sep="\t", dtype={"chrom": str})
	return df[~df["lrt_zero_dominated"] & ~df["lrt_convergence_failed"]]
```

## 1. Regions specific to condition X

First ask whether a region high in X and one other condition counts. If yes,
this recipe. If no ("only in X"), recipe 3.

Test all conditions together, then filter. Running only X against a subset
changes the size factors and the question.

```bash
fertilizer enrich -i signals.tsv -c liver heart brain kidney -o enrichment.tsv 2> enrich.log
```

```python
df = load("enrichment.tsv")
liver = df[df["enriched_condition"] == "liver"]
liver.sort_values(["chrom", "start"])[["chrom", "start", "end"]].to_csv(
	"liver_specific.bed", sep="\t", header=False, index=False)
```

At a rank above 3, "at most one other condition as high" is the recipe-3
margin on `log2_top_vs_third`.

## 2. Top N per condition, or a ranking of every region

Keep all rows (also what plots need), then rank. Sort by `q_value` for
confidence, by `effect_size` for magnitude; report which.

```bash
fertilizer enrich -i signals.tsv -c liver heart brain kidney -o all.tsv --q-threshold 1.0
```

```python
df = load("all.tsv")
top = (df[df["q_value"] <= 0.05]
	.sort_values(["enriched_condition", "effect_size"], ascending=[True, False])
	.groupby("enriched_condition").head(100))
```

## 3. Only in X, or shared by A and B

Run at the highest safe rank, then require margins between the sorted
normalized values. `--background-rank 2` answers "only in X" directly but has
far less power (`references/choosing-parameters.md`). At K = 3, add
`--q-threshold 0.025` (§K = 3 there).

```bash
# highest safe rank: 3 at K = 4; K − 1 up to K = 8; K − 2 at K = 12 and 20
fertilizer enrich -i signals.tsv -c liver heart brain kidney -o enrichment.tsv \
	--background-rank 3 2> enrich.log
```

```python
import numpy as np
from fertilizer.enrichment import size_factors

conds = ["liver", "heart", "brain", "kidney"]
raw = pd.read_csv("signals.tsv", sep="\t", comment="#", dtype={"chrom": str})
sf = size_factors(raw[conds].to_numpy(float))    # identical to the stderr values

df = load("enrichment.tsv")
norm = df[conds].to_numpy(float) / sf
order = np.argsort(-norm, axis=1)
srt = np.take_along_axis(norm, order, axis=1)
pc = 0.5                                          # match enrich --pseudocount
lg = np.log2(srt + pc)
df["runner_up"] = np.asarray(conds)[order[:, 1]]
df["log2_top_vs_second"] = lg[:, 0] - lg[:, 1]
df["log2_top_vs_third"] = lg[:, 0] - lg[:, 2]
df["log2_second_vs_third"] = lg[:, 1] - lg[:, 2]

only_liver = df[(df["enriched_condition"] == "liver") & (df["log2_top_vs_second"] >= 1)]
liver_heart = df[df["enriched_condition"].isin(["liver", "heart"])
	& df["runner_up"].isin(["liver", "heart"]) & (df["log2_second_vs_third"] >= 1)]
```

`size_factors` needs the `-c` columns of the same run. With `--size-factors`,
use those values instead. The margin (here 2×) is the user's choice, not a
calibrated threshold, and the q-values are those of the rank-r test. Say both
when reporting.

## 4. Higher in A than in B

A K = 2 run on just those two columns; `enriched_condition` gives the
direction. To keep the normalization of the full run, pass its
`size factor liver:` and `size factor heart:` values from `enrich.log`.

```bash
fertilizer enrich -i signals.tsv -c liver heart -o liver_vs_heart.tsv \
	--size-factors 1.01 0.99
```

```python
liver_up = load("liver_vs_heart.tsv").query("enriched_condition == 'liver'")
```

## 5. Lower in X than in every other condition

`enrich` never calls depletion. Intersect K = 2 runs of X against each other
condition, keeping regions where the other condition won every time. Pass the
size factors from all K conditions so every pair shares one normalization:

```python
import numpy as np
from fertilizer.enrichment import enrichment_analysis, size_factors

conds = ["liver", "heart", "brain", "kidney"]
x = "liver"
raw = pd.read_csv("signals.tsv", sep="\t", comment="#", dtype={"chrom": str})
sf = dict(zip(conds, size_factors(raw[conds].to_numpy(float))))

depleted = np.ones(len(raw), dtype=bool)
worst_q = np.zeros(len(raw))
for other in conds:
	if other == x:
		continue
	res = enrichment_analysis(raw[[x, other]].to_numpy(float),
		size_factors_override=np.array([sf[x], sf[other]]))
	depleted &= (res.q_value <= 0.05) & (res.enriched_condition_idx == 1) & ~res.lrt_zero_dominated
	worst_q = np.maximum(worst_q, res.q_value)
liver_depleted = raw[depleted].assign(max_pair_q=worst_q[depleted]).sort_values("max_pair_q")
```

Low power: in simulation (4× depletion) it recovered 42 of 250 depleted regions
at K = 4 and 10 of 250 at K = 8, with no false calls. It does not control FDR
across the intersection; present it as a ranking by `max_pair_q`. "Lower than
everywhere" is not "closed": add an absolute cutoff on X's value if the user
means closed.

## 6. How robust are the calls?

Re-run at fixed dispersions spanning the fitted α in `enrich.log` (for example
half, equal, double), and keep regions called at every value.

```bash
for a in 0.03 0.06 0.12; do
	fertilizer enrich -i signals.tsv -c liver heart brain kidney \
		-o enrich_a$a.tsv --dispersion $a 2> /dev/null
done
```

```python
from functools import reduce

keys = [set(zip(d["chrom"], d["start"], d["end"]))
	for d in (load(f"enrich_a{a}.tsv") for a in ("0.03", "0.06", "0.12"))]
robust = reduce(set.intersection, keys)
print([len(k) for k in keys], "robust:", len(robust))
```

## 7. Starting regions for regulatory design (ledidi)

The package's premise (the Fertile Ground Hypothesis) is that regions where
the target condition already stands out need the fewest edits to reach
condition-specific activity. Take the target's calls from recipe 1, or recipe
3 when templates must be silent in every other condition, then write a BED.
The effect-size cutoff and the count below are placeholders; set them from the
design budget.

```python
df = load("enrichment.tsv")
t = df[(df["enriched_condition"] == "liver") & (df["effect_size"] >= 1)]
t = t.sort_values("q_value").head(200).sort_values(["chrom", "start"])
t[["chrom", "start", "end", "name"]].to_csv("templates.bed", sep="\t", header=False, index=False)
```

`name` exists only when the BED had a fourth column (`references/inputs.md`
§Region set shows how to add one after merging). Within a Cherimoya + ledidi
design workflow, the `cherimoya-ledidi-design` skill (step 2c) owns where these
files go and how they feed the design.

## 8. Normalization the user already trusts

First ask whether the factors are already applied in the bigWigs; if so, pass
`1 1 1 1`. Otherwise one size factor per `-c` entry, same order. fertilizer
**divides** each column by its factor, so a factor is proportional to depth: a
library sequenced 20% deeper gets 1.2. Spike-in scale factors meant to be
*multiplied* into the signal must be passed as their reciprocals.

**Rescale to geometric mean 1.** The overall scale of the factors changes the
dispersion fit: in simulation, multiplying every factor by 20 raised α from
0.055 to 0.93 and dropped the calls from 178 to 0.

```python
import numpy as np
spike = np.array([0.05, 0.08, 0.04, 0.06])    # multiplicative factors from the pipeline
sf = 1 / spike
sf = sf / np.exp(np.log(sf).mean())
print(" ".join(f"{v:.4f}" for v in sf))       # paste into --size-factors
```

```bash
fertilizer enrich -i signals.tsv -c liver heart brain kidney -o out.tsv \
	--size-factors 1.2 0.9 1.0 0.95
```

The example values are already in the divide-by convention.
