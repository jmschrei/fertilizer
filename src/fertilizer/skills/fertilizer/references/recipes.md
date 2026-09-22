# Recipes: user question → commands

Every recipe assumes `signals.tsv` from `extract -s sum` and a background-matched
region set (`references/inputs.md`). Load results with:

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

## 2. Top N per condition, or a ranking of every region

Keep all rows, then rank. Sort by `q_value` for confidence, by `effect_size`
for magnitude; report which.

```bash
fertilizer enrich -i signals.tsv -c liver heart brain kidney -o all.tsv --q-threshold 1.0
```

```python
df = load("all.tsv")
top = (df[df["q_value"] <= 0.05]
	.sort_values(["enriched_condition", "effect_size"], ascending=[True, False])
	.groupby("enriched_condition").head(100))
```

## 3. Active in exactly one condition

`--background-rank 2` answers this directly but is underpowered
(`references/choosing-parameters.md`). The alternative: run the default rank 3
and require the top to clear the runner-up by a margin on the normalized scale.
The same columns name the runner-up, which answers "which two conditions share
this region". At K = 3, run `enrich` with `--q-threshold 0.025`
(`references/choosing-parameters.md` §K = 3).

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
df["runner_up"] = np.asarray(conds)[order[:, 1]]
df["log2_top_vs_second"] = np.log2(srt[:, 0] + pc) - np.log2(srt[:, 1] + pc)
df["log2_second_vs_third"] = np.log2(srt[:, 1] + pc) - np.log2(srt[:, 2] + pc)

only_liver = df[(df["enriched_condition"] == "liver") & (df["log2_top_vs_second"] >= 1)]
liver_heart = df[df["enriched_condition"].isin(["liver", "heart"])
	& df["runner_up"].isin(["liver", "heart"]) & (df["log2_second_vs_third"] >= 1)]
```

The margin (here 2×) is the user's choice, not a calibrated threshold, and the
q-values are still those of the rank-3 test. Say both when reporting.

## 4. Higher in A than in B

A K = 2 run on just those two columns. `enriched_condition` gives the
direction; filter it.

```bash
fertilizer enrich -i signals.tsv -c liver heart -o liver_vs_heart.tsv
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

An intersection of K − 1 tests has low power: in simulation (K = 4, 4×
depletion) it recovered 42 of 250 depleted regions with no false calls, and
10 of 250 at K = 8. It does not control FDR across the intersection;
present it as a ranking by `max_pair_q`. "Lower than everywhere" is not
"closed": add an absolute cutoff on X's value if the user means closed.

## 6. How robust are the calls?

Re-run at fixed dispersions around the fitted one (from `enrich.log`) and
report the overlap.

```bash
for a in 0.05 0.10 0.20; do
	fertilizer enrich -i signals.tsv -c liver heart brain kidney \
		-o enrich_a$a.tsv --dispersion $a 2> /dev/null
done
```

Regions called at every α are the robust set.

## 7. Starting regions for regulatory design (ledidi)

The package's premise (the Fertile Ground Hypothesis) is that regions where
the target condition already stands out need the fewest edits to reach
condition-specific activity. Run recipe 1
for the target condition, then drop small effects, cap the count, and write a
BED:

```python
df = load("enrichment.tsv")
t = df[(df["enriched_condition"] == "liver") & (df["effect_size"] >= 1)]
t = t.sort_values("q_value").head(200).sort_values(["chrom", "start"])
t[["chrom", "start", "end", "name"]].to_csv("templates.bed", sep="\t", header=False, index=False)
```

`name` exists only when the BED had a fourth column (`references/inputs.md`
§Region set shows how to add one after merging). Within a Cherimoya +
ledidi design workflow, the `cherimoya-ledidi-design` skill (step 2c) owns
where these files go and how they feed the design.

## 8. Normalization the user already trusts

Spike-in or library-size factors, one per `-c` entry in the same order:

```bash
fertilizer enrich -i signals.tsv -c liver heart brain kidney -o out.tsv \
	--size-factors 1.2 0.9 1.0 0.95
```

`--size-factors 1 1 1 1` means "already normalized".

## 9. Keep peak names or extra BED columns

They pass through both commands. A BED with a fourth column arrives in
`enrichment.tsv` as `name`; join back on it.
