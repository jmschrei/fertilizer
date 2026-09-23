"""Generate small synthetic bigWig + BED files for the README demo.

Run once from the repository root:

    python examples/make_demo_data.py

Produces examples/regions.bed, examples/A.bw, examples/B.bw, examples/C.bw.
A and B are flat backgrounds across all regions; C has elevated signal in a
small subset of regions, simulating the "fertile ground" pattern the package
is designed to detect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyBigWig

HERE = Path(__file__).resolve().parent
CHROM_LEN = 100_000
N_REGIONS = 200
REGION_LEN = 200
SEED = 0


def make_regions(rng: np.random.Generator) -> list[tuple[str, int, int]]:
	# Evenly spaced, non-overlapping windows with a small deterministic gap.
	spacing = CHROM_LEN // N_REGIONS
	if spacing <= REGION_LEN:
		raise ValueError("regions would overlap; raise CHROM_LEN")
	starts = np.arange(N_REGIONS) * spacing
	# tiny jitter that keeps non-overlap
	max_jitter = (spacing - REGION_LEN) // 2
	if max_jitter > 0:
		starts = starts + rng.integers(0, max_jitter, size=N_REGIONS)
	return [("chr1", int(s), int(s) + REGION_LEN) for s in starts]


def make_bigwig(
    path: Path, regions: list[tuple[str, int, int]], values: np.ndarray
) -> None:
	bw = pyBigWig.open(str(path), "w")
	bw.addHeader([("chr1", CHROM_LEN)])
	chroms = [r[0] for r in regions]
	starts = [r[1] for r in regions]
	ends = [r[2] for r in regions]
	bw.addEntries(chroms, starts, ends=ends, values=[float(v) for v in values])
	bw.close()


def main() -> None:
	rng = np.random.default_rng(SEED)
	regions = make_regions(rng)

	a = rng.poisson(20.0, size=N_REGIONS).astype(float)
	b = rng.poisson(20.0, size=N_REGIONS).astype(float)
	c = rng.poisson(20.0, size=N_REGIONS).astype(float)

	enriched = rng.choice(N_REGIONS, size=10, replace=False)
	c[enriched] = rng.poisson(100.0, size=enriched.size)

	make_bigwig(HERE / "A.bw", regions, a)
	make_bigwig(HERE / "B.bw", regions, b)
	make_bigwig(HERE / "C.bw", regions, c)

	bed = HERE / "regions.bed"
	with bed.open("w") as f:
		for chrom, start, end in regions:
			f.write(f"{chrom}\t{start}\t{end}\n")

	print(f"wrote {bed}")
	print(f"wrote {HERE / 'A.bw'}, {HERE / 'B.bw'}, {HERE / 'C.bw'}")
	print(f"  C is enriched at {len(enriched)} regions: {sorted(enriched.tolist())}")


if __name__ == "__main__":
	main()
