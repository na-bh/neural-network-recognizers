"""Patch: recompute the long_L400 counterfactual cell for the 3 tasks whose
pair generator overflowed numpy int64 bounds at large bit-widths (binary
math tasks). Reuses each task's already-selected seeds (no re-selection),
updates the saved JSON in place.

PYTHONPATH=src:analysis python analysis/flare_a2_patch_overflow.py
"""
import json
from pathlib import Path

import numpy as np

import flare_a2_counterfactuals as fc
from flare_a2_worker import get_harness

RESULTS = Path("analysis_outputs/final_results")
TASKS = ["binary-addition", "binary-multiplication", "compute-sqrt"]
N_PAIRS = 100


def main():
    for task in TASKS:
        path = RESULTS / f"flare_a2_{task.replace('-', '_')}.json"
        d = json.loads(path.read_text())
        print(f"=== {task} ===")
        for arch, seed in d["selected_seeds"].items():
            h = get_harness(task, arch, seed)
            rng = np.random.default_rng(3000 + 400)
            pairs = fc.TASK_MAKERS[task](N_PAIRS, 400, rng)
            diffs, accepts_corrupt = [], []
            for p in pairs:
                cl, co = h.logit_diff(p["clean"]), h.logit_diff(p["corrupt"])
                diffs.append(abs(cl - co))
                accepts_corrupt.append(co > 0)
            row = {
                "length": len(pairs[0]["clean"]), "n_pairs": len(pairs),
                "mean_abs_delta_logit": float(np.mean(diffs)),
                "median_abs_delta_logit": float(np.median(diffs)),
                "frac_corrupt_accepted": float(np.mean(accepts_corrupt)),
            }
            d["counterfactual_logit_difference"][arch]["long_L400"] = row
            print(f"  {arch:12s} long_L400 mean|delta_logit|={row['mean_abs_delta_logit']:.4f} "
                  f"frac_corrupt_accepted={row['frac_corrupt_accepted']:.2f}")
        path.write_text(json.dumps(d, indent=2))
        print(f"  Updated {path}")


if __name__ == "__main__":
    main()
