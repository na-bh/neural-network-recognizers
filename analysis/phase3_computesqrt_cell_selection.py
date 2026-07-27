"""Compute-sqrt Phase 3 (bounded) pilot: cell selection. Pulls trained-
architecture accuracies from flare_a2_compute_sqrt.json (existing
checkpoints, no training here -- the file already exists), selects primary
(best) + secondary (next-best) seed per architecture, and checks for
single-solver bottlenecks (matching phase3_dyck/missdup/stackmanip_cell_
selection.py's discipline exactly).

PYTHONPATH=src:analysis python analysis/phase3_computesqrt_cell_selection.py
"""

import json
from pathlib import Path

RESULTS = Path("analysis_outputs/final_results")
TASK = "compute-sqrt"
COMPARABILITY_THRESHOLD = 0.15
NEAR_CHANCE_THRESHOLD = 0.60


def select_cells():
    flare_a2 = json.loads((RESULTS / "flare_a2_compute_sqrt.json").read_text())
    selected = flare_a2["selected_seeds"]
    acc_table = flare_a2["accuracy_table"]

    cells = {}
    for arch, primary_seed in selected.items():
        primary_acc = acc_table[arch][str(primary_seed)]
        ranked = sorted(acc_table[arch].items(), key=lambda kv: -kv[1])
        secondary_seed, secondary_acc = next((s, a) for s, a in ranked if s != str(primary_seed))
        gap = primary_acc - secondary_acc
        comparable = gap <= COMPARABILITY_THRESHOLD
        all_below_near_chance = all(a <= NEAR_CHANCE_THRESHOLD for _, a in ranked)
        n_above_near_chance = sum(1 for _, a in ranked if a > NEAR_CHANCE_THRESHOLD)
        cells[arch] = {
            "primary_seed": int(primary_seed), "primary_accuracy": primary_acc,
            "secondary_seed": int(secondary_seed), "secondary_accuracy": secondary_acc,
            "accuracy_gap": gap, "comparable": comparable,
            "all_seed_accuracies_ranked": ranked,
            "n_seeds_above_near_chance_threshold": n_above_near_chance,
            "n_seeds_total": len(ranked),
            "architecture_wide_near_chance": all_below_near_chance,
        }
    return cells, flare_a2.get("hard_vs_scrambled_negative_accuracy", {})


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cells, hard_vs_scrambled = select_cells()
    print("=== cell selection ===", flush=True)
    print(json.dumps(cells, indent=2, default=str))

    bottleneck_summary = {}
    for arch, c in cells.items():
        if c["architecture_wide_near_chance"]:
            bottleneck_summary[arch] = (
                f"ARCHITECTURE-WIDE NEAR-CHANCE COLLAPSE -- all {c['n_seeds_total']} seeds <= "
                f"{NEAR_CHANCE_THRESHOLD}, including the 'best' seed ({c['primary_accuracy']:.3f})."
            )
        elif not c["comparable"]:
            bottleneck_summary[arch] = (
                f"SINGLE-SOLVER BOTTLENECK -- primary seed {c['primary_seed']} "
                f"({c['primary_accuracy']:.3f}) exceeds secondary {c['secondary_seed']} "
                f"({c['secondary_accuracy']:.3f}) by {c['accuracy_gap']:.3f} (> {COMPARABILITY_THRESHOLD} "
                f"threshold)."
            )
        else:
            bottleneck_summary[arch] = (
                f"NO SEED-LEVEL BOTTLENECK -- primary/secondary seeds comparable (gap={c['accuracy_gap']:.3f})."
            )
    print("\n=== per-seed bottleneck check ===", flush=True)
    print(json.dumps(bottleneck_summary, indent=2))

    print("\n=== hard-vs-scrambled accuracy (reported for context; interpret with the task audit's "
          "confound-severity finding -- 76.9% of 'hard' is trivial, less severe than missing-"
          "duplicate-string/stack-manipulation's 87%) ===", flush=True)
    print(json.dumps(hard_vs_scrambled, indent=2, default=str))

    out = {
        "task": TASK,
        "description": (
            "Cell selection for compute-sqrt's bounded Phase 3 patching pilot: primary (best) + "
            "secondary (next-best) seed per architecture, pulled from the existing flare_a2_"
            "compute_sqrt.json (no new training). Checks for per-seed single-solver bottlenecks AND "
            "architecture-wide near-chance collapse."
        ),
        "comparability_threshold": COMPARABILITY_THRESHOLD,
        "near_chance_threshold": NEAR_CHANCE_THRESHOLD,
        "cell_selection": cells,
        "per_seed_bottleneck_check": bottleneck_summary,
        "hard_vs_scrambled_negative_accuracy_partially_confounded_see_task_audit": hard_vs_scrambled,
        "final_cells": [
            {"arch": arch, "seed": c["primary_seed"], "role": "primary"} for arch, c in cells.items()
        ] + [
            {"arch": arch, "seed": c["secondary_seed"], "role": "secondary",
             "comparable": c["comparable"]} for arch, c in cells.items()
        ],
    }
    out_path = RESULTS / "phase3_computesqrt_cell_selection.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
