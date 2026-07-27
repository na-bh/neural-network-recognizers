"""Dyck-2-3 Phase 3 (bounded) pilot: cell selection. Pulls trained-
architecture accuracies from flare_a2_dyck_2_3.json (existing checkpoints,
no training here), selects primary (best) + secondary (next-best) seed per
architecture, and checks for single-solver bottlenecks (matching the
marked-copy LSTM / odds-first Transformer / bucket-sort LSTM pattern found
elsewhere in the pilot) -- AND a distinct pattern this task's own data
raises: an architecture where EVERY seed is weak (near chance), not just a
gap between one strong seed and the rest.

PYTHONPATH=src:analysis python analysis/phase3_dyck_cell_selection.py
"""

import json
from pathlib import Path

RESULTS = Path("analysis_outputs/final_results")
TASK = "dyck-2-3"
COMPARABILITY_THRESHOLD = 0.15
NEAR_CHANCE_THRESHOLD = 0.60  # accuracy at/below this is "not really solving" for a 2-class task


def select_cells():
    flare_a2 = json.loads((RESULTS / "flare_a2_dyck_2_3.json").read_text())
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
    return cells


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cells = select_cells()
    print("=== cell selection ===")
    print(json.dumps(cells, indent=2, default=str))

    bottleneck_summary = {}
    for arch, c in cells.items():
        if c["architecture_wide_near_chance"]:
            bottleneck_summary[arch] = (
                f"ARCHITECTURE-WIDE NEAR-CHANCE COLLAPSE -- all {c['n_seeds_total']} seeds <= "
                f"{NEAR_CHANCE_THRESHOLD}, including the 'best' seed ({c['primary_accuracy']:.3f}). "
                f"Distinct from a single-solver bottleneck (one great seed, rest bad) -- here NO "
                f"seed genuinely solves the task."
            )
        elif not c["comparable"]:
            bottleneck_summary[arch] = (
                f"SINGLE-SOLVER BOTTLENECK -- primary seed {c['primary_seed']} "
                f"({c['primary_accuracy']:.3f}) exceeds secondary {c['secondary_seed']} "
                f"({c['secondary_accuracy']:.3f}) by {c['accuracy_gap']:.3f} (> {COMPARABILITY_THRESHOLD} "
                f"threshold), matching the marked-copy LSTM / odds-first Transformer / bucket-sort "
                f"LSTM pattern found elsewhere in the pilot."
            )
        else:
            bottleneck_summary[arch] = (
                f"NO BOTTLENECK -- primary/secondary seeds comparable (gap={c['accuracy_gap']:.3f})."
            )
    print("\n=== bottleneck check ===")
    print(json.dumps(bottleneck_summary, indent=2))

    out = {
        "task": TASK,
        "description": "Cell selection for Dyck-2-3's bounded Phase 3 patching pilot: primary "
                       "(best) + secondary (next-best) seed per architecture, pulled from "
                       "flare_a2_dyck_2_3.json (no new training). Checks for single-solver "
                       "bottlenecks AND architecture-wide near-chance collapse (a DISTINCT pattern "
                       "from a single-solver bottleneck: every seed weak, not just an outlier gap).",
        "comparability_threshold": COMPARABILITY_THRESHOLD,
        "near_chance_threshold": NEAR_CHANCE_THRESHOLD,
        "cell_selection": cells,
        "bottleneck_check": bottleneck_summary,
        "final_cells": [
            {"arch": arch, "seed": c["primary_seed"], "role": "primary"} for arch, c in cells.items()
        ] + [
            {"arch": arch, "seed": c["secondary_seed"], "role": "secondary",
             "comparable": c["comparable"]} for arch, c in cells.items()
        ],
    }
    out_path = RESULTS / "phase3_dyck_cell_selection.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
