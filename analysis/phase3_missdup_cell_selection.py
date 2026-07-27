"""Missing-duplicate-string Phase 3 (bounded) pilot: cell selection. Pulls
trained-architecture accuracies from flare_a2_missing_duplicate_string.json
(existing checkpoints, no training here -- the file already exists from the
modular-arithmetic-simple-protocol A2 audit), selects primary (best) +
secondary (next-best) seed per architecture, and checks for single-solver
bottlenecks (matching phase3_dyck_cell_selection.py's discipline exactly).

ADDITIONAL FLAG (beyond the standard per-seed bottleneck check): flare_a2's
own hard_vs_scrambled_negative_accuracy already reveals an ARCHITECTURE-WIDE
split directly relevant to this experiment's hypothesis -- RNN/LSTM solve
hard negatives well (~0.91) while Transformer/Mamba do not (~0.25-0.28)
despite all four solving scrambled negatives easily (~0.90-0.90). Surfaced
here explicitly as a predicted-route hint (not a substitute for the
per-seed bottleneck check): architectures with a large hard-vs-scrambled
gap are predicted to show large Condition 2 (aggregate-route) effects and
small/null Condition 3 (position) effects; architectures with a small gap
are predicted to show the reverse.

PYTHONPATH=src:analysis python analysis/phase3_missdup_cell_selection.py
"""

import json
from pathlib import Path

RESULTS = Path("analysis_outputs/final_results")
TASK = "missing-duplicate-string"
COMPARABILITY_THRESHOLD = 0.15
NEAR_CHANCE_THRESHOLD = 0.60
HARD_VS_SCRAMBLED_GAP_FLAG_THRESHOLD = 0.30


def select_cells():
    flare_a2 = json.loads((RESULTS / "flare_a2_missing_duplicate_string.json").read_text())
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

    # architecture-wide hard-vs-scrambled route hint
    route_hints = {}
    for arch, hv in hard_vs_scrambled.items():
        gap = hv["scrambled_negative_accuracy"] - hv["hard_negative_accuracy"]
        flagged = gap >= HARD_VS_SCRAMBLED_GAP_FLAG_THRESHOLD
        route_hints[arch] = {
            "hard_negative_accuracy": hv["hard_negative_accuracy"],
            "scrambled_negative_accuracy": hv["scrambled_negative_accuracy"],
            "scrambled_minus_hard_gap": gap,
            "flagged_aggregate_route_suspect": flagged,
            "predicted_condition2_vs_condition3": (
                "predicted LARGE Condition 2 (aggregate-route) effect, SMALL/NULL Condition 3 "
                "(position) effect -- solves scrambled negatives via shallow cues but fails hard "
                "negatives requiring genuine positional comparison" if flagged else
                "predicted SMALL Condition 2 effect, LARGE Condition 3 effect -- comparable "
                "hard/scrambled accuracy suggests genuine positional tracking, not aggregate-only"
            ),
        }
    print("\n=== architecture-wide hard-vs-scrambled route hint (from existing flare_a2 data) ===", flush=True)
    print(json.dumps(route_hints, indent=2, default=str))

    out = {
        "task": TASK,
        "description": (
            "Cell selection for missing-duplicate-string's bounded Phase 3 patching pilot: primary "
            "(best) + secondary (next-best) seed per architecture, pulled from the existing "
            "flare_a2_missing_duplicate_string.json (modular-arithmetic-simple protocol, no new "
            "training). Checks for per-seed single-solver bottlenecks AND architecture-wide "
            "near-chance collapse, plus surfaces the pre-existing hard-vs-scrambled-negative "
            "accuracy split as a predicted-route hint for interpreting the upcoming Condition 2/3 "
            "patching runs."
        ),
        "comparability_threshold": COMPARABILITY_THRESHOLD,
        "near_chance_threshold": NEAR_CHANCE_THRESHOLD,
        "hard_vs_scrambled_gap_flag_threshold": HARD_VS_SCRAMBLED_GAP_FLAG_THRESHOLD,
        "cell_selection": cells,
        "per_seed_bottleneck_check": bottleneck_summary,
        "architecture_wide_route_hint": route_hints,
        "final_cells": [
            {"arch": arch, "seed": c["primary_seed"], "role": "primary"} for arch, c in cells.items()
        ] + [
            {"arch": arch, "seed": c["secondary_seed"], "role": "secondary",
             "comparable": c["comparable"]} for arch, c in cells.items()
        ],
    }
    out_path = RESULTS / "phase3_missdup_cell_selection.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
