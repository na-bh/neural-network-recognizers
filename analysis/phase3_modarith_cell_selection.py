"""Modular-arithmetic-simple Phase 3 (bounded) pilot: cell selection. Pulls
trained-architecture accuracies from flare_a2_modular_arithmetic_simple.json
(built directly from data/models/models eval/test.json files under the
rec+ns/validation-short convention -- this flare_a2 file did not previously
exist for this task), selects primary (best) + secondary (next-best) seed
per architecture, and checks for single-solver bottlenecks.

PYTHONPATH=src:analysis python analysis/phase3_modarith_cell_selection.py
"""

import json
from pathlib import Path

RESULTS = Path("analysis_outputs/final_results")
TASK = "modular-arithmetic-simple"
COMPARABILITY_THRESHOLD = 0.15


def select_cells():
    flare_a2 = json.loads((RESULTS / "flare_a2_modular_arithmetic_simple.json").read_text())
    selected = flare_a2["selected_seeds"]
    acc_table = flare_a2["accuracy_table"]

    cells = {}
    for arch, primary_seed in selected.items():
        primary_acc = acc_table[arch][str(primary_seed)]
        ranked = sorted(acc_table[arch].items(), key=lambda kv: -kv[1])
        secondary_seed, secondary_acc = next((s, a) for s, a in ranked if s != str(primary_seed))
        gap = primary_acc - secondary_acc
        comparable = gap <= COMPARABILITY_THRESHOLD
        cells[arch] = {
            "primary_seed": int(primary_seed), "primary_accuracy": primary_acc,
            "secondary_seed": int(secondary_seed), "secondary_accuracy": secondary_acc,
            "accuracy_gap": gap, "comparable": comparable,
            "all_seed_accuracies_ranked": ranked,
        }
    return cells


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cells = select_cells()
    print("=== cell selection ===")
    print(json.dumps(cells, indent=2, default=str))

    bottleneck_summary = {}
    for arch, c in cells.items():
        if not c["comparable"]:
            bottleneck_summary[arch] = (
                f"SINGLE-SOLVER BOTTLENECK -- primary seed {c['primary_seed']} ({c['primary_accuracy']:.3f}) "
                f"exceeds secondary {c['secondary_seed']} ({c['secondary_accuracy']:.3f}) by "
                f"{c['accuracy_gap']:.3f} (> {COMPARABILITY_THRESHOLD} threshold)."
            )
        else:
            bottleneck_summary[arch] = f"NO BOTTLENECK (gap={c['accuracy_gap']:.3f})."
    print("\n=== bottleneck check ===")
    print(json.dumps(bottleneck_summary, indent=2))

    out = {
        "task": TASK,
        "description": "Cell selection for modular-arithmetic-simple's bounded Phase 3 patching "
                       "pilot: primary (best) + secondary (next-best) seed per architecture, pulled "
                       "from flare_a2_modular_arithmetic_simple.json (built here from eval/test.json "
                       "files, rec+ns/validation-short convention -- no new training).",
        "comparability_threshold": COMPARABILITY_THRESHOLD,
        "cell_selection": cells,
        "bottleneck_check": bottleneck_summary,
        "final_cells": [
            {"arch": arch, "seed": c["primary_seed"], "role": "primary"} for arch, c in cells.items()
        ] + [
            {"arch": arch, "seed": c["secondary_seed"], "role": "secondary", "comparable": c["comparable"]}
            for arch, c in cells.items()
        ],
    }
    out_path = RESULTS / "phase3_modarith_cell_selection.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
