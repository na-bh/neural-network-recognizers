"""Dyck-2-3 Phase 3 (bounded pilot), Condition 1: bracket_count_parity
causal patching, position-swept (early/mid/late), all 8 cells. Uses the
already-audited N=25 pairs from phase3_dyck_counterfactual_design.json
(condition_1_bracket_count_parity) -- no new pair construction here.

PRIMARY evidence is the behavioral logit gap (unpatched clean vs corrupt).
Full-state patch (top-layer readout) is a SECONDARY wiring check, flagged
unreliable when the behavioral gap is near zero.

Transformer-specific interpretive discipline (per this condition's
approval): Transformer's own accuracy population is mostly non-solving
(5/10 seeds at exact chance 0.5, only 3/10 above 0.60 -- phase3_dyck_
cell_selection.json). A near-zero causal effect for Transformer is reported
ALONGSIDE this chance-level baseline so "doesn't use this feature" can be
distinguished from "doesn't solve the task at all."

Saves analysis_outputs/final_results/phase3_dyck_condition1.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_dyck_condition1.py
"""

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "analysis")
from phase3_dyck_patching_common import (
    TASK, RESULTS, CELLS, get_tok2idx, run_cell, summarize, load_cell_selection_context,
)

POSITIONS = ["early", "mid", "late"]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    design = json.loads((RESULTS / "phase3_dyck_counterfactual_design.json").read_text())
    cond_design = design["condition_1_bracket_count_parity"]
    tok2idx = get_tok2idx()
    cell_ctx = load_cell_selection_context()

    results_by_position = {}
    for label in POSITIONS:
        pairs = cond_design["by_position"][label]["pairs"]
        assert cond_design["by_position"][label]["audit"]["target_property_isolated"]
        cells = []
        for arch, seed, role in CELLS:
            print(f"\n--- [{label}] {arch} seed{seed} ({role}) ---", flush=True)
            res, ckpt = run_cell(arch, seed, pairs, tok2idx)
            cell = summarize(res, f"bracket_count_parity_{label}", arch, seed, role, ckpt)
            cell["position_label"] = label
            cells.append(cell)
            gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
            print(f"  PRIMARY behavioral gap={gap:.4f} causal={cell['causally_effective']}  |  "
                  f"wiring rf_mean={rf:.4f}", flush=True)
        results_by_position[label] = cells

    per_cell_summary = {}
    for arch, seed, role in CELLS:
        key = f"{arch}_seed{seed}"
        gaps = {label: next(c for c in results_by_position[label] if c["arch"] == arch and c["seed"] == seed)
                       ["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
               for label in POSITIONS}
        causal = {label: next(c for c in results_by_position[label] if c["arch"] == arch and c["seed"] == seed)
                         ["causally_effective"] for label in POSITIONS}
        entry = {"arch": arch, "seed": seed, "role": role,
                "gap_by_position": gaps, "causal_by_position": causal,
                "all_positions_causal": all(causal.values()), "no_position_causal": not any(causal.values())}
        if arch == "transformer":
            seed_acc = next(a for s, a in cell_ctx["transformer"]["all_seed_accuracies_ranked"] if int(s) == seed)
            entry["transformer_own_accuracy_this_seed"] = seed_acc
            entry["transformer_population_context"] = (
                f"{cell_ctx['transformer']['n_seeds_above_near_chance_threshold']}/"
                f"{cell_ctx['transformer']['n_seeds_total']} seeds exceed the near-chance threshold "
                f"(0.60); this seed's own accuracy is {seed_acc:.3f}."
            )
            if entry["no_position_causal"]:
                entry["interpretation"] = (
                    "NULL AT EVERY POSITION. Given this architecture's population is mostly non-"
                    "solving, this null is reported ALONGSIDE the chance-level baseline rather than "
                    "read as 'Transformer uses different features' -- with this seed's own accuracy "
                    f"at {seed_acc:.3f}, a null causal effect is compatible with 'doesn't solve the "
                    "task at all,' not necessarily a genuine feature-independence finding."
                )
        per_cell_summary[key] = entry
        print(f"{key}: " + " ".join(f"{l}={gaps[l]:.3f}" for l in POSITIONS))

    n_all_causal = sum(1 for v in per_cell_summary.values() if v["all_positions_causal"])
    n_null = sum(1 for v in per_cell_summary.values() if v["no_position_causal"])
    overall = {
        "n_cells_causal_at_all_positions": n_all_causal,
        "n_cells_null_everywhere": n_null,
        "n_cells_position_specific": len(CELLS) - n_all_causal - n_null,
        "prediction_check": (
            f"Predicted causally load-bearing across most cells if the shortcut is used (bracket_"
            f"count_parity has only a 9.5% blind spot on real negatives -- a strong shortcut "
            f"signal). Found: {n_all_causal}/{len(CELLS)} cells causal at ALL 3 positions, "
            f"{n_null}/{len(CELLS)} null everywhere."
        ),
    }
    print("\n=== overall ===")
    print(json.dumps(overall, indent=2, default=str))

    out = {
        "condition": "condition_1_bracket_count_parity", "task": TASK,
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (top-layer readout) is SECONDARY, a wiring check only, flagged "
                 "UNRELIABLE when the behavioral gap is near zero.",
        "n_pairs_per_position": 25, "positions_tested": POSITIONS,
        "results_by_position": results_by_position,
        "per_cell_summary": per_cell_summary,
        "overall": overall,
    }
    out_path = RESULTS / "phase3_dyck_condition1.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
