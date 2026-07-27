"""Dyck-2-3 Phase 3 (bounded pilot), Condition 3b: target_computation
causal patching via DEPTH-EXCEEDED counterfactuals (position-swept:
early/mid/late), all 8 cells. Uses the already-audited N=25 pairs from
phase3_dyck_counterfactual_design.json (condition_3b_target_computation_
depth_exceeded) -- no new pair construction here.

Kept as its own sub-condition, distinct from 3a: a bounded-COUNTER
violation (nesting momentarily exceeds depth 3), not a LIFO-order
violation -- a different computational primitive (threshold-counting vs.
stack-order matching). All four listed shortcut features are preserved by
construction (see phase3_dyck_counterfactuals.py's docstring).

PRIMARY evidence is the behavioral logit gap. Full-state patch is SECONDARY
(wiring check), flagged unreliable when the gap is near zero.

Transformer-specific interpretive discipline: see phase3_dyck_condition1.py.

Saves analysis_outputs/final_results/phase3_dyck_condition3b.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_dyck_condition3b.py
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, "analysis")
from phase3_dyck_patching_common import (
    TASK, RESULTS, CELLS, get_tok2idx, run_cell, summarize, load_cell_selection_context,
)

POSITIONS = ["early", "mid", "late"]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    design = json.loads((RESULTS / "phase3_dyck_counterfactual_design.json").read_text())
    cond_design = design["condition_3b_target_computation_depth_exceeded"]
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
            cell = summarize(res, f"target_computation_depth_exceeded_{label}", arch, seed, role, ckpt)
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
        verdict = (
            "COMPREHENSIVE DEPTH-VERIFICATION -- causal effect at ALL positions: genuine bounded-"
            "counter (depth) tracking, a specific new finding under the carrier account, "
            "architecturally distinct from 3a's LIFO-order verification."
            if entry["all_positions_causal"] else
            "UNIFORM ABSENCE -- null at every position, consistent with the carrier account's "
            "prediction."
            if entry["no_position_causal"] else
            "POSITION-SPECIFIC -- causal at some positions but not others: partial depth-tracking, "
            "not comprehensive."
        )
        entry["verdict"] = verdict
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
                    "read as 'Transformer doesn't track depth' -- with this seed's own accuracy at "
                    f"{seed_acc:.3f}, a null causal effect here is compatible with 'doesn't solve the "
                    "task at all,' not necessarily a genuine absence-of-tracking finding."
                )
        per_cell_summary[key] = entry
        print(f"{key}: " + " ".join(f"{l}={gaps[l]:.3f}" for l in POSITIONS) + f"  [{verdict.split(' -- ')[0]}]")

    n_comprehensive = sum(1 for v in per_cell_summary.values() if v["all_positions_causal"])
    n_null = sum(1 for v in per_cell_summary.values() if v["no_position_causal"])
    n_partial = len(CELLS) - n_comprehensive - n_null
    overall = {
        "n_cells_comprehensive": n_comprehensive, "n_cells_null": n_null, "n_cells_position_specific": n_partial,
        "carrier_account_reading": (
            f"{n_null}/{len(CELLS)} cells show UNIFORM ABSENCE across all 3 positions, "
            f"{n_comprehensive}/{len(CELLS)} show COMPREHENSIVE depth-tracking, "
            f"{n_partial}/{len(CELLS)} show POSITION-SPECIFIC effects. " +
            ("Carrier account CONFIRMED for depth-tracking on this task."
             if n_comprehensive == 0 and n_partial == 0 else
             f"Carrier account is NOT uniformly confirmed for depth-tracking -- "
             f"{n_comprehensive + n_partial}/{len(CELLS)} cells show some causal use; compare "
             f"against condition 3a's LIFO-matching results to see whether the two stack-carrier "
             f"sub-mechanisms (order vs. depth) dissociate per cell."),
        ),
    }
    print("\n=== overall (carrier account reading) ===")
    print(json.dumps(overall, indent=2, default=str))

    out = {
        "condition": "condition_3b_target_computation_depth_exceeded", "task": TASK,
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (top-layer readout) is SECONDARY, a wiring check only, flagged "
                 "UNRELIABLE when the behavioral gap is near zero.",
        "description": "Carrier-account test for Dyck-2-3's BOUNDED-COUNTER (depth) requirement, "
                       "architecturally distinct from condition 3a's LIFO-order requirement: clean/"
                       "corrupted differ ONLY in a momentary depth-4 excursion (inserted matched "
                       "pair), all four listed shortcut features preserved by construction. Swept "
                       "across insertion position early/mid/late.",
        "n_pairs_per_position": 25, "positions_tested": POSITIONS,
        "results_by_position": results_by_position,
        "per_cell_summary": per_cell_summary,
        "overall": overall,
    }
    out_path = RESULTS / "phase3_dyck_condition3b.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
