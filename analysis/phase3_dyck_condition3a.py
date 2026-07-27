"""Dyck-2-3 Phase 3 (bounded pilot), Condition 3a: target_computation
causal patching via TYPE-MISMATCH counterfactuals (swap-depth-swept:
depth levels 1/2/3), all 8 cells. Uses the already-audited N=25 pairs from
phase3_dyck_counterfactual_design.json (condition_3a_target_computation_
type_mismatch) -- no new pair construction here.

This is the load-bearing test for the carrier account's cross-family
prediction on this task: clean/corrupted differ ONLY in which nested
closer's type label sits where (all four listed shortcut features
identical), isolating LIFO stack-order verification specifically. Swap-
depth is the analog of a position sweep for this task, per this condition's
approval.

PRIMARY evidence is the behavioral logit gap. Full-state patch is SECONDARY
(wiring check), flagged unreliable when the gap is near zero.

Transformer-specific interpretive discipline: see phase3_dyck_condition1.py.

Saves analysis_outputs/final_results/phase3_dyck_condition3a.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_dyck_condition3a.py
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, "analysis")
from phase3_dyck_patching_common import (
    TASK, RESULTS, CELLS, get_tok2idx, run_cell, summarize, load_cell_selection_context,
)

DEPTHS = [1, 2, 3]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    design = json.loads((RESULTS / "phase3_dyck_counterfactual_design.json").read_text())
    cond_design = design["condition_3a_target_computation_type_mismatch"]
    tok2idx = get_tok2idx()
    cell_ctx = load_cell_selection_context()

    results_by_depth = {}
    for depth in DEPTHS:
        key_d = f"depth{depth}"
        pairs = cond_design["by_depth"][key_d]["pairs"]
        assert cond_design["by_depth"][key_d]["audit"]["target_property_isolated"]
        cells = []
        for arch, seed, role in CELLS:
            print(f"\n--- [depth={depth}] {arch} seed{seed} ({role}) ---", flush=True)
            res, ckpt = run_cell(arch, seed, pairs, tok2idx)
            cell = summarize(res, f"target_computation_type_mismatch_depth{depth}", arch, seed, role, ckpt)
            cell["depth_level"] = depth
            cells.append(cell)
            gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
            print(f"  PRIMARY behavioral gap={gap:.4f} causal={cell['causally_effective']}  |  "
                  f"wiring rf_mean={rf:.4f}", flush=True)
        results_by_depth[key_d] = cells

    per_cell_summary = {}
    for arch, seed, role in CELLS:
        key = f"{arch}_seed{seed}"
        gaps = {f"depth{d}": next(c for c in results_by_depth[f"depth{d}"] if c["arch"] == arch and c["seed"] == seed)
                       ["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
               for d in DEPTHS}
        causal = {f"depth{d}": next(c for c in results_by_depth[f"depth{d}"] if c["arch"] == arch and c["seed"] == seed)
                         ["causally_effective"] for d in DEPTHS}
        entry = {"arch": arch, "seed": seed, "role": role,
                "gap_by_depth": gaps, "causal_by_depth": causal,
                "all_depths_causal": all(causal.values()), "no_depth_causal": not any(causal.values())}
        verdict = (
            "COMPREHENSIVE STACK-VERIFICATION -- causal effect at ALL depth levels (1,2,3): genuine "
            "LIFO type-matching, a specific new finding under the carrier account."
            if entry["all_depths_causal"] else
            "UNIFORM ABSENCE -- null at every depth level, consistent with the carrier account's "
            "prediction (matching marker-family durable-carrier tasks)."
            if entry["no_depth_causal"] else
            "DEPTH-SPECIFIC -- causal at some depth levels but not others: partial stack-"
            "verification, not comprehensive."
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
            if entry["no_depth_causal"]:
                entry["interpretation"] = (
                    "NULL AT EVERY DEPTH. Given this architecture's population is mostly non-solving, "
                    "this null is reported ALONGSIDE the chance-level baseline rather than read as "
                    "'Transformer doesn't stack-verify' -- with this seed's own accuracy at "
                    f"{seed_acc:.3f}, a null causal effect here is compatible with 'doesn't solve the "
                    "task at all,' not necessarily a genuine absence-of-verification finding."
                )
        per_cell_summary[key] = entry
        print(f"{key}: " + " ".join(f"depth{d}={gaps[f'depth{d}']:.3f}" for d in DEPTHS) + f"  [{verdict.split(' -- ')[0]}]")

    n_comprehensive = sum(1 for v in per_cell_summary.values() if v["all_depths_causal"])
    n_null = sum(1 for v in per_cell_summary.values() if v["no_depth_causal"])
    n_partial = len(CELLS) - n_comprehensive - n_null
    overall = {
        "n_cells_comprehensive": n_comprehensive, "n_cells_null": n_null, "n_cells_depth_specific": n_partial,
        "carrier_account_reading": (
            f"{n_null}/{len(CELLS)} cells show UNIFORM ABSENCE across all 3 depth levels, "
            f"{n_comprehensive}/{len(CELLS)} show COMPREHENSIVE stack-verification, "
            f"{n_partial}/{len(CELLS)} show DEPTH-SPECIFIC effects. " +
            ("Carrier account CONFIRMED for this task: no architecture causally verifies LIFO "
             "type-matching -- matches the marker-family durable-carrier pattern (uniform absence)."
             if n_comprehensive == 0 and n_partial == 0 else
             f"Carrier account is NOT uniformly confirmed -- {n_comprehensive + n_partial}/{len(CELLS)} "
             f"cells show some causal use of type-matching; see per-cell verdicts above (and the "
             f"Transformer population-context caveat where applicable)."),
        ),
    }
    print("\n=== overall (carrier account reading) ===")
    print(json.dumps(overall, indent=2, default=str))

    out = {
        "condition": "condition_3a_target_computation_type_mismatch", "task": TASK,
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (top-layer readout) is SECONDARY, a wiring check only, flagged "
                 "UNRELIABLE when the behavioral gap is near zero.",
        "description": "Load-bearing carrier-account test for Dyck-2-3: clean/corrupted differ ONLY "
                       "in which nested closer's type sits where (all four listed shortcut features "
                       "identical), isolating LIFO stack-order verification. Swept across nesting "
                       "depth levels 1/2/3 (the position-sweep analog for a stack-based task).",
        "n_pairs_per_depth": 25, "depths_tested": DEPTHS,
        "results_by_depth": results_by_depth,
        "per_cell_summary": per_cell_summary,
        "overall": overall,
    }
    out_path = RESULTS / "phase3_dyck_condition3a.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
