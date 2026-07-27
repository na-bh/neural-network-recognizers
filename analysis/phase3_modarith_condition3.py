"""Modular-arithmetic-simple Phase 3 (bounded pilot), Condition 3:
target-computation causal patching via PERMUTED (op,digit)-pair
counterfactuals (position-swept early/mid/late), all 8 cells. Uses the
already-audited N=25 pairs from phase3_modarith_counterfactual_design.json
(condition_3_target_computation_permuted_pairs) -- no new pair
construction here.

This is the load-bearing carrier-account test for this task: clean/
corrupted differ ONLY in which (op,digit) pair sits where (all six listed
shortcut features identical), isolating order-sensitive mod-5 accumulation
specifically.

SPECIFIC INVESTIGATION per this condition's approval: three architectures
solve this task at ~0.997 accuracy (rnn, lstm, mamba per cell_selection).
If Condition 3 shows large causal gaps at multiple positions, these models
causally use the order-sensitive computation. If gaps are near-zero DESPITE
that high accuracy, something else is being detected by the model instead
-- this is investigated directly here (not smoothed over) by additionally
reporting, per cell: (a) each cell's own known standard-test accuracy
(from phase3_modarith_cell_selection.json), (b) the raw clean/corrupt
logits and whether the model's UNPATCHED decision on the corrupted example
is even directionally correct (i.e., does logit_corrupt < logit_clean, or
are they statistically indistinguishable), to distinguish "can't tell
clean from corrupt" from "can tell them apart, but the patched-state gap
is smaller than the causal threshold."

PRIMARY evidence is the behavioral logit gap. Full-state patch (top-layer
readout) is SECONDARY (wiring check), flagged unreliable when the gap is
near zero.

Saves analysis_outputs/final_results/phase3_modarith_condition3.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_modarith_condition3.py
"""

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "analysis")
from phase3_modarith_patching_common import (
    TASK, RESULTS, CELLS, get_tok2idx, run_cell, summarize, load_cell_selection_context,
)

POSITIONS = ["early", "mid", "late"]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    design = json.loads((RESULTS / "phase3_modarith_counterfactual_design.json").read_text())
    cond_design = design["condition_3_target_computation_permuted_pairs"]
    tok2idx = get_tok2idx()
    cell_ctx = load_cell_selection_context()

    results_by_position = {}
    raw_logits_by_position = {}
    for label in POSITIONS:
        pairs = cond_design["by_position"][label]["pairs"]
        assert cond_design["by_position"][label]["audit"]["target_property_isolated"]
        cells = []
        raw_logits = {}
        for arch, seed, role in CELLS:
            print(f"\n--- [{label}] {arch} seed{seed} ({role}) ---", flush=True)
            res, ckpt = run_cell(arch, seed, pairs, tok2idx)
            cell = summarize(res, f"target_computation_permuted_{label}", arch, seed, role, ckpt)
            cell["position_label"] = label
            cells.append(cell)
            # diagnostic: does the UNPATCHED model even distinguish clean from corrupt directionally?
            clean_logits = np.array([r["clean_logit"] for r in res])
            corrupt_logits = np.array([r["corrupt_logit"] for r in res])
            n_clean_correct_direction = int(np.sum(clean_logits > 0))  # clean should be accepted
            n_corrupt_correct_direction = int(np.sum(corrupt_logits < 0))  # corrupt should be rejected
            raw_logits[f"{arch}_seed{seed}"] = {
                "mean_clean_logit": float(clean_logits.mean()), "mean_corrupt_logit": float(corrupt_logits.mean()),
                "n_clean_accepted (logit>0)": n_clean_correct_direction, "n_total": len(res),
                "n_corrupt_rejected (logit<0)": n_corrupt_correct_direction,
                "unpatched_behavioral_discrimination": (
                    "MODEL DOES distinguish clean/corrupt behaviorally (accepts clean, rejects "
                    "corrupt) even if the CAUSAL gap from patching is small -- suggests the "
                    "decision-relevant difference may not be localized at the single top-layer "
                    "readout position patched here."
                    if n_clean_correct_direction >= len(res) * 0.8 and n_corrupt_correct_direction >= len(res) * 0.8
                    else "Model does NOT reliably distinguish clean from corrupt behaviorally either -- "
                         "consistent with the causal gap being genuinely near-zero, not a localization "
                         "artifact."
                ),
            }
            gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
            print(f"  PRIMARY behavioral gap={gap:.4f} causal={cell['causally_effective']}  |  "
                  f"wiring rf_mean={rf:.4f}  |  mean_clean_logit={clean_logits.mean():.3f} "
                  f"mean_corrupt_logit={corrupt_logits.mean():.3f}", flush=True)
        results_by_position[label] = cells
        raw_logits_by_position[label] = raw_logits

    per_cell_summary = {}
    for arch, seed, role in CELLS:
        key = f"{arch}_seed{seed}"
        gaps = {label: next(c for c in results_by_position[label] if c["arch"] == arch and c["seed"] == seed)
                       ["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
               for label in POSITIONS}
        causal = {label: next(c for c in results_by_position[label] if c["arch"] == arch and c["seed"] == seed)
                         ["causally_effective"] for label in POSITIONS}
        own_accuracy = next((a for s, a in cell_ctx[arch]["all_seed_accuracies_ranked"] if int(s) == seed), None)
        entry = {
            "arch": arch, "seed": seed, "role": role,
            "gap_by_position": gaps, "causal_by_position": causal,
            "all_positions_causal": all(causal.values()), "no_position_causal": not any(causal.values()),
            "this_seed_own_standard_test_accuracy": own_accuracy,
            "diagnostic_by_position": {label: raw_logits_by_position[label][key] for label in POSITIONS},
        }
        verdict = (
            "GENUINE ORDER-SENSITIVE VERIFICATION -- causal effect at ALL positions despite high "
            "standalone accuracy: this cell causally uses the order-sensitive mod-5 computation."
            if entry["all_positions_causal"] else
            "NEAR-ZERO CAUSAL GAP DESPITE HIGH ACCURACY -- investigate: see diagnostic_by_position "
            "for whether the model even behaviorally distinguishes clean/corrupt (accepts clean, "
            "rejects corrupt) despite the small patched-state gap. If it DOES distinguish them "
            "behaviorally, the decision-relevant computation may not be localized at the single "
            "readout position/layer patched here -- a genuine open question, not resolved by this "
            "condition alone."
            if entry["no_position_causal"] and own_accuracy is not None and own_accuracy > 0.9 else
            "NULL/POSITION-SPECIFIC -- see per-position breakdown."
        )
        entry["verdict"] = verdict
        per_cell_summary[key] = entry
        print(f"{key}: " + " ".join(f"{l}={gaps[l]:.3f}" for l in POSITIONS) +
              f"  own_acc={own_accuracy}  [{verdict.split(' -- ')[0]}]")

    n_all_causal = sum(1 for v in per_cell_summary.values() if v["all_positions_causal"])
    n_null = sum(1 for v in per_cell_summary.values() if v["no_position_causal"])
    n_investigate = sum(1 for v in per_cell_summary.values()
                        if v["no_position_causal"] and v["this_seed_own_standard_test_accuracy"]
                        and v["this_seed_own_standard_test_accuracy"] > 0.9)
    overall = {
        "n_cells_causal_at_all_positions": n_all_causal, "n_cells_null_everywhere": n_null,
        "n_cells_position_specific": len(CELLS) - n_all_causal - n_null,
        "n_cells_null_despite_high_accuracy_FLAGGED_FOR_INVESTIGATION": n_investigate,
        "carrier_account_reading": (
            f"{n_null}/{len(CELLS)} cells show UNIFORM ABSENCE, {n_all_causal}/{len(CELLS)} show "
            f"COMPREHENSIVE order-sensitive verification, {len(CELLS)-n_all_causal-n_null}/{len(CELLS)} "
            f"position-specific. {n_investigate} of the null cells have own standard-test accuracy "
            f">0.9 -- these are flagged explicitly (per this condition's approval) as requiring "
            f"further investigation rather than being read as clean carrier-account confirmations, "
            f"since a model achieving >99% accuracy while showing zero causal sensitivity to an "
            f"order-permutation that changes the true answer needs an explanation for HOW it is "
            f"achieving that accuracy."
        ),
    }
    print("\n=== overall ===")
    print(json.dumps(overall, indent=2, default=str))

    out = {
        "condition": "condition_3_target_computation_permuted_pairs", "task": TASK,
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (top-layer readout) is SECONDARY, a wiring check only, flagged "
                 "UNRELIABLE when the behavioral gap is near zero.",
        "description": "Load-bearing carrier-account test: clean/corrupted differ ONLY in which "
                       "(op,digit) pair sits where (all six shortcut features identical), isolating "
                       "order-sensitive mod-5 accumulation. Swept early/mid/late.",
        "n_pairs_per_position": 25, "positions_tested": POSITIONS,
        "results_by_position": results_by_position,
        "per_cell_summary": per_cell_summary,
        "overall": overall,
    }
    out_path = RESULTS / "phase3_modarith_condition3.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
