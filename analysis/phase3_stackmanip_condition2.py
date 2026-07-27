"""Stack-manipulation Phase 3, condition 2 of 3: task-specific structural
features (length_mismatch / stack_size_arithmetic_consistent,
bit_count_availability). Prediction: causally load-bearing across most
cells, since these two features (plus Condition 1's marker_count/well-
formedness) jointly cover 97.6% of all negatives per the task audit.

PRIMARY evidence is the behavioral logit gap; full-state patch is a
SECONDARY wiring check, flagged UNRELIABLE in near-zero-gap cells.

Saves analysis_outputs/final_results/phase3_stackmanip_condition2.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_stackmanip_condition2.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_stackmanip_counterfactual_design import (
    length_mismatch_pairs, bit_count_availability_pairs,
    audit_length_mismatch_pairs, audit_bit_count_availability_pairs,
)
from phase3_stackmanip_patching_common import CELLS, get_tok2idx, run_cell, summarize

RESULTS = Path("analysis_outputs/final_results")
N_PAIRS = 25


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)

    lmp = length_mismatch_pairs(N_PAIRS, rng)
    bcp = bit_count_availability_pairs(N_PAIRS, rng)
    audits = {
        "length_mismatch_pairs": audit_length_mismatch_pairs(lmp),
        "bit_count_availability_pairs": audit_bit_count_availability_pairs(bcp),
    }
    for name, a in audits.items():
        print(f"=== {name} audit ===", flush=True)
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    tok2idx = get_tok2idx()

    cells = []
    for arch, seed, role in CELLS:
        print(f"\n--- {arch} seed{seed} ({role}) ---", flush=True)
        for pair_type, pairs in [("length_mismatch", lmp), ("bit_count_availability", bcp)]:
            results, ckpt = run_cell(arch, seed, pairs, tok2idx)
            cell = summarize(results, pair_type, arch, seed, role, ckpt)
            cells.append(cell)
            gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
            unreliable = cell["wiring_check_full_state_patch_SECONDARY"]["UNRELIABLE_near_zero_behavioral_gap"]
            print(f"  [{pair_type}] PRIMARY behavioral gap={gap:.4f} (causally_effective={cell['causally_effective']})  |  "
                  f"wiring rf_mean={rf:.4f} n={cell['wiring_check_full_state_patch_SECONDARY']['n_nondegenerate_denom']} "
                  f"{'[UNRELIABLE]' if unreliable else ''}", flush=True)

    n_effective = sum(1 for c in cells if c["causally_effective"])
    n_total = len(cells)
    per_arch = {}
    for arch in ["rnn", "lstm", "transformer", "mamba"]:
        arch_cells = [c for c in cells if c["arch"] == arch]
        per_arch[arch] = {
            c["pair_type"] + "_" + c["role"]: c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            for c in arch_cells
        }
    prediction_check = {
        "n_cells_total (8 cells x 2 pair types)": n_total,
        "n_cells_causally_effective (gap >= 0.05)": n_effective,
        "fraction_causally_effective": n_effective / n_total,
        "per_architecture_gaps": per_arch,
        "prediction": "causally load-bearing across MOST cells, given 97.6% joint shortcut coverage",
        "verdict": (
            "CONFIRMED -- most cells show a strong causal effect from these task-specific structural "
            "features, consistent with the 97.6% joint coverage finding from the task audit."
            if n_effective / n_total >= 0.75 else
            "PARTIALLY CONFIRMED or NOT CONFIRMED -- reported per-cell numbers above; investigate "
            "any cell/pair-type combination showing an unexpectedly small effect given the high "
            "coverage these features provide over the real negative distribution."
        ),
    }
    print("\n=== prediction check ===", flush=True)
    print(json.dumps(prediction_check, indent=2, default=str))

    out = {
        "condition": "condition_2_task_specific_structural_critical",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only, flagged UNRELIABLE when the behavioral gap is "
                 "near zero (< 0.05).",
        "description": (
            "Tests whether stack_size_arithmetic_consistent (length_mismatch) and answer_bit_counts_"
            "subset_of_available (bit_count_availability) causally drive the accept/reject decision, "
            "all 8 cells. These two features, together with Condition 1's marker_count/well-"
            "formedness, jointly cover 97.6% of all real negatives (task audit finding) -- predicted "
            "causally load-bearing across most cells."
        ),
        "n_pairs_generated": N_PAIRS,
        "pair_construction_audits": audits,
        "sample_pairs": {"length_mismatch": lmp[:3], "bit_count_availability": bcp[:3]},
        "cells": cells,
        "prediction_check": prediction_check,
    }
    out_path = RESULTS / "phase3_stackmanip_condition2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
