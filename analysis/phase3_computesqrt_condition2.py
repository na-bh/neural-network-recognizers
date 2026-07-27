"""Compute-sqrt Phase 3, condition 2 of 3: task-specific structural
features (answer_length_insufficient, late_second_marker -- the exact O(1)
length shortcut and the LATE-placement multiple_markers probe matching the
real hard-negative distribution's own pattern).

PRIMARY evidence is the behavioral logit gap; full-state patch is a
SECONDARY wiring check, flagged UNRELIABLE in near-zero-gap cells.

Saves analysis_outputs/final_results/phase3_computesqrt_condition2.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_computesqrt_condition2.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_computesqrt_counterfactual_design import (
    answer_length_insufficient_pairs, late_second_marker_pairs,
    audit_answer_length_insufficient_pairs, audit_late_second_marker_pairs,
)
from phase3_computesqrt_patching_common import CELLS, get_tok2idx, run_cell, summarize

RESULTS = Path("analysis_outputs/final_results")
N_PAIRS = 25


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)

    alip = answer_length_insufficient_pairs(N_PAIRS, rng)
    lsmp = late_second_marker_pairs(N_PAIRS, rng)
    audits = {
        "answer_length_insufficient_pairs": audit_answer_length_insufficient_pairs(alip),
        "late_second_marker_pairs": audit_late_second_marker_pairs(lsmp),
    }
    for name, a in audits.items():
        print(f"=== {name} audit ===", flush=True)
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    tok2idx = get_tok2idx()

    cells = []
    for arch, seed, role in CELLS:
        print(f"\n--- {arch} seed{seed} ({role}) ---", flush=True)
        for pair_type, pairs in [("answer_length_insufficient", alip), ("late_second_marker", lsmp)]:
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
        "context": (
            "Condition 1 already found ALL 8 cells strongly causally reliant on marker_count/no_"
            "marker/marker_position (2.7-12.9 logit gaps, no exceptions). This condition tests two "
            "MORE task-specific structural features (answer-length arithmetic, LATE marker "
            "placement matching the real hard-negative distribution)."
        ),
    }
    print("\n=== summary ===", flush=True)
    print(json.dumps(prediction_check, indent=2, default=str))

    out = {
        "condition": "condition_2_task_specific_structural",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only, flagged UNRELIABLE when the behavioral gap is "
                 "near zero (< 0.05).",
        "description": (
            "Tests whether answer_length_sufficient (via a minimal-width-field truncation) and a "
            "LATE-placed second marker (matching the real hard-negative population's own "
            "distribution, 46.5% of nominally-hard negatives) causally drive the accept/reject "
            "decision, all 8 cells."
        ),
        "n_pairs_generated": N_PAIRS,
        "pair_construction_audits": audits,
        "sample_pairs": {"answer_length_insufficient": alip[:3], "late_second_marker": lsmp[:3]},
        "cells": cells,
        "summary": prediction_check,
    }
    out_path = RESULTS / "phase3_computesqrt_condition2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
