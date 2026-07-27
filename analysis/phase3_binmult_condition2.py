"""Binary-multiplication Phase 3, condition 2 of 3: task-specific
structural features -- the two CHEAPEST POSSIBLE shortcuts (length_
sufficient, the classical multiplication length constraint, and lsb_
consistent, the exact LSB(z)=LSB(x) AND LSB(y) identity). Prediction:
causally load-bearing across most cells.

PRIMARY evidence is the behavioral logit gap; full-state patch is a
SECONDARY wiring check, flagged UNRELIABLE in near-zero-gap cells.

Saves analysis_outputs/final_results/phase3_binmult_condition2.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_binmult_condition2.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_binmult_counterfactual_design import (
    length_insufficient_pairs, lsb_flip_pairs,
    audit_length_insufficient_pairs, audit_lsb_flip_pairs,
)
from phase3_binmult_patching_common import CELLS, get_tok2idx, run_cell, summarize

RESULTS = Path("analysis_outputs/final_results")
N_PAIRS = 25


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)

    lip = length_insufficient_pairs(N_PAIRS, rng)
    lfp = lsb_flip_pairs(N_PAIRS, rng)
    audits = {
        "length_insufficient_pairs": audit_length_insufficient_pairs(lip),
        "lsb_flip_pairs": audit_lsb_flip_pairs(lfp),
    }
    for name, a in audits.items():
        print(f"=== {name} audit ===", flush=True)
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    tok2idx = get_tok2idx()

    cells = []
    for arch, seed, role in CELLS:
        print(f"\n--- {arch} seed{seed} ({role}) ---", flush=True)
        for pair_type, pairs in [("length_insufficient", lip), ("lsb_flip", lfp)]:
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
    summary = {
        "n_cells_total (8 cells x 2 pair types)": n_total,
        "n_cells_causally_effective (gap >= 0.05)": n_effective,
        "fraction_causally_effective": n_effective / n_total,
        "per_architecture_gaps": per_arch,
        "context": (
            "Condition 1 already found ALL 8 cells strongly causally reliant on operator_count/"
            "equals_count/fields_well_formed (with the transformer-primary fields_well_formed "
            "exception). This condition tests the two CHEAPEST possible shortcuts (length "
            "arithmetic, a single LSB AND check)."
        ),
    }
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, indent=2, default=str))

    out = {
        "condition": "condition_2_task_specific_structural_cheapest_shortcuts",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only, flagged UNRELIABLE when the behavioral gap is "
                 "near zero (< 0.05).",
        "description": (
            "Tests whether length_sufficient (bitlength(x)+bitlength(y)-1 arithmetic) and lsb_"
            "consistent (LSB(z)=LSB(x) AND LSB(y), an exact identity requiring no multiplication) "
            "causally drive the accept/reject decision, all 8 cells. These are the two cheapest "
            "possible shortcuts for this task."
        ),
        "n_pairs_generated": N_PAIRS,
        "pair_construction_audits": audits,
        "sample_pairs": {"length_insufficient": lip[:3], "lsb_flip": lfp[:3]},
        "cells": cells,
        "summary": summary,
    }
    out_path = RESULTS / "phase3_binmult_condition2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
