"""Binary-multiplication Phase 3, condition 1 of 3: structural features
(operator_count, equals_count, fields_well_formed, null_control TRUE
MINIMAL PAIR via commutativity).

PRIMARY evidence is the behavioral logit gap; full-state patch is a
SECONDARY wiring check, flagged UNRELIABLE in near-zero-gap cells.

Saves analysis_outputs/final_results/phase3_binmult_condition1.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_binmult_condition1.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_binmult_counterfactual_design import (
    operator_count_pairs, equals_count_pairs, fields_well_formed_pairs, null_control_pairs,
    audit_operator_count_pairs, audit_equals_count_pairs, audit_fields_well_formed_pairs, audit_null_control_pairs,
)
from phase3_binmult_patching_common import CELLS, get_tok2idx, run_cell, summarize

RESULTS = Path("analysis_outputs/final_results")
N_PAIRS = 25


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    ocp = operator_count_pairs(N_PAIRS, rng)
    ecp = equals_count_pairs(N_PAIRS, rng)
    fwp = fields_well_formed_pairs(N_PAIRS, rng)
    ncp = null_control_pairs(N_PAIRS, rng)
    audits = {
        "operator_count_pairs": audit_operator_count_pairs(ocp),
        "equals_count_pairs": audit_equals_count_pairs(ecp),
        "fields_well_formed_pairs": audit_fields_well_formed_pairs(fwp),
        "null_control_pairs": audit_null_control_pairs(ncp),
    }
    for name, a in audits.items():
        print(f"=== {name} audit ===", flush=True)
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    tok2idx = get_tok2idx()

    cells = []
    for arch, seed, role in CELLS:
        print(f"\n--- {arch} seed{seed} ({role}) ---", flush=True)
        for pair_type, pairs in [
            ("operator_count", ocp), ("equals_count", ecp), ("fields_well_formed", fwp),
            ("null_control_TRUE_MINIMAL_PAIR_COMMUTATIVITY", ncp),
        ]:
            results, ckpt = run_cell(arch, seed, pairs, tok2idx)
            cell = summarize(results, pair_type, arch, seed, role, ckpt)
            cells.append(cell)
            gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
            unreliable = cell["wiring_check_full_state_patch_SECONDARY"]["UNRELIABLE_near_zero_behavioral_gap"]
            print(f"  [{pair_type}] PRIMARY behavioral gap={gap:.4f} (causally_effective={cell['causally_effective']})  |  "
                  f"wiring rf_mean={rf:.4f} n={cell['wiring_check_full_state_patch_SECONDARY']['n_nondegenerate_denom']} "
                  f"{'[UNRELIABLE]' if unreliable else ''}", flush=True)

    null_control_check = {}
    for arch, seed, role in CELLS:
        c = next(c for c in cells if c["arch"] == arch and c["seed"] == seed and c["pair_type"] == "null_control_TRUE_MINIMAL_PAIR_COMMUTATIVITY")
        gap = c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
        null_control_check[f"{arch}_seed{seed}_{role}"] = {"gap": gap, "spuriously_causal": gap >= 0.05}
    print("\n=== null-control check (TRUE minimal pair via commutativity; expect ~0 gap everywhere) ===", flush=True)
    print(json.dumps(null_control_check, indent=2, default=str))

    out = {
        "condition": "condition_1_structural_features",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only, flagged UNRELIABLE when the behavioral gap is "
                 "near zero (< 0.05).",
        "description": "Tests whether operator_count, equals_count, and fields_well_formed causally "
                       "drive the accept/reject decision, all 8 cells. null_control is a TRUE "
                       "minimal pair via commutativity (x*y == y*x, u_z identical, operands swapped) "
                       "-- both members genuinely valid, same true label; expected near-zero gap "
                       "everywhere.",
        "n_pairs_generated": N_PAIRS,
        "pair_construction_audits": audits,
        "sample_pairs": {"operator_count": ocp[:3], "equals_count": ecp[:3],
                         "fields_well_formed": fwp[:3], "null_control": ncp[:3]},
        "cells": cells,
        "null_control_check": null_control_check,
    }
    out_path = RESULTS / "phase3_binmult_condition1.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
