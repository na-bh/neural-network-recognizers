"""Missing-duplicate-string Phase 3, condition 1 of 3: structural features
(sequence_length_parity, blank_position, first_last_symbol NULL CONTROL).

PRIMARY evidence is the behavioral logit gap (unpatched clean vs corrupt);
full-state patch (canonical readout site per architecture) is a SECONDARY
wiring check, flagged UNRELIABLE in any cell/pair-type whose behavioral gap
is near zero (< 0.05).

Saves analysis_outputs/final_results/phase3_missdup_condition1.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_missdup_condition1.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_missdup_counterfactual_design import (
    length_parity_pairs, blank_position_pairs, first_last_symbol_pairs,
    audit_length_parity_pairs, audit_blank_position_pairs, audit_first_last_symbol_pairs,
)
from phase3_missdup_patching_common import CELLS, get_tok2idx, run_cell, summarize

RESULTS = Path("analysis_outputs/final_results")
N_HALF = 40
N_PAIRS = 25


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    lpp = length_parity_pairs(N_HALF, N_PAIRS, rng)
    bpp = blank_position_pairs(N_HALF, N_PAIRS, rng)
    flsp = first_last_symbol_pairs(N_HALF, N_PAIRS, rng)
    audits = {
        "length_parity_pairs": audit_length_parity_pairs(lpp),
        "blank_position_pairs": audit_blank_position_pairs(bpp),
        "first_last_symbol_pairs": audit_first_last_symbol_pairs(flsp),
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
            ("length_parity", lpp), ("blank_position", bpp), ("first_last_symbol_NULL_CONTROL", flsp),
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

    # ------------------------------------------------------------------
    # null-control check: first_last_symbol pairs should show ~0 gap for
    # EVERY architecture (both members of every pair are genuinely valid,
    # same label) -- a real behavioral gap here would mean a model is
    # spuriously keying off first-symbol identity, unrelated to validity.
    # ------------------------------------------------------------------
    null_control_check = {}
    for arch, seed, role in CELLS:
        c = next(c for c in cells if c["arch"] == arch and c["seed"] == seed and c["pair_type"] == "first_last_symbol_NULL_CONTROL")
        gap = c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
        null_control_check[f"{arch}_seed{seed}_{role}"] = {
            "gap": gap, "spuriously_causal": gap >= 0.05,
        }
    print("\n=== first_last_symbol null-control check (expect ~0 gap everywhere) ===", flush=True)
    print(json.dumps(null_control_check, indent=2, default=str))

    out = {
        "condition": "condition_1_structural_features",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only, flagged UNRELIABLE when the behavioral gap is "
                 "near zero (< 0.05).",
        "description": "Tests whether sequence_length_parity and blank_position causally drive the "
                       "accept/reject decision, all 8 cells (4 architectures x primary+secondary "
                       "seed). first_last_symbol is a NULL CONTROL (both pair members are genuinely "
                       "valid, same true label) -- expected near-zero gap everywhere; a nonzero gap "
                       "here would indicate a model spuriously keying off first-symbol identity.",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "pair_construction_audits": audits,
        "sample_pairs": {"length_parity": lpp[:3], "blank_position": bpp[:3], "first_last_symbol": flsp[:3]},
        "cells": cells,
        "first_last_symbol_null_control_check": null_control_check,
    }
    out_path = RESULTS / "phase3_missdup_condition1.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
