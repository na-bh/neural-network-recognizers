"""Missing-duplicate-string Phase 3, condition 3 of 3: position-of-mismatch
null test (revised per phase3_missdup_task_audit.json -- "which symbol is
duplicated" isn't constructible since the blank fill is a fixed constant).
clean=valid positive; corrupt=SAME length, SAME blank position, per-half
counts of '1's MATCHING BETWEEN CLEAN AND CORRUPT (built via a same-half
SWAP, which cannot change a multiset) -- the aggregate route is blind to
the difference, yet the true label flips. Swept EARLY/MID/LATE by swap
position, N=25 PER TERTILE PER CELL (not just N=25 total).

FLOOR-EFFECT PREDICTION (per phase3_missdup_condition2_investigation.json):
these pairs are drawn from the SAME genuinely-hard population (blank_count
==1, even length) that Condition 2 found all four architectures solve at
floor (RNN/LSTM/Mamba) or chance (Transformer) accuracy -- independent of
patching. Predicted: near-zero behavioral gap across all cells and all
three tertiles, for the SAME underlying reason (no non-degenerate decision
to probe), NOT because position-sensitivity is inherently absent as a
capability.

PRIMARY evidence is the behavioral logit gap; full-state patch is a
SECONDARY wiring check, flagged UNRELIABLE in near-zero-gap cells.

Saves analysis_outputs/final_results/phase3_missdup_condition3.json. STOP.

PYTHONPATH=src:analysis python analysis/phase3_missdup_condition3.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_missdup_counterfactual_design import position_mismatch_pairs, audit_position_mismatch_pairs, TERTILES
from phase3_missdup_patching_common import CELLS, get_tok2idx, run_cell, summarize

RESULTS = Path("analysis_outputs/final_results")
N_HALF = 40
N_PAIRS_PER_TERTILE = 25
DEGENERATE_GAP_THRESHOLD = 0.05


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    pairs_by_tertile = {}
    audits_by_tertile = {}
    for i, tertile in enumerate(TERTILES):
        rng = np.random.default_rng(4 + i)
        pmp = position_mismatch_pairs(N_HALF, N_PAIRS_PER_TERTILE, rng, fixed_tertile=tertile)
        audit = audit_position_mismatch_pairs(pmp)
        print(f"=== position_mismatch_pairs [{tertile}] audit ===", flush=True)
        print(json.dumps(audit, indent=2, default=str))
        assert audit["target_property_isolated"], f"position_mismatch_pairs[{tertile}] failed to isolate its target property"
        assert all(p["swap_position_tertile"] == tertile for p in pmp)
        pairs_by_tertile[tertile] = pmp
        audits_by_tertile[tertile] = audit

    tok2idx = get_tok2idx()

    cells = []
    for arch, seed, role in CELLS:
        print(f"\n--- {arch} seed{seed} ({role}) ---", flush=True)
        for tertile in TERTILES:
            pairs = pairs_by_tertile[tertile]
            results, ckpt = run_cell(arch, seed, pairs, tok2idx)
            cell = summarize(results, f"position_mismatch_{tertile}", arch, seed, role, ckpt)
            cells.append(cell)
            gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
            unreliable = cell["wiring_check_full_state_patch_SECONDARY"]["UNRELIABLE_near_zero_behavioral_gap"]
            print(f"  [{tertile}] PRIMARY behavioral gap={gap:.4f} (causally_effective={cell['causally_effective']})  |  "
                  f"wiring rf_mean={rf:.4f} n={cell['wiring_check_full_state_patch_SECONDARY']['n_nondegenerate_denom']} "
                  f"{'[UNRELIABLE]' if unreliable else ''}", flush=True)

    # ------------------------------------------------------------------
    # floor-effect prediction check
    # ------------------------------------------------------------------
    n_effective = sum(1 for c in cells if c["causally_effective"])
    n_total = len(cells)
    max_gap_cell = max(cells, key=lambda c: c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"])
    floor_effect_confirmed = n_effective == 0
    prediction_check = {
        "n_cells_total (8 architecture-seed pairs x 3 tertiles)": n_total,
        "n_cells_causally_effective (gap >= 0.05)": n_effective,
        "largest_observed_gap": {
            "arch": max_gap_cell["arch"], "seed": max_gap_cell["seed"], "role": max_gap_cell["role"],
            "pair_type": max_gap_cell["pair_type"],
            "gap": max_gap_cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"],
        },
        "floor_effect_prediction_confirmed": floor_effect_confirmed,
        "verdict": (
            "CONFIRMED -- near-zero behavioral gap across every cell and every swap-position tertile, "
            "matching the floor-effect prediction from the Condition 2 investigation. Missing-"
            "duplicate-string joins the marker-family and modular-arithmetic-simple pattern: "
            "target-computation (genuine duplicate-content verification) is ABSENT in the population "
            "where it is genuinely required. No architecture shows position-sensitivity on this task "
            "in the regime where positional tracking would actually matter -- consistent with all "
            "four architectures instead relying on the two trivial structural shortcuts "
            "(blank-count-is-one, and for RNN/LSTM, length-parity) documented in Condition 1/2."
            if floor_effect_confirmed else
            f"NOT CONFIRMED -- {n_effective}/{n_total} cells show a non-negligible gap despite the "
            "predicted degenerate-decision floor effect. This is a SURPRISING result requiring "
            "characterization, not smoothing over -- see the flagged_cells section below for the "
            "specific architecture/seed/tertile combinations and their raw logits."
        ),
    }
    print("\n=== floor-effect prediction check ===", flush=True)
    print(json.dumps(prediction_check, indent=2, default=str))

    flagged_cells = [
        {
            "arch": c["arch"], "seed": c["seed"], "role": c["role"], "pair_type": c["pair_type"],
            "gap": c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"],
            "rf_mean": c["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"],
        }
        for c in cells if c["causally_effective"]
    ]
    if flagged_cells:
        print("\n=== FLAGGED: cells with non-negligible gap despite floor-effect prediction ===", flush=True)
        print(json.dumps(flagged_cells, indent=2, default=str))

    out = {
        "condition": "condition_3_position_of_mismatch_null_test",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only, flagged UNRELIABLE when the behavioral gap is "
                 "near zero (< 0.05).",
        "description": (
            "Tests position-of-mismatch sensitivity: clean/corrupt pairs have IDENTICAL per-half "
            "symbol counts (aggregate route blind to the difference) but differ in which specific "
            "position-pair is mismatched, built via a same-half swap. Swept early/mid/late by swap "
            "position, N=25 PER TERTILE per cell (75 pairs per cell, 8 cells)."
        ),
        "floor_effect_prediction": (
            "Predicted near-zero gap across all cells/tertiles, since these pairs are drawn from the "
            "same genuinely-hard population (blank_count==1, even length) that Condition 2's "
            "investigation found all four architectures solve at floor (RNN/LSTM/Mamba) or chance "
            "(Transformer) accuracy, independent of patching."
        ),
        "n_half": N_HALF, "n_pairs_per_tertile": N_PAIRS_PER_TERTILE,
        "pair_construction_audits_by_tertile": audits_by_tertile,
        "sample_pairs_by_tertile": {t: pairs_by_tertile[t][:2] for t in TERTILES},
        "cells": cells,
        "prediction_check": prediction_check,
        "flagged_non_negligible_cells": flagged_cells,
    }
    out_path = RESULTS / "phase3_missdup_condition3.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
