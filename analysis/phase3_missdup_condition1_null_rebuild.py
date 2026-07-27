"""Missing-duplicate-string Phase 3, condition 1 null-control REBUILD: the
original first_last_symbol_pairs null control (phase3_missdup_condition1.json)
independently resampled clean/corrupt, confounding any observed gap with
unrelated content differences (flagged, not smoothed over). Rebuilt here as
a TRUE 2-position-flip minimal pair: flip position 0 and its mirror
(half+0) simultaneously, holding every other position (incl. the blank)
identical -- both members remain genuinely valid positives with the SAME
true label.

PRIMARY evidence is the behavioral logit gap; near-zero gap = clean null
baseline; a genuine gap despite the true minimal-pair construction is a real
finding of spurious first-symbol dependence.

Saves analysis_outputs/final_results/phase3_missdup_condition1_null_rebuild.json.

PYTHONPATH=src:analysis python analysis/phase3_missdup_condition1_null_rebuild.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_missdup_counterfactual_design import (
    null_control_rebuild_pairs, audit_null_control_rebuild_pairs,
)
from phase3_missdup_patching_common import CELLS, get_tok2idx, run_cell, summarize

RESULTS = Path("analysis_outputs/final_results")
N_HALF = 40
N_PAIRS = 25


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)

    ncp = null_control_rebuild_pairs(N_HALF, N_PAIRS, rng)
    audit = audit_null_control_rebuild_pairs(ncp)
    print("=== null_control_rebuild_pairs audit ===", flush=True)
    print(json.dumps(audit, indent=2, default=str))
    assert audit["target_property_isolated"], "null_control_rebuild_pairs failed to isolate its target property"

    tok2idx = get_tok2idx()

    cells = []
    for arch, seed, role in CELLS:
        results, ckpt = run_cell(arch, seed, ncp, tok2idx)
        cell = summarize(results, "first_last_symbol_TRUE_MINIMAL_PAIR", arch, seed, role, ckpt)
        cells.append(cell)
        gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
        rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
        print(f"{arch} seed{seed} ({role}): gap={gap:.4f} causally_effective={cell['causally_effective']} "
              f"rf_mean={rf:.4f}", flush=True)

    n_spurious = sum(1 for c in cells if c["causally_effective"])
    verdict = {
        "n_cells_total": len(cells), "n_cells_with_genuine_first_symbol_gap": n_spurious,
        "clean_null_baseline": n_spurious == 0,
        "interpretation": (
            "CLEAN NULL BASELINE -- no cell shows a genuine gap under the true minimal-pair "
            "construction; the original null control's nonzero gaps (4/8 cells) were an artifact "
            "of independently-resampled content, not genuine first-symbol dependence."
            if n_spurious == 0 else
            f"{n_spurious}/8 cells show a GENUINE gap despite the true minimal-pair construction -- "
            "this is a real finding of spurious first-symbol dependence, not a construction artifact. "
            "Reported per-cell below, not smoothed over."
        ),
    }
    print("\n=== verdict ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str))

    out = {
        "condition": "condition_1_null_control_REBUILD",
        "description": (
            "Rebuild of the first_last_symbol null control from phase3_missdup_condition1.json, "
            "which used independently-resampled clean/corrupt content (a construction flaw flagged "
            "after that run, not a genuine finding). This version flips position 0 and its mirror "
            "(half+0) simultaneously, holding every other position (incl. the blank) identical -- a "
            "TRUE minimal pair, both members genuinely valid with the same true label."
        ),
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "pair_construction_audit": audit,
        "sample_pairs": ncp[:3],
        "cells": cells,
        "verdict": verdict,
    }
    out_path = RESULTS / "phase3_missdup_condition1_null_rebuild.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
