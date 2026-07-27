"""Missing-duplicate-string Phase 3, condition 2 of 3: per-half symbol-count
features (the aggregate-route CRITICAL TEST). clean=valid positive (per-half
counts of '1's match by construction). corrupt=SAME length, SAME blank
position, ONE non-blank bit flipped in the half not containing the blank --
breaks the per-half count match (aggregate mismatch) AND the local
positional match at that index simultaneously (an unavoidable coupling for
a binary alphabet, documented in phase3_missdup_counterfactual_design.json).

PRIMARY evidence is the behavioral logit gap; full-state patch is a
SECONDARY wiring check, flagged UNRELIABLE in near-zero-gap cells.

Reported explicitly against the flare_a2-derived prediction (phase3_missdup_
cell_selection.json's architecture_wide_route_hint): Transformer/Mamba
(large hard-vs-scrambled accuracy gap, ~0.62-0.64) predicted to show LARGE
Condition 2 effects; RNN/LSTM (~0 gap, comparable hard/scrambled accuracy)
predicted to show MODEST effects.

Saves analysis_outputs/final_results/phase3_missdup_condition2.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_missdup_condition2.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_missdup_counterfactual_design import count_mismatch_pairs, audit_count_mismatch_pairs
from phase3_missdup_patching_common import CELLS, get_tok2idx, run_cell, summarize, load_cell_selection_context

RESULTS = Path("analysis_outputs/final_results")
N_HALF = 40
N_PAIRS = 25
PREDICTED_LARGE_EFFECT_ARCHS = {"transformer", "mamba"}
PREDICTED_MODEST_EFFECT_ARCHS = {"rnn", "lstm"}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)

    cmp_ = count_mismatch_pairs(N_HALF, N_PAIRS, rng)
    audit = audit_count_mismatch_pairs(cmp_)
    print("=== count_mismatch_pairs audit ===", flush=True)
    print(json.dumps(audit, indent=2, default=str))
    assert audit["target_property_isolated"], "count_mismatch_pairs failed to isolate its target property"

    tok2idx = get_tok2idx()
    _, route_hint = load_cell_selection_context()

    cells = []
    for arch, seed, role in CELLS:
        results, ckpt = run_cell(arch, seed, cmp_, tok2idx)
        cell = summarize(results, "count_mismatch", arch, seed, role, ckpt)
        cells.append(cell)
        gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
        rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
        unreliable = cell["wiring_check_full_state_patch_SECONDARY"]["UNRELIABLE_near_zero_behavioral_gap"]
        print(f"{arch} seed{seed} ({role}): gap={gap:.4f} causally_effective={cell['causally_effective']} "
              f"rf_mean={rf:.4f} {'[UNRELIABLE]' if unreliable else ''}", flush=True)

    # ------------------------------------------------------------------
    # explicit report against the flare_a2-derived prediction
    # ------------------------------------------------------------------
    per_arch_gap = {}
    for arch in ["rnn", "lstm", "transformer", "mamba"]:
        arch_cells = [c for c in cells if c["arch"] == arch]
        gaps = [c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"] for c in arch_cells]
        per_arch_gap[arch] = {
            "primary_gap": arch_cells[0]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"],
            "secondary_gap": arch_cells[1]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"],
            "mean_gap": float(np.mean(gaps)),
            "predicted_route_hint": route_hint.get(arch, {}).get("predicted_condition2_vs_condition3"),
            "flagged_aggregate_route_suspect": route_hint.get(arch, {}).get("flagged_aggregate_route_suspect"),
        }
    print("\n=== per-architecture Condition 2 gap vs. flare_a2 prediction ===", flush=True)
    print(json.dumps(per_arch_gap, indent=2, default=str))

    predicted_large_mean = float(np.mean([per_arch_gap[a]["mean_gap"] for a in PREDICTED_LARGE_EFFECT_ARCHS]))
    predicted_modest_mean = float(np.mean([per_arch_gap[a]["mean_gap"] for a in PREDICTED_MODEST_EFFECT_ARCHS]))
    split_direction_confirmed = predicted_large_mean > predicted_modest_mean
    # per-architecture confirmation: every predicted-large arch causally effective,
    # AND no predicted-modest arch shows an equally-large (or larger) effect
    all_large_effective = all(
        any(c["arch"] == a and c["causally_effective"] for c in cells) for a in PREDICTED_LARGE_EFFECT_ARCHS
    )
    any_modest_shows_large_too = any(
        per_arch_gap[a]["mean_gap"] >= min(per_arch_gap[b]["mean_gap"] for b in PREDICTED_LARGE_EFFECT_ARCHS)
        for a in PREDICTED_MODEST_EFFECT_ARCHS
    )
    prediction_check = {
        "predicted_large_effect_archs": sorted(PREDICTED_LARGE_EFFECT_ARCHS),
        "predicted_modest_effect_archs": sorted(PREDICTED_MODEST_EFFECT_ARCHS),
        "predicted_large_archs_mean_gap": predicted_large_mean,
        "predicted_modest_archs_mean_gap": predicted_modest_mean,
        "split_direction_confirmed (large > modest, aggregate)": split_direction_confirmed,
        "all_predicted_large_archs_causally_effective": all_large_effective,
        "a_predicted_modest_arch_matches_or_exceeds_smallest_large_arch_gap (would undercut the split)": any_modest_shows_large_too,
        "verdict": (
            "CONFIRMED -- Transformer/Mamba show substantially larger Condition 2 (aggregate-route) "
            "causal effects than RNN/LSTM, matching the flare_a2 hard-vs-scrambled-accuracy-derived "
            "prediction. Consistent with a genuine architectural split: Transformer/Mamba rely on "
            "shallow per-half symbol counting, RNN/LSTM do not (or not as heavily)."
            if split_direction_confirmed and all_large_effective and not any_modest_shows_large_too else
            "NOT CLEANLY CONFIRMED -- reported per-architecture numbers above; investigate rather than "
            "assume the predicted split holds. Check whether a predicted-modest architecture (RNN/"
            "LSTM) shows an unexpectedly large effect, or whether a predicted-large architecture "
            "(Transformer/Mamba) fails to show the expected effect, per-cell."
        ),
    }
    print("\n=== prediction check ===", flush=True)
    print(json.dumps(prediction_check, indent=2, default=str))

    out = {
        "condition": "condition_2_aggregate_route_critical_test",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only, flagged UNRELIABLE when the behavioral gap is "
                 "near zero (< 0.05).",
        "description": (
            "Tests whether per-half symbol-count mismatch (the aggregate route) causally drives the "
            "accept/reject decision, all 8 cells. Coupling caveat: for this binary alphabet, the "
            "single-bit-flip construction necessarily changes local positional match too (see "
            "count_mismatch_pairs' audit note) -- a genuine causal effect here is consistent with "
            "(not exclusive proof of) aggregate-route reliance; Condition 3 isolates position-only "
            "sensitivity by holding the aggregate feature fixed instead."
        ),
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "pair_construction_audit": audit,
        "sample_pairs": cmp_[:3],
        "cells": cells,
        "per_architecture_gap_vs_prediction": per_arch_gap,
        "prediction_check": prediction_check,
    }
    out_path = RESULTS / "phase3_missdup_condition2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
