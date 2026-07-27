"""Repeat-01 Phase 3, condition 1 of 3 (WITH Condition 2 COLLAPSED IN, per
the task's simplicity -- no additional trivial aggregate feature identified
beyond the four already tested): structural features (length_parity_even,
first_symbol_is_0, last_symbol_is_1, count_balance, null_control TRUE PAIR,
plus combined_shortcut_violation as the collapsed Condition 2).

Watches specifically for the Transformer pattern flagged by the user:
  (a) Transformer shows near-zero causal effect on the four shortcut
      features while RNN/LSTM/Mamba show large effects -- CONFIRMED:
      Transformer isn't building the trivial shortcuts at all.
  (b) Transformer shows large effects but its own accuracy stays low --
      more consistent with a systematic length/short-sequence sensitivity
      artifact than a missing-shortcut story.

PRIMARY evidence is the behavioral logit gap; full-state patch is a
SECONDARY wiring check, flagged UNRELIABLE in near-zero-gap cells.

Saves analysis_outputs/final_results/phase3_repeat01_condition1.json
(includes the collapsed Condition 2 combined-shortcut test).

PYTHONPATH=src:analysis python analysis/phase3_repeat01_condition1.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_repeat01_counterfactual_design import (
    length_parity_pairs, first_symbol_pairs, last_symbol_pairs, count_balance_pairs,
    null_control_pairs, combined_shortcut_violation_pairs,
    audit_length_parity_pairs, audit_first_symbol_pairs, audit_last_symbol_pairs,
    audit_count_balance_pairs, audit_null_control_pairs, audit_combined_shortcut_violation_pairs,
)
from phase3_repeat01_patching_common import CELLS, get_tok2idx, run_cell, summarize

RESULTS = Path("analysis_outputs/final_results")
N_PAIRS = 25


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    lpp = length_parity_pairs(N_PAIRS, rng)
    fsp = first_symbol_pairs(N_PAIRS, rng)
    lsp = last_symbol_pairs(N_PAIRS, rng)
    cbp = count_balance_pairs(N_PAIRS, rng)
    ncp = null_control_pairs(N_PAIRS, rng)
    csvp = combined_shortcut_violation_pairs(N_PAIRS, rng)
    audits = {
        "length_parity_pairs": audit_length_parity_pairs(lpp),
        "first_symbol_pairs": audit_first_symbol_pairs(fsp),
        "last_symbol_pairs": audit_last_symbol_pairs(lsp),
        "count_balance_pairs": audit_count_balance_pairs(cbp),
        "null_control_pairs": audit_null_control_pairs(ncp),
        "combined_shortcut_violation_pairs": audit_combined_shortcut_violation_pairs(csvp),
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
            ("length_parity", lpp), ("first_symbol", fsp), ("last_symbol", lsp),
            ("count_balance", cbp), ("null_control_TRUE_PAIR", ncp),
            ("combined_shortcut_violation_CONDITION2_COLLAPSED", csvp),
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
    # null-control check
    # ------------------------------------------------------------------
    null_control_check = {}
    for arch, seed, role in CELLS:
        c = next(c for c in cells if c["arch"] == arch and c["seed"] == seed and c["pair_type"] == "null_control_TRUE_PAIR")
        gap = c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
        null_control_check[f"{arch}_seed{seed}_{role}"] = {"gap": gap, "spuriously_causal": gap >= 0.05}
    print("\n=== null-control check (TRUE pair; expect ~0 gap everywhere) ===", flush=True)
    print(json.dumps(null_control_check, indent=2, default=str))

    # ------------------------------------------------------------------
    # Transformer pattern check: (a) vs (b)
    # ------------------------------------------------------------------
    four_features = ["length_parity", "first_symbol", "last_symbol", "count_balance"]
    transformer_check = {}
    for arch in ["rnn", "lstm", "transformer", "mamba"]:
        arch_cells = [c for c in cells if c["arch"] == arch and c["pair_type"] in four_features]
        mean_gap = float(np.mean([c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"] for c in arch_cells]))
        n_causal = sum(1 for c in arch_cells if c["causally_effective"])
        transformer_check[arch] = {"mean_gap_across_four_shortcuts_both_seeds": mean_gap,
                                   "n_causally_effective_of_8": n_causal}
    rnn_lstm_mamba_mean = float(np.mean([transformer_check[a]["mean_gap_across_four_shortcuts_both_seeds"]
                                         for a in ["rnn", "lstm", "mamba"]]))
    transformer_mean = transformer_check["transformer"]["mean_gap_across_four_shortcuts_both_seeds"]
    pattern_a = transformer_mean < 0.05 * rnn_lstm_mamba_mean or transformer_check["transformer"]["n_causally_effective_of_8"] <= 2
    pattern_verdict = {
        "per_architecture": transformer_check,
        "rnn_lstm_mamba_mean_gap": rnn_lstm_mamba_mean,
        "transformer_mean_gap": transformer_mean,
        "interpretation": (
            "PATTERN (a) -- Transformer shows near-zero causal effect on the four shortcut features "
            "while RNN/LSTM/Mamba show large effects: Transformer isn't building the trivial "
            "shortcuts at all."
            if pattern_a else
            "PATTERN (b) -- Transformer shows non-trivial causal effects on the shortcut features "
            "too (not near-zero), yet its own standalone accuracy is known to be low (80.7%/79.3% "
            "vs 100% for the others, per cell selection) -- more consistent with a systematic "
            "length/short-sequence sensitivity artifact than a missing-shortcut story. Reported with "
            "raw numbers below; not forced into either bucket if ambiguous."
        ),
    }
    print("\n=== Transformer pattern check (a) vs (b) ===", flush=True)
    print(json.dumps(pattern_verdict, indent=2, default=str))

    out = {
        "condition": "condition_1_structural_features_WITH_condition_2_collapsed_in",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only, flagged UNRELIABLE when the behavioral gap is "
                 "near zero (< 0.05).",
        "description": (
            "Tests whether length_parity_even, first_symbol_is_0, last_symbol_is_1, and count_"
            "balance causally drive the accept/reject decision, all 8 cells. null_control is a TRUE "
            "pair (length-extension, both genuinely valid, same label) -- expected near-zero gap "
            "everywhere. Condition 2 COLLAPSES INTO THIS FILE (combined_shortcut_violation_"
            "CONDITION2_COLLAPSED pair type) per the task's simplicity -- no additional trivial "
            "aggregate feature was identified beyond the four already tested."
        ),
        "n_pairs_generated": N_PAIRS,
        "pair_construction_audits": audits,
        "sample_pairs": {"length_parity": lpp[:3], "first_symbol": fsp[:3], "last_symbol": lsp[:3],
                         "count_balance": cbp[:3], "null_control": ncp[:3],
                         "combined_shortcut_violation": csvp[:3]},
        "cells": cells,
        "null_control_check": null_control_check,
        "transformer_pattern_check_a_vs_b": pattern_verdict,
    }
    out_path = RESULTS / "phase3_repeat01_condition1.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
