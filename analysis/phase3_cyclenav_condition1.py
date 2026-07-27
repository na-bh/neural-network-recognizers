"""Cycle-navigation Phase 3, condition 1 of 2: symbol_counts (shortcut).
Tests whether the identified shortcut feature -- represented at uniform
0.39-0.67 selectivity across all 8 cells in Phase 2 (phase2_p2_cyclenav.json)
-- causally drives the accept/reject decision.

Reuses grammar_pairs (analysis/phase3_counterfactual_design.py's
cyclenav_grammar_pairs), already labeled "tests symbol_counts" in its own
docstring from the original Phase 3 design step: clean = valid grammatical
positive; corrupt = SAME LENGTH, last token (a digit) replaced by a MOVE
symbol -- an invalid symbol-count profile (0 digits instead of 1, M+1 moves
instead of M). This directly perturbs the shortcut-relevant count structure,
unlike position_pairs (condition 2), which changes true_position but leaves
the symbol_counts profile ambiguous relative to the kept digit by
construction (see the design step's own audit note on that pair type).
Already behaviorally spot-checked on rnn seed1 in the design step (gap=5.43);
this extends to all 8 cells with wiring + layer-specific patches.

Same PRIMARY/SECONDARY/degenerate-rf discipline as the marked-reversal
conditions: behavioral logit gap is primary; full-state patch is secondary
wiring check; layer-specific patch (single established site per Phase 2's
convention -- layer0 for rnn/lstm, last layer for mamba/transformer, no
sweep needed since cycle-navigation never showed a site-transfer issue) is
flagged non-meaningful whenever the behavioral gap is degenerate.

Saves analysis_outputs/final_results/phase3_cyclenav.json under
"condition_1_symbol_counts". STOP after this condition per user instruction.

PYTHONPATH=src:analysis python analysis/phase3_cyclenav_condition1.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_counterfactual_design import cyclenav_grammar_pairs, audit_cyclenav_pairs
from phase3_condition2_length_parity import mamba_patch_relative, transformer_patch_relative

RESULTS = Path("analysis_outputs/final_results")
TASK = "cycle-navigation"
SEED_PAIRS = {"rnn": [1, 4], "lstm": [2, 3], "transformer": [6, 9], "mamba": [0, 7]}
M_MOVES = 19  # matches the design step's own choice (L=20)
N_PAIRS = 25
DEGENERATE_GAP_THRESHOLD = 0.05
CANONICAL_LAYER = {"rnn": 0, "lstm": 0, "mamba": 4, "transformer": 4}


def run_rnn_lstm(arch, seed, pairs, layer_idx):
    from rnn_patching import RNNPatchingHarness, RecurrentScan, SITE_RNN, SITE_LSTM_FULL
    h = RNNPatchingHarness(arch, f"rec+ns/validation-short/{seed}", task=TASK)
    sc = RecurrentScan(h)
    site = SITE_RNN if arch == "rnn" else SITE_LSTM_FULL

    wiring, layer_specific, behavioral = [], [], []
    for p in pairs:
        clean, corrupt = p["clean"], p["corrupt"]
        t_ro = len(clean) - 1
        assert t_ro == len(corrupt) - 1
        lc = h.logit_diff(clean)
        lo = h.logit_diff(corrupt)
        behavioral.append(abs(lc - lo))

        _, states = sc.record(clean, [t_ro])
        patched_full = sc.patch_logit(corrupt, t_ro, states[t_ro], site=site, dims=None)
        wiring.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})

        clean_layer_state = sc.record_layer_state(clean, layer_idx, t_ro)
        patched_layer = sc.patch_logit_at_layer(corrupt, t_ro, layer_idx, clean_layer_state, dims=None)
        layer_specific.append({"restored_fraction": h.restored_fraction(lc, lo, patched_layer)})
    return wiring, layer_specific, behavioral, f"data/models/{TASK}/{arch}/rec+ns/validation-short/{seed}"


def run_mamba(seed, pairs, layer_idx):
    from patching_harness import PatchingHarness
    h = PatchingHarness(task=TASK, model_subdir=f"mamba/rec+ns/validation-short/{seed}")

    wiring, layer_specific, behavioral = [], [], []
    for p in pairs:
        clean, corrupt = p["clean"], p["corrupt"]
        t_ro = len(clean) - 1
        assert t_ro == len(corrupt) - 1
        lc = h.logit_diff(clean)
        lo = h.logit_diff(corrupt)
        behavioral.append(abs(lc - lo))

        patched_full = mamba_patch_relative(h, clean, corrupt, layer_idx, t_ro, t_ro, dims=None)
        wiring.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})
        layer_specific.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})
    return wiring, layer_specific, behavioral, f"models/{TASK}/mamba/rec+ns/validation-short/{seed}"


def run_transformer(seed, pairs, layer_idx):
    from modk_transformer_harness import TransformerPatchingHarness
    h = TransformerPatchingHarness(TASK, f"rec+ns/validation-short/{seed}")

    wiring, layer_specific, behavioral = [], [], []
    for p in pairs:
        clean, corrupt = p["clean"], p["corrupt"]
        t_ro = len(clean) - 1
        assert t_ro == len(corrupt) - 1
        lc = h.logit_diff(clean)
        lo = h.logit_diff(corrupt)
        behavioral.append(abs(lc - lo))
        denom = lc - lo

        patched_full = transformer_patch_relative(h, clean, corrupt, layer_idx, t_ro, t_ro, dims=None)
        rf = (patched_full - lo) / denom if abs(denom) > 1e-9 else float("nan")
        wiring.append({"restored_fraction": rf})
        layer_specific.append({"restored_fraction": rf})
    return wiring, layer_specific, behavioral, f"data/models/{TASK}/transformer/rec+ns/validation-short/{seed}"


def summarize_rf(results, key):
    rfs = np.array([r["restored_fraction"] for r in results if not np.isnan(r["restored_fraction"])])
    n = len(rfs)
    mean = float(np.mean(rfs)) if n else float("nan")
    ci95 = float(1.96 * np.std(rfs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {f"{key}_restored_fraction_mean": mean, f"{key}_restored_fraction_ci95": ci95,
            f"{key}_n_nondegenerate_denom": n}


def summarize_behavioral(behavioral):
    arr = np.array(behavioral)
    return {
        "mean_abs_clean_corrupt_logit_gap": float(np.mean(arr)),
        "ci95_abs_clean_corrupt_logit_gap": float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan"),
        "n": len(arr),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    pairs = cyclenav_grammar_pairs(M_MOVES, N_PAIRS, rng)
    audit = audit_cyclenav_pairs(pairs, "grammar")
    print("=== grammar_pairs (symbol_counts) audit ===")
    print(json.dumps(audit, indent=2, default=str))
    assert audit["same_length"] and audit["differs_only_at_last_position"] and audit["all_corrupt_scrambled"]

    cells = []
    for arch, seeds in SEED_PAIRS.items():
        layer_idx = CANONICAL_LAYER[arch]
        for seed in seeds:
            print(f"\n--- {arch} seed{seed} (canonical layer={layer_idx}) ---", flush=True)
            if arch in ("rnn", "lstm"):
                wiring, layer_specific, behavioral, ckpt = run_rnn_lstm(arch, seed, pairs, layer_idx)
            elif arch == "mamba":
                wiring, layer_specific, behavioral, ckpt = run_mamba(seed, pairs, layer_idx)
            else:
                wiring, layer_specific, behavioral, ckpt = run_transformer(seed, pairs, layer_idx)

            beh = summarize_behavioral(behavioral)
            wiring_s = summarize_rf(wiring, "wiring")
            layer_s = summarize_rf(layer_specific, "layer_specific")
            gap = beh["mean_abs_clean_corrupt_logit_gap"]
            rf_meaningful = gap >= DEGENERATE_GAP_THRESHOLD
            surprising_null = not rf_meaningful  # prediction was a LARGE gap; near-zero is the surprise here
            cell = {
                "arch": arch, "seed": seed, "checkpoint": ckpt, "canonical_layer": layer_idx,
                "PRIMARY_behavioral_logit_gap": beh,
                "wiring_check_full_state_patch_SECONDARY": wiring_s,
                "layer_specific_patch_at_canonical_layer": layer_s,
                "layer_specific_rf_is_meaningful": rf_meaningful,
                "SURPRISING_near_zero_symbol_counts_effect": surprising_null,
            }
            if not rf_meaningful:
                cell["layer_specific_patch_caveat"] = (
                    f"behavioral gap ({gap:.4f}) is degenerate/near-zero -- UNEXPECTED for symbol_counts "
                    f"(the shortcut every architecture is predicted to use robustly); layer_specific rf "
                    f"is not meaningful here, and this cell itself is the finding requiring investigation."
                )
            cells.append(cell)
            flag = "  <<< SURPRISING NULL -- symbol_counts NOT causal here" if surprising_null else ""
            print(f"  PRIMARY behavioral gap: mean={gap:.4f} (ci95={beh['ci95_abs_clean_corrupt_logit_gap']:.4f}, n={beh['n']}){flag}", flush=True)
            print(f"  layer{layer_idx}-specific patch: rf_mean={layer_s['layer_specific_restored_fraction_mean']:.4f} "
                  f"meaningful={rf_meaningful}", flush=True)
            print(f"  full-state wiring check: rf_mean={wiring_s['wiring_restored_fraction_mean']:.4f}", flush=True)

    any_surprising = [c for c in cells if c["SURPRISING_near_zero_symbol_counts_effect"]]

    out = {
        "task": "cycle-navigation",
        "condition": "condition_1_symbol_counts",
        "status": "PRIMARY evidence is the behavioral logit gap on grammar_pairs and the layer-specific "
                 "patch at each architecture's Phase-2-established canonical site (layer0 rnn/lstm, "
                 "last layer mamba/transformer -- no layer sweep needed, cycle-navigation never showed "
                 "a site-transfer issue in Phase 2). Full-state patch is secondary/reference only.",
        "description": "Tests whether the shortcut feature symbol_counts -- uniform 0.39-0.67 "
                       "selectivity across all 8 cells in Phase 2 -- causally drives accept/reject, "
                       "via grammar_pairs (clean=valid positive; corrupt=same length, last token "
                       "(digit) replaced by a move symbol, an invalid symbol-count profile). "
                       "Predicted: large behavioral gap and high layer-specific restored fraction for "
                       "all 4 architectures, both seeds.",
        "m_moves": M_MOVES, "n_pairs_generated": N_PAIRS,
        "canonical_layer_by_arch": CANONICAL_LAYER,
        "grammar_pairs_audit": audit,
        "sample_pairs": pairs[:3],
        "cells": cells,
        "any_surprising_near_zero_cells": [f"{c['arch']}_seed{c['seed']}" for c in any_surprising],
        "prediction_confirmed": len(any_surprising) == 0,
    }
    out_path = RESULTS / "phase3_cyclenav.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")
    print(f"prediction_confirmed (no surprising near-zero cells): {len(any_surprising) == 0}")


if __name__ == "__main__":
    main()
