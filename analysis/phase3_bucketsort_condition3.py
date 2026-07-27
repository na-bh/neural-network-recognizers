"""Bucket-sort Phase 3, condition 3 of 3 (confirmatory null test):
target_computation_multiset_matched_case. Tests whether genuine sort-ORDER
verification -- held marker_count/position/is_balanced/is_multiset_matched
constant, only ORDER varied -- causally drives the accept/reject decision,
all 8 cells.

Uses target_computation_pairs (phase3_bucketsort_counterfactuals.py): clean
= w#sorted(w); corrupt = same length/marker/balance/multiset, two adjacent
sorted-half positions (fixed at the middle of the half, away from both ends
to avoid reintroducing the max-check/min-check trivial pathways from Phase 1
P2) swapped. Construction audited BEFORE this condition ran -- see
phase3_bucketsort_counterfactual_design.json's target_computation_pairs
entry: all confound checks (including the max/min-check re-verification)
passed cleanly.

NOTE: unlike conditions 1-2 (and unlike the target-computation null tests
for the three prior tasks), Phase 2 did NOT run a probe layer sweep for
this target on bucket-sort (deferred to Phase 3 per the sparse-natural-
negatives finding, phase1_bucketsort_baseline_design.json). There is
therefore no probe-informed "best layer" to test a layer-specific patch
against -- only the PRIMARY behavioral gap and the SECONDARY full-state
wiring check are reported here, not a third layer-specific patch.

Predicted: near-zero behavioral gap across all 8 cells, matching all three
prior marker-family tasks' target-computation null result -- though given
condition 2's is_multiset_matched surprise (probe drastically underestimated
a huge, near-universal causal effect), this prediction is tested directly,
not assumed to hold just because the pattern held before.

Saves analysis_outputs/final_results/phase3_bucketsort_condition3.json.

PYTHONPATH=src:analysis python analysis/phase3_bucketsort_condition3.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_bucketsort_counterfactuals import target_computation_pairs, audit_target_computation_pairs

RESULTS = Path("analysis_outputs/final_results")
TASK = "bucket-sort"
CELLS = [
    ("rnn", 4, "primary"), ("rnn", 10, "secondary_comparable"),
    ("lstm", 2, "primary"), ("lstm", 3, "secondary_near_chance_partial"),
    ("transformer", 7, "primary"), ("transformer", 3, "secondary_comparable"),
    ("mamba", 7, "primary"), ("mamba", 3, "secondary_comparable"),
]
N_HALF = 50
N_PAIRS = 25
DEGENERATE_GAP_THRESHOLD = 0.05


def run_rnn_lstm(arch, seed, pairs, tok2idx):
    from rnn_patching import RNNPatchingHarness, RecurrentScan, SITE_RNN, SITE_LSTM_FULL
    h = RNNPatchingHarness(arch, f"rec+ns/validation-short/{seed}", task=TASK)
    sc = RecurrentScan(h)
    site = SITE_RNN if arch == "rnn" else SITE_LSTM_FULL

    wiring, behavioral = [], []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = h.readout_position(p["clean"])
        assert t_ro == h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))

        _, states = sc.record(clean_idx, [t_ro])
        patched_full = sc.patch_logit(corrupt_idx, t_ro, states[t_ro], site=site, dims=None)
        wiring.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})
    return wiring, behavioral, f"data/models/{TASK}/{arch}/rec+ns/validation-short/{seed}"


def run_mamba(seed, pairs, tok2idx):
    from patching_harness import PatchingHarness
    h = PatchingHarness(task=TASK, model_subdir=f"mamba/rec+ns/validation-short/{seed}")
    top_layer = h.num_layers - 1

    wiring, behavioral = [], []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = h.readout_position(p["clean"])
        assert t_ro == h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))

        _, clean_cache = h.run_with_cache(clean_idx, layers=[top_layer])
        patched_full = h.patch_run(clean_idx, corrupt_idx, layer=top_layer, position=t_ro, dims=None,
                                   clean_cache=clean_cache)
        wiring.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})
    return wiring, behavioral, f"models/{TASK}/mamba/rec+ns/validation-short/{seed}"


def run_transformer(seed, pairs, tok2idx):
    from modk_transformer_harness import TransformerPatchingHarness
    h = TransformerPatchingHarness(TASK, f"rec+ns/validation-short/{seed}")
    layer = h.num_layers - 1

    wiring, behavioral = [], []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = len(p["clean"]) - 1
        assert t_ro == len(p["corrupt"]) - 1
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))

        _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
        clean_row = clean_cache[layer][t_ro + h.bos_off]
        patched_full = h.patch_logit(corrupt_idx, layer, t_ro, clean_row, dims=None)
        rf = (patched_full - lo) / (lc - lo) if abs(lc - lo) > 1e-9 else float("nan")
        wiring.append({"restored_fraction": rf})
    return wiring, behavioral, f"data/models/{TASK}/transformer/rec+ns/validation-short/{seed}"


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

    pairs = target_computation_pairs(N_HALF, N_PAIRS, rng)
    audit = audit_target_computation_pairs(pairs)
    print("=== target_computation_pairs audit (construction report) ===")
    print(json.dumps(audit, indent=2))
    assert audit["target_property_isolated"]
    print(f"\nsample clean : {''.join(pairs[0]['clean'])}")
    print(f"sample corrupt: {''.join(pairs[0]['corrupt'])}")
    print(f"swap_positions: {pairs[0]['swap_positions']}\n")

    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    cells = []
    for arch, seed, role in CELLS:
        print(f"\n--- {arch} seed{seed} ({role}) ---", flush=True)
        if arch in ("rnn", "lstm"):
            wiring, behavioral, ckpt = run_rnn_lstm(arch, seed, pairs, tok2idx)
        elif arch == "mamba":
            wiring, behavioral, ckpt = run_mamba(seed, pairs, tok2idx)
        else:
            wiring, behavioral, ckpt = run_transformer(seed, pairs, tok2idx)

        beh = summarize_behavioral(behavioral)
        wiring_s = summarize_rf(wiring, "wiring")
        gap = beh["mean_abs_clean_corrupt_logit_gap"]
        surprising = gap >= DEGENERATE_GAP_THRESHOLD
        cell = {
            "arch": arch, "seed": seed, "role": role, "checkpoint": ckpt,
            "PRIMARY_behavioral_logit_gap": beh,
            "wiring_check_full_state_patch_SECONDARY": wiring_s,
            "SURPRISING_above_noise_sort_order_sensitivity": surprising,
            "note": "no Phase 2 probe layer sweep exists for this target (deferred) -- only "
                   "full-state wiring check reported, no layer-specific patch.",
        }
        cells.append(cell)
        flag = "  <<< SURPRISING -- ABOVE NOISE" if surprising else ""
        print(f"  PRIMARY behavioral gap: mean={gap:.4f} (ci95={beh['ci95_abs_clean_corrupt_logit_gap']:.4f}, n={beh['n']}){flag}", flush=True)
        print(f"  full-state wiring check (secondary): rf_mean={wiring_s['wiring_restored_fraction_mean']:.4f}", flush=True)

    any_surprising = [c for c in cells if c["SURPRISING_above_noise_sort_order_sensitivity"]]
    all_null = len(any_surprising) == 0

    out = {
        "condition": "condition_3_target_computation_null_test",
        "status": "CONFIRMATORY null test, all 8 cells. PRIMARY evidence is the behavioral logit "
                 "gap; full-state wiring check is secondary/reference only. No layer-specific patch "
                 "(no Phase 2 probe layer sweep exists for this deferred target).",
        "description": "Tests genuine sort-ORDER verification via target_computation_pairs (two "
                       "adjacent post-marker positions swapped, marker_count/position/is_balanced/"
                       "is_multiset_matched all held constant) for all 8 cells. Predicted: near-zero "
                       "effect everywhere, matching all three prior marker-family tasks -- tested "
                       "directly given condition 2's is_multiset_matched surprise, not assumed.",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "target_computation_pairs_construction_audit": audit,
        "sample_pairs": pairs[:3],
        "cells": cells,
        "all_cells_null": all_null,
        "surprising_cells": any_surprising,
    }
    out_path = RESULTS / "phase3_bucketsort_condition3.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")
    print(f"\nALL CELLS NULL: {all_null}")


if __name__ == "__main__":
    main()
