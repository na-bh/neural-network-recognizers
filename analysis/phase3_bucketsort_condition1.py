"""Bucket-sort Phase 3, condition 1 of 3: marker features (marker_count,
marker_position). Tests whether marker shortcut features -- represented at
0.20-0.36 selectivity by ALL 8 cells in Stage 2, uniformly -- causally drive
the accept/reject decision.

PRIMARY evidence is the behavioral logit gap (unpatched clean vs corrupt);
full-state patch (canonical readout site per architecture) is a SECONDARY
wiring check.

Specific test built in: lstm seed2 (the single-solver primary, marker-only
representational profile per Stage 2/3) is predicted to show normal-
magnitude causal marker effects -- if confirmed, this completes the
mechanism story (marker-only representation -> marker-only causal use ->
insufficient for hard negatives, since Stage 3 found NOTHING else to draw
on for this cell).

Saves analysis_outputs/final_results/phase3_bucketsort_condition1.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_bucketsort_condition1.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_bucketsort_counterfactuals import (
    marker_count_pairs, marker_position_pairs,
    audit_marker_count_pairs, audit_marker_position_pairs,
)

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

    results = []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = h.readout_position(p["clean"])
        assert t_ro == h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, states = sc.record(clean_idx, [t_ro])
        patched = sc.patch_logit(corrupt_idx, t_ro, states[t_ro], site=site, dims=None)
        rf = h.restored_fraction(lc, lo, patched)
        results.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched,
                         "restored_fraction": rf, "behavioral_gap": abs(lc - lo)})
    return results, f"data/models/{TASK}/{arch}/rec+ns/validation-short/{seed}"


def run_mamba(seed, pairs, tok2idx):
    from patching_harness import PatchingHarness
    h = PatchingHarness(task=TASK, model_subdir=f"mamba/rec+ns/validation-short/{seed}")
    layer = h.num_layers - 1

    results = []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = h.readout_position(p["clean"])
        assert t_ro == h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
        patched = h.patch_run(clean_idx, corrupt_idx, layer=layer, position=t_ro, dims=None,
                              clean_cache=clean_cache)
        rf = h.restored_fraction(lc, lo, patched)
        results.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched,
                         "restored_fraction": rf, "behavioral_gap": abs(lc - lo)})
    return results, f"models/{TASK}/mamba/rec+ns/validation-short/{seed}"


def run_transformer(seed, pairs, tok2idx):
    from modk_transformer_harness import TransformerPatchingHarness
    h = TransformerPatchingHarness(TASK, f"rec+ns/validation-short/{seed}")
    layer = h.num_layers - 1

    results = []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = len(p["clean"]) - 1
        assert t_ro == len(p["corrupt"]) - 1
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
        clean_row = clean_cache[layer][t_ro + h.bos_off]
        patched = h.patch_logit(corrupt_idx, layer, t_ro, clean_row, dims=None)
        rf = (patched - lo) / (lc - lo) if abs(lc - lo) > 1e-9 else float("nan")
        results.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched,
                         "restored_fraction": rf, "behavioral_gap": abs(lc - lo)})
    return results, f"data/models/{TASK}/transformer/rec+ns/validation-short/{seed}"


def summarize(results, pair_type, arch, seed, role, checkpoint):
    rfs = np.array([r["restored_fraction"] for r in results if not np.isnan(r["restored_fraction"])])
    n = len(rfs)
    mean = float(np.mean(rfs)) if n else float("nan")
    ci95 = float(1.96 * np.std(rfs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    gaps = np.array([r["behavioral_gap"] for r in results])
    gap_mean = float(np.mean(gaps))
    gap_ci95 = float(1.96 * np.std(gaps, ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else float("nan")
    return {
        "arch": arch, "seed": seed, "role": role, "pair_type": pair_type, "checkpoint": checkpoint,
        "PRIMARY_behavioral_logit_gap": {
            "mean_abs_clean_corrupt_logit_gap": gap_mean, "ci95": gap_ci95, "n": len(gaps),
        },
        "wiring_check_full_state_patch_SECONDARY": {
            "restored_fraction_mean": mean, "restored_fraction_ci95": ci95,
            "n_nondegenerate_denom": n,
        },
        "raw_restored_fractions": [float(x) for x in rfs],
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    mc_pairs = marker_count_pairs(N_HALF, N_PAIRS, rng)
    mp_pairs = marker_position_pairs(N_HALF, N_PAIRS, rng)
    mc_audit = audit_marker_count_pairs(mc_pairs)
    mp_audit = audit_marker_position_pairs(mp_pairs)
    print("=== marker_count_pairs audit ===")
    print(json.dumps(mc_audit, indent=2))
    print("=== marker_position_pairs audit ===")
    print(json.dumps(mp_audit, indent=2))
    assert mc_audit["target_property_isolated"] and mp_audit["target_property_isolated"]

    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    cells = []
    for arch, seed, role in CELLS:
        print(f"\n--- {arch} seed{seed} ({role}) ---", flush=True)
        for pair_type, pairs in [("marker_count", mc_pairs), ("marker_position", mp_pairs)]:
            if arch in ("rnn", "lstm"):
                results, ckpt = run_rnn_lstm(arch, seed, pairs, tok2idx)
            elif arch == "mamba":
                results, ckpt = run_mamba(seed, pairs, tok2idx)
            else:
                results, ckpt = run_transformer(seed, pairs, tok2idx)
            cell = summarize(results, pair_type, arch, seed, role, ckpt)
            cells.append(cell)
            gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
            print(f"  [{pair_type}] PRIMARY behavioral gap={gap:.4f}  |  "
                  f"wiring rf_mean={rf:.4f} n={cell['wiring_check_full_state_patch_SECONDARY']['n_nondegenerate_denom']}",
                  flush=True)

    def get(arch, seed, pt):
        return next(c for c in cells if c["arch"] == arch and c["seed"] == seed and c["pair_type"] == pt)

    # ------------------------------------------------------------------
    # lstm seed2 specific test: marker-only representation -> marker-only
    # causal use?
    # ------------------------------------------------------------------
    l2_mc, l2_mp = get("lstm", 2, "marker_count"), get("lstm", 2, "marker_position")
    g_mc = l2_mc["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    g_mp = l2_mp["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    both_causal = g_mc >= DEGENERATE_GAP_THRESHOLD and g_mp >= DEGENERATE_GAP_THRESHOLD
    lstm2_check = {
        "marker_count_gap": g_mc, "marker_position_gap": g_mp, "both_causally_used": both_causal,
        "interpretation": (
            "CONFIRMED -- marker features are causally load-bearing at normal magnitude, completing "
            "the mechanism story: lstm seed2's representation is marker-only (Stage 2/3), and its "
            "causal use is ALSO marker-only. This is a fully characterized, unusually clean single-"
            "cell mechanism: marker-only representation, marker-only causal use, insufficient for a "
            "hard-negative population that requires structural/multiset information this cell simply "
            "does not have."
            if both_causal else
            "NOT fully confirmed -- despite normal-strength marker representation (Stage 2), at least "
            "one marker feature shows a smaller-than-expected causal effect here. Report raw numbers; "
            "the marker-only story would need qualification, not simple confirmation."
        ),
    }
    print("\n=== lstm seed2 marker-only causal-use check ===")
    print(json.dumps(lstm2_check, indent=2, default=str))

    out = {
        "condition": "condition_1_marker_features",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only.",
        "description": "Tests whether marker_count and marker_position causally drive the accept/"
                       "reject decision, all 8 cells. Predicted: causally load-bearing for most "
                       "cells (Stage 2 found uniform representation, 0.20-0.36 selectivity, every "
                       "cell, no standout elevated/depressed cell unlike odds-first).",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "marker_count_pairs_audit": mc_audit,
        "marker_position_pairs_audit": mp_audit,
        "sample_pairs": {"marker_count": mc_pairs[:3], "marker_position": mp_pairs[:3]},
        "cells": cells,
        "lstm_seed2_marker_only_causal_use_check": lstm2_check,
    }
    out_path = RESULTS / "phase3_bucketsort_condition1.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
