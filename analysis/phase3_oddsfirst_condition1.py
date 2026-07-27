"""Odds-first Phase 3, condition 1 of 3: marker features (marker_count,
marker_position). Tests whether marker shortcut features -- represented at
0.21-0.41 selectivity by ALL 8 cells in Stage 2 -- causally drive the
accept/reject decision.

PRIMARY evidence is the behavioral logit gap (unpatched clean vs corrupt);
full-state patch (canonical readout site per architecture) is a SECONDARY
wiring check.

Two cell-specific investigations built into this condition's own analysis
(per the Phase 2 summary's explicit Phase 3 predictions):

  - lstm seed4 vs seed5: Stage 2/3 found IDENTICAL representation at every
    layer/target. Does the causal pattern match too?
  - transformer seed6: marker_count_pairs' corruption (multi-marker) is
    structurally the SAME construction as A1's 'scrambled' negative bucket
    (easy); marker_position_pairs' corruption (single marker, off-center =
    imbalanced) is structurally the SAME construction as A1's 'hard'
    negative bucket (per Phase 1's own audit: odds-first's hard negatives
    are dominated by single-marker unequal-halves cases). Comparing these
    two pair types' behavioral gaps directly tests whether marker-feature
    causal use is normal on the 'easy' construction but fails specifically
    on the 'hard' one -- the mechanistic explanation for why seed6 scores
    BELOW its own marker-shortcut baseline on real hard negatives.

Saves analysis_outputs/final_results/phase3_oddsfirst_condition1.json.
STOP after this condition to check whether the LSTM cross-architecture
convergence hypothesis is causally supported.

PYTHONPATH=src:analysis python analysis/phase3_oddsfirst_condition1.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_oddsfirst_counterfactuals import (
    marker_count_pairs, marker_position_pairs,
    audit_marker_count_pairs, audit_marker_position_pairs,
)

RESULTS = Path("analysis_outputs/final_results")
TASK = "odds-first"
CELLS = [
    ("rnn", 3, "primary"), ("rnn", 1, "secondary_comparable"),
    ("lstm", 4, "primary"), ("lstm", 5, "secondary_comparable_twin"),
    ("transformer", 6, "primary"), ("transformer", 3, "secondary_near_chance_partial"),
    ("mamba", 0, "primary"), ("mamba", 6, "secondary_comparable_twin"),
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
    # lstm seed4 vs seed5 causal comparison
    # ------------------------------------------------------------------
    lstm4_mc, lstm4_mp = get("lstm", 4, "marker_count"), get("lstm", 4, "marker_position")
    lstm5_mc, lstm5_mp = get("lstm", 5, "marker_count"), get("lstm", 5, "marker_position")
    g4mc = lstm4_mc["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    g4mp = lstm4_mp["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    g5mc = lstm5_mc["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    g5mp = lstm5_mp["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    twin_causal_match = abs(g4mc - g5mc) < 0.5 * max(g4mc, g5mc, 1e-9) and abs(g4mp - g5mp) < 0.5 * max(g4mp, g5mp, 1e-9)
    lstm_twin_causal_comparison = {
        "marker_count_gap": {"seed4": g4mc, "seed5": g5mc},
        "marker_position_gap": {"seed4": g4mp, "seed5": g5mp},
        "pattern_matches": twin_causal_match,
        "interpretation": (
            "Causal pattern MATCHES the representational twin-convergence found in Stage 2/3 -- both "
            "seeds show comparable-magnitude behavioral gaps for both marker features, extending "
            "'identical representation' to 'identical causal use.'"
            if twin_causal_match else
            "Causal pattern DIVERGES despite Stage 2/3's representational identity -- seed4 and seed5 "
            "show meaningfully different behavioral gaps for at least one marker feature. This would "
            "be a genuine representation-shared-but-causal-use-differs dissociation, worth flagging "
            "rather than assuming the twin pattern automatically extends to causal use."
        ),
    }
    print("\n=== lstm seed4 vs seed5 causal comparison ===")
    print(json.dumps(lstm_twin_causal_comparison, indent=2, default=str))

    # ------------------------------------------------------------------
    # transformer seed6: marker_count (scrambled-like) vs marker_position
    # (hard-negative-like) gap comparison
    # ------------------------------------------------------------------
    t6_mc = get("transformer", 6, "marker_count")
    t6_mp = get("transformer", 6, "marker_position")
    g_mc = t6_mc["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    g_mp = t6_mp["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    rf_mc = t6_mc["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
    rf_mp = t6_mp["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
    hard_vs_scrambled_asymmetric = (g_mc >= DEGENERATE_GAP_THRESHOLD) and (g_mp < DEGENERATE_GAP_THRESHOLD)
    transformer6_hard_vs_scrambled = {
        "rationale": "marker_count_pairs' corruption (extra marker, multi-marker) structurally matches "
                    "A1's 'scrambled' negative bucket (easy); marker_position_pairs' corruption "
                    "(single marker, off-center = imbalanced) structurally matches A1's 'hard' negative "
                    "bucket (per Phase 1's audit: odds-first hard negatives are dominated by single-"
                    "marker unequal-halves cases).",
        "marker_count_scrambled_like_gap": g_mc, "marker_count_wiring_rf": rf_mc,
        "marker_position_hard_like_gap": g_mp, "marker_position_wiring_rf": rf_mp,
        "asymmetric_pattern_confirmed": hard_vs_scrambled_asymmetric,
        "interpretation": (
            f"CONFIRMED asymmetric pattern -- marker_count/scrambled-like gap ({g_mc:.4f}) is causally "
            f"large while marker_position/hard-like gap ({g_mp:.4f}) is near noise floor. This directly "
            f"explains the sub-baseline hard-negative accuracy: seed6 causally uses marker information "
            f"for the EASY (multi-marker) construction but fails to causally use it for the HARD "
            f"(single-marker-imbalanced) construction -- a targeted, mechanistic explanation, not just "
            f"a correlational observation."
            if hard_vs_scrambled_asymmetric else
            f"NOT the simple asymmetric pattern hypothesized -- marker_count/scrambled-like gap "
            f"({g_mc:.4f}) and marker_position/hard-like gap ({g_mp:.4f}) do not split cleanly across "
            f"the 0.05 threshold in the predicted direction. Report the raw numbers rather than forcing "
            f"the hypothesized story; the sub-baseline hard-negative accuracy needs a different "
            f"explanation, to be pursued via conditions 2/3."
        ),
    }
    print("\n=== transformer seed6: marker_count (scrambled-like) vs marker_position (hard-like) ===")
    print(json.dumps(transformer6_hard_vs_scrambled, indent=2, default=str))

    out = {
        "condition": "condition_1_marker_features",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only.",
        "description": "Tests whether marker_count and marker_position causally drive the accept/"
                       "reject decision, all 8 cells. Predicted: causally load-bearing for most "
                       "cells (Stage 2 found broad representation, 0.21-0.41 selectivity, every cell).",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "marker_count_pairs_audit": mc_audit,
        "marker_position_pairs_audit": mp_audit,
        "sample_pairs": {"marker_count": mc_pairs[:3], "marker_position": mp_pairs[:3]},
        "cells": cells,
        "lstm_seed4_vs_seed5_causal_comparison": lstm_twin_causal_comparison,
        "transformer_seed6_hard_vs_scrambled_like_gap_comparison": transformer6_hard_vs_scrambled,
    }
    out_path = RESULTS / "phase3_oddsfirst_condition1.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
