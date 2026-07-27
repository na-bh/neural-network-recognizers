"""Odds-first Phase 3, condition 2 of 3: structural features (length_parity,
is_balanced). Tests Stage 3's finding that these features are represented
strongly by rnn (both seeds) and lstm (both seeds, elevated far beyond any
prior marker-family LSTM cell), and NOT represented by transformer or mamba
(any seed).

PRIMARY evidence is the behavioral logit gap per pair type; full-state patch
(canonical readout site) is a SECONDARY wiring check; a LAYER-SPECIFIC patch
at each cell's own best-selectivity layer (Stage 3 sweep) is reported
alongside, flagged non-meaningful whenever the behavioral gap itself is
near the noise floor (numerically-unstable-denominator caveat, established
in marked-copy's condition 2).

Five predictions tested explicitly:
  (1) lstm seed4 AND seed5: both features causally used at large magnitude
      -- would confirm genuine (not just representational) convergence on
      RNN's multi-feature mechanism.
  (2) rnn seed3 AND seed1: length_parity causal, is_balanced NOT causal
      (probe correlate) -- two-cell replication of the marked-reversal/
      marked-copy precedent.
  (3) transformer seed6: near-zero for both -- mechanism is marker_position-
      dominant with weakly-used marker_count (per condition 1).
  (4) transformer seed3: near-zero for both -- clean marker-features-only
      mechanism.
  (5) mamba seed0 AND seed6: near-zero for both -- standard shortcut
      behavior.

Additional analysis: explicit transformer seed6-vs-seed3 comparison on both
structural features, to test whether condition 1's marker_count causal-
strength asymmetry (seed6 5x weaker than seed3, despite near-identical
probe selectivity) is marker_count-specific or a broader seed-level pattern.

Saves analysis_outputs/final_results/phase3_oddsfirst_condition2.json.

PYTHONPATH=src:analysis python analysis/phase3_oddsfirst_condition2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_oddsfirst_counterfactuals import (
    balance_pairs, length_parity_pairs, audit_balance_pairs, audit_length_parity_pairs,
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


def load_best_layers():
    d = json.loads((RESULTS / "phase2_oddsfirst_structural_probes.json").read_text())
    best = {}
    for m in d["models"]:
        key = (m["arch"], m["seed"])
        best[key] = {}
        for target in ["length_parity", "is_balanced"]:
            sels = {int(l): v["selectivity"] for l, v in m[f"{target}_probe_by_layer"].items()}
            bl = max(sels, key=sels.get)
            best[key][target] = {"best_layer": bl, "all_layer_selectivity": sels}
    return best


def mamba_patch_relative(h, clean_idx, corrupt_idx, layer, t_clean, t_corrupt, dims=None):
    _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
    cv = clean_cache[layer][t_clean].to(h.device)

    def hook(module, inputs, output):
        out = output.clone()
        if dims is None:
            out[0, t_corrupt, :] = cv
        else:
            out[0, t_corrupt, dims] = cv[dims]
        return out

    handle = h.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            rec, _, _ = h.iface.get_logits(h.model, h._model_input(corrupt_idx))
    finally:
        handle.remove()
    return float(rec.item())


def transformer_patch_relative(h, clean_idx, corrupt_idx, layer, t_clean, t_corrupt, dims=None):
    _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
    cv = clean_cache[layer][t_clean + h.bos_off].to(h.device)
    row = t_corrupt + h.bos_off

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        out = output[0] if is_tuple else output
        out = out.clone()
        if dims is None:
            out[0, row, :] = cv
        else:
            out[0, row, dims] = cv[dims]
        return (out,) + output[1:] if is_tuple else out

    handle = h.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            rec, _, _ = h.iface.get_logits(h.model, h._model_input(corrupt_idx))
    finally:
        handle.remove()
    return float(rec.item())


def run_rnn_lstm(arch, seed, pairs, tok2idx, best_layer):
    from rnn_patching import RNNPatchingHarness, RecurrentScan, SITE_RNN, SITE_LSTM_FULL
    h = RNNPatchingHarness(arch, f"rec+ns/validation-short/{seed}", task=TASK)
    sc = RecurrentScan(h)
    site = SITE_RNN if arch == "rnn" else SITE_LSTM_FULL

    wiring, layer_specific, behavioral = [], [], []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_clean = h.readout_position(p["clean"])
        t_corrupt = h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))

        _, states = sc.record(clean_idx, [t_clean])
        patched_full = sc.patch_logit(corrupt_idx, t_corrupt, states[t_clean], site=site, dims=None)
        wiring.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})

        clean_layer_state = sc.record_layer_state(clean_idx, best_layer, t_clean)
        patched_layer = sc.patch_logit_at_layer(corrupt_idx, t_corrupt, best_layer, clean_layer_state, dims=None)
        layer_specific.append({"restored_fraction": h.restored_fraction(lc, lo, patched_layer)})
    return wiring, layer_specific, behavioral, f"data/models/{TASK}/{arch}/rec+ns/validation-short/{seed}"


def run_mamba(seed, pairs, tok2idx, best_layer):
    from patching_harness import PatchingHarness
    h = PatchingHarness(task=TASK, model_subdir=f"mamba/rec+ns/validation-short/{seed}")
    top_layer = h.num_layers - 1

    wiring, layer_specific, behavioral = [], [], []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_clean = h.readout_position(p["clean"])
        t_corrupt = h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))

        patched_full = mamba_patch_relative(h, clean_idx, corrupt_idx, top_layer, t_clean, t_corrupt, dims=None)
        wiring.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})

        patched_layer = mamba_patch_relative(h, clean_idx, corrupt_idx, best_layer, t_clean, t_corrupt, dims=None)
        layer_specific.append({"restored_fraction": h.restored_fraction(lc, lo, patched_layer)})
    return wiring, layer_specific, behavioral, f"models/{TASK}/mamba/rec+ns/validation-short/{seed}"


def run_transformer(seed, pairs, tok2idx, best_layer):
    from modk_transformer_harness import TransformerPatchingHarness
    h = TransformerPatchingHarness(TASK, f"rec+ns/validation-short/{seed}")
    top_layer = h.num_layers - 1

    wiring, layer_specific, behavioral = [], [], []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_clean = len(p["clean"]) - 1
        t_corrupt = len(p["corrupt"]) - 1
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))
        denom = lc - lo

        patched_full = transformer_patch_relative(h, clean_idx, corrupt_idx, top_layer, t_clean, t_corrupt, dims=None)
        rf_full = (patched_full - lo) / denom if abs(denom) > 1e-9 else float("nan")
        wiring.append({"restored_fraction": rf_full})

        patched_layer = transformer_patch_relative(h, clean_idx, corrupt_idx, best_layer, t_clean, t_corrupt, dims=None)
        rf_layer = (patched_layer - lo) / denom if abs(denom) > 1e-9 else float("nan")
        layer_specific.append({"restored_fraction": rf_layer})
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


def run_condition(pair_type, pairs, best_layers_for_target, tok2idx):
    cells = []
    for arch, seed, role in CELLS:
        bl_info = best_layers_for_target[(arch, seed)]
        best_layer = bl_info["best_layer"]
        print(f"\n--- [{pair_type}] {arch} seed{seed} ({role}, best_layer={best_layer}) ---", flush=True)
        if arch in ("rnn", "lstm"):
            wiring, layer_specific, behavioral, ckpt = run_rnn_lstm(arch, seed, pairs, tok2idx, best_layer)
        elif arch == "mamba":
            wiring, layer_specific, behavioral, ckpt = run_mamba(seed, pairs, tok2idx, best_layer)
        else:
            wiring, layer_specific, behavioral, ckpt = run_transformer(seed, pairs, tok2idx, best_layer)

        beh = summarize_behavioral(behavioral)
        wiring_s = summarize_rf(wiring, "wiring")
        layer_s = summarize_rf(layer_specific, "layer_specific")
        gap = beh["mean_abs_clean_corrupt_logit_gap"]
        rf_meaningful = gap >= DEGENERATE_GAP_THRESHOLD
        cell = {
            "arch": arch, "seed": seed, "role": role, "checkpoint": ckpt,
            "pair_type": pair_type,
            "best_layer": best_layer, "layer_selectivity_from_sweep": bl_info["all_layer_selectivity"],
            "PRIMARY_behavioral_logit_gap": beh,
            "wiring_check_full_state_patch_SECONDARY": wiring_s,
            "layer_specific_patch_at_best_layer": layer_s,
            "layer_specific_rf_is_meaningful": rf_meaningful,
        }
        if not rf_meaningful:
            cell["layer_specific_patch_caveat"] = (
                f"behavioral gap ({gap:.4f}) is near noise floor -- layer-specific rf is not a "
                f"reliable causal estimate here; near-zero gap IS itself the finding (feature not "
                f"used), not a measurement failure."
            )
        cells.append(cell)
        print(f"  PRIMARY behavioral gap: mean={gap:.4f} (ci95={beh['ci95_abs_clean_corrupt_logit_gap']:.4f}, n={beh['n']})", flush=True)
        print(f"  layer{best_layer}-specific patch: rf_mean={layer_s['layer_specific_restored_fraction_mean']:.4f} "
              f"meaningful={rf_meaningful}", flush=True)
        print(f"  full-state wiring check (secondary): rf_mean={wiring_s['wiring_restored_fraction_mean']:.4f}", flush=True)
    return cells


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    bp = balance_pairs(N_HALF, N_PAIRS, rng)
    lpp = length_parity_pairs(N_HALF, N_PAIRS, rng)
    bp_audit = audit_balance_pairs(bp)
    lpp_audit = audit_length_parity_pairs(lpp)
    print("=== balance_pairs (is_balanced) audit ===")
    print(json.dumps(bp_audit, indent=2))
    print("=== length_parity_pairs audit ===")
    print(json.dumps(lpp_audit, indent=2))
    assert bp_audit["target_property_isolated"] and lpp_audit["target_property_isolated"]

    best_layers = load_best_layers()
    is_balanced_best = {k: v["is_balanced"] for k, v in best_layers.items()}
    length_parity_best = {k: v["length_parity"] for k, v in best_layers.items()}

    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    lp_cells = run_condition("length_parity", lpp, length_parity_best, tok2idx)
    ib_cells = run_condition("is_balanced", bp, is_balanced_best, tok2idx)
    all_cells = lp_cells + ib_cells

    def get(arch, seed, pt):
        return next(c for c in all_cells if c["arch"] == arch and c["seed"] == seed and c["pair_type"] == pt)

    # ------------------------------------------------------------------
    # per-prediction checks
    # ------------------------------------------------------------------
    def gap_of(arch, seed, pt):
        return get(arch, seed, pt)["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]

    prediction_checks = {}
    for arch, seed, label in [("lstm", 4, "lstm_seed4"), ("lstm", 5, "lstm_seed5")]:
        lp_g, ib_g = gap_of(arch, seed, "length_parity"), gap_of(arch, seed, "is_balanced")
        both_causal = lp_g >= DEGENERATE_GAP_THRESHOLD and ib_g >= DEGENERATE_GAP_THRESHOLD
        prediction_checks[label] = {
            "length_parity_gap": lp_g, "is_balanced_gap": ib_g, "both_causally_used": both_causal,
            "verdict": (
                "CONFIRMED -- both structural features causally used at large magnitude, matching Stage "
                "3's elevated selectivity. Genuine (not just representational) convergence on RNN's "
                "multi-feature mechanism."
                if both_causal else
                "NOT CONFIRMED -- despite Stage 3's elevated probe selectivity (0.097-0.171), at least "
                "one structural feature shows near-zero causal effect. The convergence is "
                "REPRESENTATIONAL ONLY -- decodable but not (or not fully) causally used."
            ),
        }
    for arch, seed, label in [("rnn", 3, "rnn_seed3"), ("rnn", 1, "rnn_seed1")]:
        lp_g, ib_g = gap_of(arch, seed, "length_parity"), gap_of(arch, seed, "is_balanced")
        matches_precedent = lp_g >= DEGENERATE_GAP_THRESHOLD and ib_g < DEGENERATE_GAP_THRESHOLD
        prediction_checks[label] = {
            "length_parity_gap": lp_g, "is_balanced_gap": ib_g, "matches_marker_family_precedent": matches_precedent,
            "verdict": (
                "CONFIRMED -- length_parity causal, is_balanced not causal, replicating marked-reversal/"
                "marked-copy's RNN precedent (length parity is the real mechanism; is_balanced was a "
                "probe correlate confounded with it)."
                if matches_precedent else
                "DOES NOT MATCH the two-cell replication predicted -- report raw numbers; either both "
                "features are causal, neither is, or is_balanced is causal too (which would be a novel "
                "deviation from the length-parity-only precedent)."
            ),
        }
    for arch, seed, label in [("transformer", 6, "transformer_seed6"), ("transformer", 3, "transformer_seed3"),
                              ("mamba", 0, "mamba_seed0"), ("mamba", 6, "mamba_seed6")]:
        lp_g, ib_g = gap_of(arch, seed, "length_parity"), gap_of(arch, seed, "is_balanced")
        both_null = lp_g < DEGENERATE_GAP_THRESHOLD and ib_g < DEGENERATE_GAP_THRESHOLD
        prediction_checks[label] = {
            "length_parity_gap": lp_g, "is_balanced_gap": ib_g, "both_null_as_predicted": both_null,
            "verdict": (
                "CONFIRMED -- near-zero causal effect for both structural features, matching Stage 3's "
                "near-zero probe signal."
                if both_null else
                "SURPRISE -- at least one structural feature shows above-noise causal effect despite "
                "near-zero Stage 3 probe signal, echoing marked-copy's probe-blind-spot pattern. Flagged, "
                "not smoothed over."
            ),
        }
    print("\n=== per-prediction checks ===")
    print(json.dumps(prediction_checks, indent=2, default=str))

    # ------------------------------------------------------------------
    # transformer seed6 vs seed3: structural feature comparison (does the
    # marker_count causal asymmetry from condition 1 generalize?)
    # ------------------------------------------------------------------
    t6_lp, t6_ib = gap_of("transformer", 6, "length_parity"), gap_of("transformer", 6, "is_balanced")
    t3_lp, t3_ib = gap_of("transformer", 3, "length_parity"), gap_of("transformer", 3, "is_balanced")
    cond1 = json.loads((RESULTS / "phase3_oddsfirst_condition1.json").read_text())
    t6_mc = next(c for c in cond1["cells"] if c["arch"] == "transformer" and c["seed"] == 6 and c["pair_type"] == "marker_count")
    t3_mc = next(c for c in cond1["cells"] if c["arch"] == "transformer" and c["seed"] == 3 and c["pair_type"] == "marker_count")
    mc_gap6 = t6_mc["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    mc_gap3 = t3_mc["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]

    both_near_zero_6 = t6_lp < DEGENERATE_GAP_THRESHOLD and t6_ib < DEGENERATE_GAP_THRESHOLD
    both_near_zero_3 = t3_lp < DEGENERATE_GAP_THRESHOLD and t3_ib < DEGENERATE_GAP_THRESHOLD
    structural_asymmetry_present = both_near_zero_6 and both_near_zero_3 and abs(t6_lp - t3_lp) < 0.05 and abs(t6_ib - t3_ib) < 0.05

    transformer_seed_comparison = {
        "condition_1_marker_count_gap_for_reference": {"seed6": mc_gap6, "seed3": mc_gap3, "ratio_seed3_over_seed6": mc_gap3 / mc_gap6 if mc_gap6 else float("inf")},
        "condition_2_length_parity_gap": {"seed6": t6_lp, "seed3": t3_lp},
        "condition_2_is_balanced_gap": {"seed6": t6_ib, "seed3": t3_ib},
        "both_seeds_structural_null": both_near_zero_6 and both_near_zero_3,
        "interpretation": (
            "MARKER_COUNT-SPECIFIC EFFECT -- both seed6 and seed3 show equally near-zero structural-"
            "feature causal effects (neither uses length_parity or is_balanced), so the sharp condition-1 "
            "asymmetry (seed3 causally uses marker_count ~5x more strongly than seed6, despite matched "
            "probe selectivity) does NOT generalize to a broader representational/causal split between "
            "these two seeds -- it is confined to marker_count specifically. Both seeds' mechanisms are "
            "otherwise identical (marker-features-only, no structural component); they differ only in "
            "HOW STRONGLY they lean on marker_count."
            if (both_near_zero_6 and both_near_zero_3) else
            "BROADER PATTERN -- seed6 and seed3 differ on at least one structural feature too (not just "
            "marker_count from condition 1), suggesting the seed-level dissociation is not confined to "
            "one feature but reflects a more general difference in how these two seeds process the "
            "input -- see raw numbers above rather than a single clean story."
        ),
    }
    print("\n=== transformer seed6 vs seed3: structural feature comparison ===")
    print(json.dumps(transformer_seed_comparison, indent=2, default=str))

    out = {
        "condition": "condition_2_structural_features",
        "status": "PRIMARY evidence is the behavioral logit gap per pair type; full-state patch is a "
                 "SECONDARY wiring check; layer-specific patch at each cell's own best-selectivity layer "
                 "(Stage 3 sweep) is reported and flagged non-meaningful whenever the behavioral gap is "
                 "near the noise floor.",
        "description": "Tests length_parity (via length_parity_pairs) and is_balanced (via balance_"
                       "pairs) causally, all 8 cells.",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "balance_pairs_audit": bp_audit, "length_parity_pairs_audit": lpp_audit,
        "sample_pairs": {"length_parity": lpp[:3], "is_balanced": bp[:3]},
        "cells": all_cells,
        "prediction_checks": prediction_checks,
        "transformer_seed6_vs_seed3_structural_comparison": transformer_seed_comparison,
    }
    out_path = RESULTS / "phase3_oddsfirst_condition2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
