"""Bucket-sort Phase 3, condition 2 of 3: structural + multiset features
(length_parity, is_balanced, is_multiset_matched). Tests Stage 3's finding
that length_parity/is_balanced are strongly represented by rnn (both seeds)
and moderately by lstm seed3 (not seed2, which is clean marker-only), and
that is_multiset_matched shows its highest selectivity in transformer (both
seeds, ~0.047-0.050) -- a candidate mechanism for transformer's family-wide
hard-negative high-water mark (0.560).

PRIMARY evidence is the behavioral logit gap per pair type; full-state
patch (canonical readout site) is a SECONDARY wiring check; a LAYER-
SPECIFIC patch at each cell's own best-selectivity layer (Stage 3 sweep) is
reported alongside, flagged non-meaningful whenever the behavioral gap
itself is near the noise floor.

Saves analysis_outputs/final_results/phase3_bucketsort_condition2.json.

PYTHONPATH=src:analysis python analysis/phase3_bucketsort_condition2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_bucketsort_counterfactuals import (
    balance_pairs, length_parity_pairs, multiset_pairs,
    audit_balance_pairs, audit_length_parity_pairs, audit_multiset_pairs,
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


def load_best_layers():
    d = json.loads((RESULTS / "phase2_bucketsort_structural_probes.json").read_text())
    best = {}
    for m in d["models"]:
        key = (m["arch"], m["seed"])
        best[key] = {}
        for target in ["length_parity", "is_balanced", "is_multiset_matched"]:
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
    msp = multiset_pairs(N_HALF, N_PAIRS, rng)
    bp_audit = audit_balance_pairs(bp)
    lpp_audit = audit_length_parity_pairs(lpp)
    msp_audit = audit_multiset_pairs(msp)
    print("=== balance_pairs (is_balanced) audit ===")
    print(json.dumps(bp_audit, indent=2))
    print("=== length_parity_pairs audit ===")
    print(json.dumps(lpp_audit, indent=2))
    print("=== multiset_pairs (is_multiset_matched) audit ===")
    print(json.dumps(msp_audit, indent=2))
    assert bp_audit["target_property_isolated"] and lpp_audit["target_property_isolated"] and msp_audit["target_property_isolated"]

    best_layers = load_best_layers()
    is_balanced_best = {k: v["is_balanced"] for k, v in best_layers.items()}
    length_parity_best = {k: v["length_parity"] for k, v in best_layers.items()}
    multiset_best = {k: v["is_multiset_matched"] for k, v in best_layers.items()}

    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    lp_cells = run_condition("length_parity", lpp, length_parity_best, tok2idx)
    ib_cells = run_condition("is_balanced", bp, is_balanced_best, tok2idx)
    ms_cells = run_condition("is_multiset_matched", msp, multiset_best, tok2idx)
    all_cells = lp_cells + ib_cells + ms_cells

    # ------------------------------------------------------------------
    # transformer multiset-sensitivity check (both seeds -- the key Stage 3
    # hypothesis test for the family-wide hard-negative high-water mark)
    # ------------------------------------------------------------------
    def gap_of(arch, seed, pt):
        c = next(x for x in all_cells if x["arch"] == arch and x["seed"] == seed and x["pair_type"] == pt)
        return c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"], c

    t7_gap, t7_cell = gap_of("transformer", 7, "is_multiset_matched")
    t3_gap, t3_cell = gap_of("transformer", 3, "is_multiset_matched")
    t7_causal = t7_gap >= DEGENERATE_GAP_THRESHOLD
    t3_causal = t3_gap >= DEGENERATE_GAP_THRESHOLD
    transformer_multiset_check = {
        "seed7_gap": t7_gap, "seed7_causal": bool(t7_causal),
        "seed3_gap": t3_gap, "seed3_causal": bool(t3_causal),
        "seed7_layer_specific_rf": t7_cell["layer_specific_patch_at_best_layer"]["layer_specific_restored_fraction_mean"],
        "seed3_layer_specific_rf": t3_cell["layer_specific_patch_at_best_layer"]["layer_specific_restored_fraction_mean"],
        "interpretation": (
            "CONFIRMED -- is_multiset_matched is causally load-bearing for transformer (at least one, "
            "check both seeds' magnitudes above), providing a genuine mechanistic explanation for "
            "transformer's family-wide hard-negative high-water mark (0.560): this cell partially "
            "verifies sort-transform validity via multiset matching, a content-sensitivity capability "
            "not seen at this strength in any other architecture on any of the four marker-family "
            "tasks."
            if (t7_causal or t3_causal) else
            "NOT CONFIRMED -- despite the highest Stage 3 probe selectivity of any architecture "
            "(0.047-0.050), is_multiset_matched shows near-zero causal effect for transformer. This "
            "would be a probe-blind-spot dissociation in the OPPOSITE direction from marked-copy's "
            "precedent: signal present, but NOT causally used. Transformer's elevated hard-negative "
            "accuracy would then require a different explanation, not yet identified by this pilot's "
            "target inventory."
        ),
    }
    print("\n=== transformer is_multiset_matched causal-use check (both seeds) ===")
    print(json.dumps(transformer_multiset_check, indent=2, default=str))

    # ------------------------------------------------------------------
    # rnn is_balanced-vs-length_parity check (does the odds-first seed3
    # deviation -- is_balanced genuinely causal, not just a length-parity
    # confound -- recur here?)
    # ------------------------------------------------------------------
    rnn_check = {}
    for seed in [4, 10]:
        lp_gap, _ = gap_of("rnn", seed, "length_parity")
        ib_gap, _ = gap_of("rnn", seed, "is_balanced")
        rnn_check[f"seed{seed}"] = {
            "length_parity_gap": lp_gap, "is_balanced_gap": ib_gap,
            "is_balanced_also_causal": bool(ib_gap >= DEGENERATE_GAP_THRESHOLD),
        }
    print("\n=== rnn length_parity vs is_balanced causal check (both seeds) ===")
    print(json.dumps(rnn_check, indent=2, default=str))

    # ------------------------------------------------------------------
    # lstm seed2 structural/multiset null check (closes the marker-only loop)
    # ------------------------------------------------------------------
    l2_lp, _ = gap_of("lstm", 2, "length_parity")
    l2_ib, _ = gap_of("lstm", 2, "is_balanced")
    l2_ms, _ = gap_of("lstm", 2, "is_multiset_matched")
    lstm2_null_check = {
        "length_parity_gap": l2_lp, "is_balanced_gap": l2_ib, "is_multiset_matched_gap": l2_ms,
        "all_null": bool(l2_lp < DEGENERATE_GAP_THRESHOLD and l2_ib < DEGENERATE_GAP_THRESHOLD and l2_ms < DEGENERATE_GAP_THRESHOLD),
    }
    print("\n=== lstm seed2 structural/multiset null check ===")
    print(json.dumps(lstm2_null_check, indent=2, default=str))

    out = {
        "condition": "condition_2_structural_and_multiset_features",
        "status": "PRIMARY evidence is the behavioral logit gap per pair type; full-state patch is "
                 "a SECONDARY wiring check; layer-specific patch at each cell's own best-selectivity "
                 "layer (Stage 3 sweep) is reported and flagged non-meaningful whenever the "
                 "behavioral gap is near the noise floor.",
        "description": "Tests length_parity, is_balanced, and is_multiset_matched causally, all 8 "
                       "cells. is_multiset_matched added per user instruction as a candidate causal "
                       "mechanism for transformer's hard-negative high-water mark.",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "balance_pairs_audit": bp_audit, "length_parity_pairs_audit": lpp_audit,
        "multiset_pairs_audit": msp_audit,
        "sample_pairs": {"length_parity": lpp[:3], "is_balanced": bp[:3], "is_multiset_matched": msp[:3]},
        "cells": all_cells,
        "transformer_multiset_causal_use_check": transformer_multiset_check,
        "rnn_length_parity_vs_is_balanced_check": rnn_check,
        "lstm_seed2_structural_multiset_null_check": lstm2_null_check,
    }
    out_path = RESULTS / "phase3_bucketsort_condition2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
