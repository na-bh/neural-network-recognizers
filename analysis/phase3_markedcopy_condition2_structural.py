"""Marked-copy Phase 3, condition 2 of 3: structural features (length_parity,
is_balanced). Tests Stage 3's finding that these two features are NOT
uniformly represented (unlike marker features, Stage 2) -- only rnn seed3
(both) and transformer seed3 (is_balanced only) clear the 0.05 probe
selectivity threshold; every other cell is near-zero.

Per the approved protocol: PRIMARY evidence is the behavioral logit gap on
each pair type; full-state patch (canonical readout site) is a SECONDARY
wiring check; and (matching condition 1's numerically-unstable-denominator
caveat, established in marked-reversal's condition 2) a LAYER-SPECIFIC patch
at each cell's OWN best-selectivity layer for that target (from Stage 3's
phase2_markedcopy_structural_probes.json) is reported alongside the
full-state check, since these features -- unlike marker features -- are NOT
broadly represented and a canonical-site-only patch could miss a signal
concentrated at one layer.

Because length_parity_pairs' corrupt is one token LONGER than clean, the
read/write positions are decoupled (t_clean != t_corrupt in general) --
reuses the same decoupled-position single-layer hooks marked-reversal's
condition 2 built for mamba/transformer, and RNN/LSTM's record_layer_state /
patch_logit_at_layer (already layer-decoupled by construction).

LSTM seed1 gets explicit attention here (per instruction): does the
"representation causally reachable, real-input routing fails" pattern from
condition 1 (marker features) generalize to length_parity/is_balanced, or
are marker features the only thing this cell represents?

Saves analysis_outputs/final_results/phase3_markedcopy_condition2_structural.json.
STOP after this condition for review before condition 3 (copy_match_balanced_case null test).

PYTHONPATH=src:analysis python analysis/phase3_markedcopy_condition2_structural.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_markedcopy_counterfactuals import (
    balance_pairs, length_parity_pairs, audit_balance_pairs, audit_length_parity_pairs,
)

RESULTS = Path("analysis_outputs/final_results")
TASK = "marked-copy"
CELLS = [
    ("rnn", 3, "primary"), ("rnn", 2, "secondary_comparable"),
    ("lstm", 9, "primary"), ("lstm", 1, "secondary_chance_collapser"),
    ("transformer", 2, "primary"), ("transformer", 3, "secondary_comparable"),
    ("mamba", 7, "primary"), ("mamba", 8, "secondary_comparable"),
]
N_HALF = 50
N_PAIRS = 25
DEGENERATE_GAP_THRESHOLD = 0.05


def load_best_layers():
    d = json.loads((RESULTS / "phase2_markedcopy_structural_probes.json").read_text())
    best = {}
    for m in d["models"]:
        key = (m["arch"], m["seed"])
        best[key] = {}
        for target in ["length_parity", "is_balanced"]:
            sels = {int(l): v["selectivity"] for l, v in m[f"{target}_probe_by_layer"].items()}
            bl = max(sels, key=sels.get)
            best[key][target] = {"best_layer": bl, "all_layer_selectivity": sels}
    return best


# ---------------------------------------------------------------------------
# decoupled-position single-layer patch helpers for mamba/transformer (clean
# and corrupt differ in length for length_parity_pairs)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# per-architecture cell runner
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # rnn seed3 focused check (prediction: large effect, both features)
    # ------------------------------------------------------------------
    rnn3 = {c["pair_type"]: c for c in all_cells if c["arch"] == "rnn" and c["seed"] == 3}
    rnn3_check = {
        "length_parity_gap": rnn3["length_parity"]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"],
        "length_parity_layer_rf": rnn3["length_parity"]["layer_specific_patch_at_best_layer"]["layer_specific_restored_fraction_mean"],
        "is_balanced_gap": rnn3["is_balanced"]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"],
        "is_balanced_layer_rf": rnn3["is_balanced"]["layer_specific_patch_at_best_layer"]["layer_specific_restored_fraction_mean"],
    }
    print("\n=== rnn seed3 focused check ===")
    print(json.dumps(rnn3_check, indent=2))

    # ------------------------------------------------------------------
    # transformer seed3 is_balanced focused check (0.069 probe selectivity --
    # is it causal?)
    # ------------------------------------------------------------------
    t3_ib = next(c for c in ib_cells if c["arch"] == "transformer" and c["seed"] == 3)
    t3_gap = t3_ib["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    t3_rf = t3_ib["layer_specific_patch_at_best_layer"]["layer_specific_restored_fraction_mean"]
    t3_causal = t3_gap >= DEGENERATE_GAP_THRESHOLD
    transformer3_check = {
        "probe_selectivity": t3_ib["layer_selectivity_from_sweep"][str(t3_ib["best_layer"])] if str(t3_ib["best_layer"]) in t3_ib["layer_selectivity_from_sweep"] else t3_ib["layer_selectivity_from_sweep"].get(t3_ib["best_layer"]),
        "best_layer": t3_ib["best_layer"],
        "behavioral_gap": t3_gap,
        "layer_specific_rf": t3_rf,
        "is_causal": t3_causal,
        "interpretation": (
            f"transformer seed3's is_balanced probe selectivity (0.069 at layer{t3_ib['best_layer']}) "
            + ("DOES translate into a real (if small) behavioral effect -- a genuine, if minor, "
               "causal mechanism, not just a decodable correlate."
               if t3_causal else
               "does NOT translate into a meaningful behavioral effect (gap below noise floor) -- "
               "a clean example of probe-detectable representation with no causal role, matching "
               "the pattern already seen for rnn's marker_position and several other Stage 3 cells.")
        ),
    }
    print("\n=== transformer seed3 is_balanced focused check ===")
    print(json.dumps(transformer3_check, indent=2, default=str))

    # ------------------------------------------------------------------
    # LSTM seed1: does condition 1's "reachable but not routed to" pattern
    # generalize to length_parity/is_balanced, or is it marker-feature-only?
    # ------------------------------------------------------------------
    lstm1 = {c["pair_type"]: c for c in all_cells if c["arch"] == "lstm" and c["seed"] == 1}
    cond1 = json.loads((RESULTS / "phase3_markedcopy_condition1_marker.json").read_text())
    lstm1_marker = {c["pair_type"]: c for c in cond1["cells"] if c["arch"] == "lstm" and c["seed"] == 1}

    lp_gap = lstm1["length_parity"]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    ib_gap = lstm1["is_balanced"]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    lp_probe_sel = length_parity_best[("lstm", 1)]["all_layer_selectivity"]
    ib_probe_sel = is_balanced_best[("lstm", 1)]["all_layer_selectivity"]
    lp_probe_best = length_parity_best[("lstm", 1)]["best_layer"]
    ib_probe_best = is_balanced_best[("lstm", 1)]["best_layer"]

    lp_has_probe_signal = lp_probe_sel[lp_probe_best] >= DEGENERATE_GAP_THRESHOLD
    ib_has_probe_signal = ib_probe_sel[ib_probe_best] >= DEGENERATE_GAP_THRESHOLD
    lp_is_causal = lp_gap >= DEGENERATE_GAP_THRESHOLD
    ib_is_causal = ib_gap >= DEGENERATE_GAP_THRESHOLD

    mc_gap = lstm1_marker["marker_count"]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    mp_gap = lstm1_marker["marker_position"]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]

    lstm1_generalization = {
        "condition_1_marker_features_recap": {
            "marker_count_gap": mc_gap, "marker_position_gap": mp_gap,
            "pattern": "both causally reachable (large gap, rf~1.0) despite chance behavior on real inputs",
        },
        "condition_2_structural_features": {
            "length_parity": {
                "probe_selectivity_best_layer": {"layer": lp_probe_best, "selectivity": lp_probe_sel[lp_probe_best]},
                "has_probe_signal": bool(lp_has_probe_signal),
                "behavioral_gap": lp_gap, "is_causally_reachable": bool(lp_is_causal),
            },
            "is_balanced": {
                "probe_selectivity_best_layer": {"layer": ib_probe_best, "selectivity": ib_probe_sel[ib_probe_best]},
                "has_probe_signal": bool(ib_has_probe_signal),
                "behavioral_gap": ib_gap, "is_causally_reachable": bool(ib_is_causal),
            },
        },
        "verdict": None,
    }
    if not lp_has_probe_signal and not ib_has_probe_signal:
        lstm1_generalization["verdict"] = (
            "FEATURE-SPECIFIC: lstm seed1 shows NO probe signal for length_parity or is_balanced "
            "(both near-zero selectivity, consistent with Stage 3's finding that only rnn seed3 and "
            "transformer seed3 represent these features at all). The 'representation causally "
            "reachable but input-routing fails' story from condition 1 does NOT generalize -- "
            "seed1 only REPRESENTS marker features (and represents them more strongly than its own "
            "working primary seed9); it never represented length_parity/is_balanced in the first "
            "place, so there is no representation-vs-use gap to test for those two targets. The "
            "input-routing failure story is specific to the marker-feature family, not general to "
            "every feature this cell could in principle carry."
        )
    elif (lp_has_probe_signal and lp_is_causal) or (ib_has_probe_signal and ib_is_causal):
        lstm1_generalization["verdict"] = (
            "GENERALIZES: at least one structural feature shows both probe signal AND causal "
            "reachability, extending the 'representation present, real-input routing fails' pattern "
            "beyond marker features to the structural-feature family as well -- suggesting seed1's "
            "routing failure is a general property of this cell's forward pass, not specific to how "
            "marker features are read out."
        )
    else:
        lstm1_generalization["verdict"] = (
            "MIXED/AMBIGUOUS: probe signal and causal reachability do not line up cleanly for the "
            "structural features -- report the raw numbers above rather than forcing a single "
            "verdict; this does not cleanly confirm or refute generalization of condition 1's "
            "pattern."
        )
    print("\n=== LSTM seed1 generalization check (condition 1 pattern -> structural features) ===")
    print(json.dumps(lstm1_generalization, indent=2, default=str))

    out = {
        "condition": "condition_2_structural_features",
        "status": "PRIMARY evidence is the behavioral logit gap per pair type; full-state patch is "
                 "a SECONDARY wiring check; layer-specific patch at each cell's own best-selectivity "
                 "layer (Stage 3 sweep) is reported and explicitly flagged non-meaningful whenever "
                 "the behavioral gap is near the noise floor.",
        "description": "Tests length_parity (via length_parity_pairs) and is_balanced (via "
                       "balance_pairs) causally, all 8 cells. Predicted: large effect for rnn seed3 "
                       "on both; near-zero elsewhere except transformer seed3's is_balanced (probe "
                       "selectivity 0.069, causal status tested explicitly here, not assumed).",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "balance_pairs_audit": bp_audit, "length_parity_pairs_audit": lpp_audit,
        "sample_pairs": {"length_parity": lpp[:3], "is_balanced": bp[:3]},
        "cells": all_cells,
        "rnn_seed3_focused_check": rnn3_check,
        "transformer_seed3_is_balanced_focused_check": transformer3_check,
        "lstm_seed1_generalization_check": lstm1_generalization,
    }
    out_path = RESULTS / "phase3_markedcopy_condition2_structural.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
