"""Phase 3, patch condition 3 of 3 (confirmatory null test): balanced_content_
match. Tests whether genuine mirror-content verification -- held is_balanced
(marker centered, count=1) constant, only CONTENT varied -- causally drives
the accept/reject decision, for all 8 cells (4 architectures x 2 seeds).
Uses mirror_pairs (phase3_counterfactual_design.py), previously behaviorally
tested only on rnn seed3 (near-zero gap, 4.8e-08) during the design step;
this condition extends that test to every cell.

Predicted (per Phase 2's corrected balanced_content_match probe, which is
exactly 0.0 at every layer, every seed, all 4 architectures): near-zero
behavioral gap and near-zero/non-meaningful layer-specific restored fraction
everywhere. This also serves as the FOURTH independent null for lstm seed10's
marker_count-only attribution (marker preserved, content varied -- the one
axis not yet tested for that cell).

Same PRIMARY/SECONDARY structure as conditions 1-2: behavioral logit gap is
the primary evidence; full-state wiring check is secondary/reference; layer-
specific patch is flagged non-meaningful whenever the behavioral gap is
degenerate (near-zero), per the numerically-unstable-denominator caveat
established in condition 2.

Saves analysis_outputs/final_results/phase3_condition3_balanced_content_match.json.

PYTHONPATH=src:analysis python analysis/phase3_condition3_balanced_content_match.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_counterfactual_design import markedrev_mirror_pairs, audit_markedrev_pairs
from phase3_condition2_length_parity import mamba_patch_relative, transformer_patch_relative

RESULTS = Path("analysis_outputs/final_results")
TASK = "marked-reversal"
SEED_PAIRS = {"rnn": [3, 7], "lstm": [10, 1], "transformer": [9, 1], "mamba": [4, 6]}
N_HALF = 50
N_PAIRS = 25
DEGENERATE_GAP_THRESHOLD = 0.05


def load_bcm_best_layers():
    d = json.loads((RESULTS / "phase2_p3_layer_sweep.json").read_text())
    best = {}
    for m in d["models"]:
        bcm = {int(l): v["selectivity"] for l, v in m["balanced_content_match_probe_by_layer"].items()}
        best_layer = max(bcm, key=bcm.get)
        best[(m["arch"], m["seed"])] = {"best_layer": best_layer, "all_layer_selectivity": bcm}
    return best


def run_rnn_lstm(arch, seed, pairs, tok2idx, best_layer):
    from rnn_patching import RNNPatchingHarness, RecurrentScan, SITE_RNN, SITE_LSTM_FULL
    h = RNNPatchingHarness(arch, f"rec+ns/validation-short/{seed}", task=TASK)
    sc = RecurrentScan(h)
    site = SITE_RNN if arch == "rnn" else SITE_LSTM_FULL

    wiring, layer_specific, behavioral = [], [], []
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

        clean_layer_state = sc.record_layer_state(clean_idx, best_layer, t_ro)
        patched_layer = sc.patch_logit_at_layer(corrupt_idx, t_ro, best_layer, clean_layer_state, dims=None)
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
        t_ro = h.readout_position(p["clean"])
        assert t_ro == h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))

        patched_full = mamba_patch_relative(h, clean_idx, corrupt_idx, top_layer, t_ro, t_ro, dims=None)
        wiring.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})

        patched_layer = mamba_patch_relative(h, clean_idx, corrupt_idx, best_layer, t_ro, t_ro, dims=None)
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
        t_ro = len(p["clean"]) - 1
        assert t_ro == len(p["corrupt"]) - 1
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))
        denom = lc - lo

        patched_full = transformer_patch_relative(h, clean_idx, corrupt_idx, top_layer, t_ro, t_ro, dims=None)
        rf_full = (patched_full - lo) / denom if abs(denom) > 1e-9 else float("nan")
        wiring.append({"restored_fraction": rf_full})

        patched_layer = transformer_patch_relative(h, clean_idx, corrupt_idx, best_layer, t_ro, t_ro, dims=None)
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


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    pairs = markedrev_mirror_pairs(N_HALF, N_PAIRS, rng)
    audit = audit_markedrev_pairs(pairs)
    print("=== mirror_pairs (balanced_content_match) audit ===")
    print(json.dumps(audit, indent=2))
    assert audit["target_property_isolated"]

    best_layers = load_bcm_best_layers()
    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    cells = []
    for arch, seeds in SEED_PAIRS.items():
        for seed in seeds:
            bl_info = best_layers[(arch, seed)]
            best_layer = bl_info["best_layer"]
            print(f"\n--- {arch} seed{seed} (bcm best_layer={best_layer}) ---", flush=True)
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
            surprising = gap >= DEGENERATE_GAP_THRESHOLD
            cell = {
                "arch": arch, "seed": seed, "checkpoint": ckpt,
                "balanced_content_match_best_layer": best_layer,
                "layer_selectivity_from_sweep": bl_info["all_layer_selectivity"],
                "PRIMARY_behavioral_logit_gap": beh,
                "wiring_check_full_state_patch_SECONDARY": wiring_s,
                "layer_specific_patch_at_best_layer": layer_s,
                "layer_specific_rf_is_meaningful": rf_meaningful,
                "SURPRISING_above_noise_content_sensitivity": surprising,
            }
            if not rf_meaningful:
                cell["layer_specific_patch_caveat"] = (
                    f"behavioral gap ({gap:.4f}) is degenerate/near-zero -- layer_specific rf is not "
                    f"meaningful; near-zero gap IS the finding (no content verification, as predicted)."
                )
            cells.append(cell)
            flag = "  <<< SURPRISING -- ABOVE NOISE" if surprising else ""
            print(f"  PRIMARY behavioral gap: mean={gap:.4f} (ci95={beh['ci95_abs_clean_corrupt_logit_gap']:.4f}, n={beh['n']}){flag}", flush=True)
            print(f"  layer{best_layer}-specific patch: rf_mean={layer_s['layer_specific_restored_fraction_mean']:.4f} "
                  f"meaningful={rf_meaningful}", flush=True)

    any_surprising = [c for c in cells if c["SURPRISING_above_noise_content_sensitivity"]]
    all_null = len(any_surprising) == 0

    # lstm seed10 fourth-null confirmation
    lstm10 = next(c for c in cells if c["arch"] == "lstm" and c["seed"] == 10)
    lstm10_gap = lstm10["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    seed10_fourth_null = {
        "content_gap": lstm10_gap,
        "confirms_marker_count_only": lstm10_gap < DEGENERATE_GAP_THRESHOLD,
        "interpretation": (
            f"lstm seed10's balanced_content_match gap ({lstm10_gap:.4f}) is "
            + ("near-zero, the FOURTH independent null (alongside marker_position, length_parity, "
               "balance) -- marker_count-only attribution is now fully confirmed: seed10 is "
               "insensitive to position, parity, balance, AND content, responding only to marker "
               "count." if lstm10_gap < DEGENERATE_GAP_THRESHOLD else
               "SURPRISINGLY non-null, contradicting the marker_count-only attribution -- requires "
               "re-investigation, not smoothing over.")
        ),
    }
    print("\n=== lstm seed10 fourth-null confirmation ===")
    print(json.dumps(seed10_fourth_null, indent=2))

    out = {
        "condition": "condition_3_balanced_content_match",
        "status": "CONFIRMATORY null test, applied to all 8 cells (previously only behaviorally "
                 "spot-checked on rnn seed3 during the Phase 3 design step). PRIMARY evidence is the "
                 "behavioral logit gap; full-state wiring check is secondary/reference only.",
        "description": "Tests genuine mirror-content verification via mirror_pairs (content flipped, "
                       "is_balanced/marker_count held constant) for all 4 architectures x 2 seeds. "
                       "Predicted: near-zero effect everywhere, matching the corrected Phase 2 "
                       "balanced_content_match probe (selectivity = 0.0 at every layer, every seed, "
                       "all 4 architectures).",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "mirror_pairs_audit": audit,
        "cells": cells,
        "all_cells_null": all_null,
        "surprising_cells": any_surprising,
        "lstm_seed10_fourth_null_confirmation": seed10_fourth_null,
    }
    out_path = RESULTS / "phase3_condition3_balanced_content_match.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")
    print(f"\nALL CELLS NULL: {all_null}")


if __name__ == "__main__":
    main()
