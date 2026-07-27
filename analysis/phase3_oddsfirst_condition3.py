"""Odds-first Phase 3, condition 3 of 3 (confirmatory null test):
target_computation_balanced_case. Tests whether genuine odds-first-transform
verification -- held is_balanced/marker_count/marker_position constant,
only CONTENT varied -- causally drives the accept/reject decision, all 8
cells.

Predicted (per Stage 3's probe, which is ~0.0000-0.0007 at every layer,
every cell -- the cleanest null in the whole three-task pilot): near-zero
behavioral gap and non-meaningful layer-specific restored fraction
everywhere, matching marked-reversal/marked-copy precedent.

Given condition 2's finding that lstm seed4/seed5 causally use BOTH marker
features AND BOTH structural features at large magnitude (a genuine multi-
feature convergence on RNN's own mechanism), this condition gives explicit
extra scrutiny to those two cells specifically: if target_computation also
shows a non-trivial causal effect there, that would be the pilot's FIRST
cell showing causal use of actual target computation, not just structural
shortcuts -- a materially different finding from every other cell/task
probed so far. Prediction remains near-zero (per precedent), but this is
tested directly, not assumed.

Same PRIMARY/SECONDARY structure as conditions 1-2.

Saves analysis_outputs/final_results/phase3_oddsfirst_condition3.json.

PYTHONPATH=src:analysis python analysis/phase3_oddsfirst_condition3.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_oddsfirst_counterfactuals import target_computation_pairs, audit_target_computation_pairs
from phase3_oddsfirst_condition2 import mamba_patch_relative, transformer_patch_relative

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
LSTM_SPECIAL_ATTENTION = {("lstm", 4), ("lstm", 5)}


def load_best_layers():
    d = json.loads((RESULTS / "phase2_oddsfirst_structural_probes.json").read_text())
    best = {}
    for m in d["models"]:
        sels = {int(l): v["selectivity"] for l, v in m["target_computation_balanced_case_probe_by_layer"].items()}
        bl = max(sels, key=sels.get)
        best[(m["arch"], m["seed"])] = {"best_layer": bl, "all_layer_selectivity": sels}
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

    pairs = target_computation_pairs(N_HALF, N_PAIRS, rng)
    audit = audit_target_computation_pairs(pairs)
    print("=== target_computation_pairs audit ===")
    print(json.dumps(audit, indent=2))
    assert audit["target_property_isolated"]

    best_layers = load_best_layers()
    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    cells = []
    for arch, seed, role in CELLS:
        bl_info = best_layers[(arch, seed)]
        best_layer = bl_info["best_layer"]
        print(f"\n--- {arch} seed{seed} ({role}, target_computation best_layer={best_layer}) ---", flush=True)
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
            "arch": arch, "seed": seed, "role": role, "checkpoint": ckpt,
            "target_computation_best_layer": best_layer,
            "layer_selectivity_from_sweep": bl_info["all_layer_selectivity"],
            "PRIMARY_behavioral_logit_gap": beh,
            "wiring_check_full_state_patch_SECONDARY": wiring_s,
            "layer_specific_patch_at_best_layer": layer_s,
            "layer_specific_rf_is_meaningful": rf_meaningful,
            "SURPRISING_above_noise_target_computation_sensitivity": surprising,
            "flagged_lstm_convergence_cell": (arch, seed) in LSTM_SPECIAL_ATTENTION,
        }
        if not rf_meaningful:
            cell["layer_specific_patch_caveat"] = (
                f"behavioral gap ({gap:.4f}) is degenerate/near-zero -- layer_specific rf is not "
                f"meaningful; near-zero gap IS the finding (no target-computation verification, as "
                f"predicted)."
            )
        cells.append(cell)
        flag = "  <<< SURPRISING -- ABOVE NOISE" if surprising else ""
        special = "  [LSTM CONVERGENCE CELL]" if cell["flagged_lstm_convergence_cell"] else ""
        print(f"  PRIMARY behavioral gap: mean={gap:.4f} (ci95={beh['ci95_abs_clean_corrupt_logit_gap']:.4f}, n={beh['n']}){flag}{special}", flush=True)
        print(f"  layer{best_layer}-specific patch: rf_mean={layer_s['layer_specific_restored_fraction_mean']:.4f} "
              f"meaningful={rf_meaningful}", flush=True)
        print(f"  full-state wiring check (secondary): rf_mean={wiring_s['wiring_restored_fraction_mean']:.4f}", flush=True)

    any_surprising = [c for c in cells if c["SURPRISING_above_noise_target_computation_sensitivity"]]
    all_null = len(any_surprising) == 0

    # ------------------------------------------------------------------
    # lstm seed4/seed5 special-attention report: does the multi-feature
    # convergence extend to genuine target-computation verification?
    # ------------------------------------------------------------------
    lstm_report = {}
    for arch, seed in LSTM_SPECIAL_ATTENTION:
        c = next(x for x in cells if x["arch"] == arch and x["seed"] == seed)
        gap = c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
        is_above_noise = gap >= DEGENERATE_GAP_THRESHOLD
        lstm_report[f"{arch}_seed{seed}"] = {
            "target_computation_gap": gap,
            "above_noise": is_above_noise,
            "interpretation": (
                f"SUBSTANTIAL FINDING -- target_computation_balanced_case gap ({gap:.4f}) is above "
                f"noise for this cell, which condition 2 already showed causally uses BOTH marker "
                f"features AND BOTH structural features at large magnitude. This would be the "
                f"PILOT'S FIRST cell showing causal sensitivity to actual target computation, not "
                f"just structural shortcuts -- a materially different finding from every other cell/"
                f"task probed so far, warranting its own dedicated follow-up before treating it as "
                f"established."
                if is_above_noise else
                f"Convergence does NOT extend to target computation -- gap ({gap:.4f}) is null, "
                f"matching every other cell in the pilot. LSTM's multi-feature convergence (marker + "
                f"structural, condition 1/2) is comprehensive across every SHORTCUT feature tested, "
                f"but does not include genuine target-computation verification -- the shortcut-"
                f"exploitation picture remains intact even for this unusually shortcut-rich cell."
            ),
        }
    print("\n=== lstm seed4/seed5 special-attention report ===")
    print(json.dumps(lstm_report, indent=2, default=str))

    out = {
        "condition": "condition_3_target_computation_null_test",
        "status": "CONFIRMATORY null test, all 8 cells. PRIMARY evidence is the behavioral logit "
                 "gap; full-state wiring check is secondary/reference only. lstm seed4/seed5 get "
                 "explicit extra scrutiny given condition 2's finding that they causally use every "
                 "other probed feature (marker + structural) at large magnitude.",
        "description": "Tests genuine odds-first-transform verification via target_computation_pairs "
                       "(one first-half content bit flipped, marker_count/position/is_balanced held "
                       "constant) for all 8 cells. Predicted: near-zero effect everywhere, matching "
                       "Stage 3's target_computation_balanced_case probe (selectivity <=0.0007 at "
                       "every layer, every cell -- the cleanest null in the three-task pilot).",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "target_computation_pairs_audit": audit,
        "sample_pairs": pairs[:3],
        "cells": cells,
        "all_cells_null": all_null,
        "surprising_cells": any_surprising,
        "lstm_convergence_cells_special_attention_report": lstm_report,
    }
    out_path = RESULTS / "phase3_oddsfirst_condition3.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")
    print(f"\nALL CELLS NULL: {all_null}")


if __name__ == "__main__":
    main()
