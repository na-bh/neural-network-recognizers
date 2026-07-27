"""Phase 3, patch condition 2 of 3: length parity. Tests RNN's refined
structural shortcut (sequence-length parity, odd/even token count) causally,
using length_parity_pairs (clean = genuine positive, odd length; corrupt =
clean + one appended token, even length -- isolates length parity while
keeping marker_count=1 and content otherwise identical, per phase3_
counterfactual_design.json).

Per user instruction, this condition reports TWO things per cell, with the
BEHAVIORAL LOGIT GAP as the primary mechanistic evidence (not the full-state
wiring check, which -- like condition 1 -- trivially gives rf~1.0 whenever
non-degenerate, and cannot attribute causality to length parity specifically):
  (a) full-state patch (wiring check, secondary, confirms harness plumbing)
  (b) LAYER-SPECIFIC patch at each cell's own best length_parity layer (from
      phase2_p3_layer_sweep.json), using the newly-implemented
      RecurrentScan.patch_logit_at_layer for RNN/LSTM and decoupled-position
      single-layer hooks for Mamba/Transformer (clean/corrupt have DIFFERENT
      lengths here, so read-position and write-position must be handled
      independently -- off-the-shelf patch_run/patch_logit assume the same
      absolute position for both and would silently misalign).

LSTM seed10 vs seed1 are broken out EXPLICITLY (not averaged) per user
instruction, to test the seed-dependent-mechanism hypothesis: seed10's
condition-1 marker_position gap was near-zero (like rnn) while seed1's was
large (like mamba/transformer) -- if that pattern is a real mechanism split
(not noise), seed10 should show a LARGE length_parity gap (confirming it
uses length parity like rnn) while seed1 should show a SMALL length_parity
gap (confirming it does not).

Saves analysis_outputs/final_results/phase3_causal_patching_condition2.json.
STOP after this condition for review before condition 3 (balanced_content_match).

PYTHONPATH=src:analysis python analysis/phase3_condition2_length_parity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_counterfactual_design import (
    markedrev_length_parity_pairs, audit_markedrev_length_parity_pairs,
)

RESULTS = Path("analysis_outputs/final_results")
TASK = "marked-reversal"
SEED_PAIRS = {"rnn": [3, 7], "lstm": [10, 1], "transformer": [9, 1], "mamba": [4, 6]}
N_HALF = 50
N_PAIRS = 25


def load_best_layers():
    d = json.loads((RESULTS / "phase2_p3_layer_sweep.json").read_text())
    best = {}
    for m in d["models"]:
        lp = {int(l): v["selectivity"] for l, v in m["length_parity_probe_by_layer"].items()}
        best_layer = max(lp, key=lp.get)
        best[(m["arch"], m["seed"])] = {"best_layer": best_layer, "all_layer_selectivity": lp}
    return best


# ---------------------------------------------------------------------------
# decoupled-position single-layer patch helpers for mamba/transformer
# (needed because clean is one token SHORTER than corrupt here -- off-the-
# shelf patch_run/patch_logit assume clean and corrupt share one position)
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

        # (a) full-state wiring check
        _, states = sc.record(clean_idx, [t_clean])
        patched_full = sc.patch_logit(corrupt_idx, t_corrupt, states[t_clean], site=site, dims=None)
        wiring.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched_full,
                        "restored_fraction": h.restored_fraction(lc, lo, patched_full)})

        # (b) layer-specific patch at this cell's best length_parity layer
        clean_layer_state = sc.record_layer_state(clean_idx, best_layer, t_clean)
        patched_layer = sc.patch_logit_at_layer(corrupt_idx, t_corrupt, best_layer, clean_layer_state, dims=None)
        layer_specific.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched_layer,
                                "restored_fraction": h.restored_fraction(lc, lo, patched_layer)})
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
        wiring.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched_full,
                        "restored_fraction": h.restored_fraction(lc, lo, patched_full)})

        patched_layer = mamba_patch_relative(h, clean_idx, corrupt_idx, best_layer, t_clean, t_corrupt, dims=None)
        layer_specific.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched_layer,
                                "restored_fraction": h.restored_fraction(lc, lo, patched_layer)})
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

        patched_full = transformer_patch_relative(h, clean_idx, corrupt_idx, top_layer, t_clean, t_corrupt, dims=None)
        rf_full = (patched_full - lo) / (lc - lo) if abs(lc - lo) > 1e-9 else float("nan")
        wiring.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched_full,
                        "restored_fraction": rf_full})

        patched_layer = transformer_patch_relative(h, clean_idx, corrupt_idx, best_layer, t_clean, t_corrupt, dims=None)
        rf_layer = (patched_layer - lo) / (lc - lo) if abs(lc - lo) > 1e-9 else float("nan")
        layer_specific.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched_layer,
                                "restored_fraction": rf_layer})
    return wiring, layer_specific, behavioral, f"data/models/{TASK}/transformer/rec+ns/validation-short/{seed}"


def summarize(results, key):
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
        "std_abs_clean_corrupt_logit_gap": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
        "ci95_abs_clean_corrupt_logit_gap": float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan"),
        "n": len(arr),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    pairs = markedrev_length_parity_pairs(N_HALF, N_PAIRS, rng)
    audit = audit_markedrev_length_parity_pairs(pairs)
    print("=== length_parity_pairs audit ===")
    print(json.dumps(audit, indent=2))
    assert audit["target_property_isolated"]

    best_layers = load_best_layers()
    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    cells = []
    for arch, seeds in SEED_PAIRS.items():
        for seed in seeds:
            bl_info = best_layers[(arch, seed)]
            best_layer = bl_info["best_layer"]
            print(f"\n--- {arch} seed{seed} (best_layer={best_layer}) ---", flush=True)
            if arch in ("rnn", "lstm"):
                wiring, layer_specific, behavioral, ckpt = run_rnn_lstm(arch, seed, pairs, tok2idx, best_layer)
            elif arch == "mamba":
                wiring, layer_specific, behavioral, ckpt = run_mamba(seed, pairs, tok2idx, best_layer)
            else:
                wiring, layer_specific, behavioral, ckpt = run_transformer(seed, pairs, tok2idx, best_layer)

            beh_summary = summarize_behavioral(behavioral)
            wiring_summary = summarize(wiring, "wiring")
            layer_summary = summarize(layer_specific, "layer_specific")
            cell = {
                "arch": arch, "seed": seed, "checkpoint": ckpt,
                "best_length_parity_layer": best_layer,
                "layer_selectivity_from_sweep": bl_info["all_layer_selectivity"],
                "PRIMARY_behavioral_logit_gap": beh_summary,
                "wiring_check_full_state_patch_SECONDARY": wiring_summary,
                "layer_specific_patch_at_best_layer": layer_summary,
                "raw_layer_specific_restored_fractions": [
                    float(r["restored_fraction"]) for r in layer_specific if not np.isnan(r["restored_fraction"])
                ],
            }
            cells.append(cell)
            print(f"  PRIMARY behavioral gap: mean={beh_summary['mean_abs_clean_corrupt_logit_gap']:.4f} "
                  f"(ci95={beh_summary['ci95_abs_clean_corrupt_logit_gap']:.4f}, n={beh_summary['n']})", flush=True)
            print(f"  layer{best_layer}-specific patch: rf_mean={layer_summary['layer_specific_restored_fraction_mean']:.4f} "
                  f"(ci95={layer_summary['layer_specific_restored_fraction_ci95']:.4f}, "
                  f"n={layer_summary['layer_specific_n_nondegenerate_denom']})", flush=True)
            print(f"  full-state wiring check (secondary): rf_mean={wiring_summary['wiring_restored_fraction_mean']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # LSTM seed10 vs seed1 explicit breakout (not averaged), testing the
    # seed-dependent-mechanism hypothesis against condition 1's marker_
    # position result
    # ------------------------------------------------------------------
    lstm10 = next(c for c in cells if c["arch"] == "lstm" and c["seed"] == 10)
    lstm1 = next(c for c in cells if c["arch"] == "lstm" and c["seed"] == 1)
    cond1 = json.loads((RESULTS / "phase3_causal_patching.json").read_text())
    mp_cells = {(c["arch"], c["seed"]): c for c in cond1["cells"] if c["pair_type"] == "marker_position"}
    lstm10_mp_gap = mp_cells[("lstm", 10)]["mean_abs_clean_corrupt_logit_gap"]
    lstm1_mp_gap = mp_cells[("lstm", 1)]["mean_abs_clean_corrupt_logit_gap"]

    lstm10_lp_gap = lstm10["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    lstm1_lp_gap = lstm1["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]

    seed10_pattern = "length_parity" if lstm10_lp_gap > lstm10_mp_gap else "marker_position"
    seed1_pattern = "length_parity" if lstm1_lp_gap > lstm1_mp_gap else "marker_position"
    hypothesis_confirmed = (seed10_pattern == "length_parity" and seed1_pattern == "marker_position")

    lstm_seed_split = {
        "hypothesis": "seed10's mechanism is length-parity-based (like rnn); seed1's mechanism is "
                     "marker-position-based (like mamba/transformer) -- a genuine seed-dependent "
                     "mechanism split, not noise to average away.",
        "condition_1_marker_position_gap": {"seed10": lstm10_mp_gap, "seed1": lstm1_mp_gap},
        "condition_2_length_parity_gap": {"seed10": lstm10_lp_gap, "seed1": lstm1_lp_gap},
        "seed10_dominant_mechanism_by_gap_comparison": seed10_pattern,
        "seed1_dominant_mechanism_by_gap_comparison": seed1_pattern,
        "hypothesis_confirmed": hypothesis_confirmed,
        "interpretation": (
            f"seed10: length_parity gap={lstm10_lp_gap:.4f} vs marker_position gap={lstm10_mp_gap:.4f} "
            f"-> dominant mechanism = {seed10_pattern}. "
            f"seed1: length_parity gap={lstm1_lp_gap:.4f} vs marker_position gap={lstm1_mp_gap:.4f} "
            f"-> dominant mechanism = {seed1_pattern}. "
            + ("CONFIRMED: LSTM's shortcut mechanism is seed-dependent -- some seeds (10) converge on "
               "RNN's refined length-parity shortcut, others (1) converge on the simpler marker-position "
               "shortcut shared with mamba/transformer. This is consistent with the paper's seed-level "
               "mechanism variation story and should NOT be averaged into a single 'LSTM' number."
               if hypothesis_confirmed else
               "NOT CONFIRMED in this simple form -- see raw gaps above; report the actual pattern found, "
               "not the hypothesized one.")
        ),
    }
    print("\n=== LSTM seed10 vs seed1 mechanism-split analysis ===")
    print(json.dumps(lstm_seed_split, indent=2, default=str))

    out = {
        "condition": "condition_2_length_parity",
        "status": "PRIMARY evidence is the behavioral logit gap on length_parity_pairs and the "
                 "LAYER-SPECIFIC patch at each cell's own best length_parity layer (from the P3 "
                 "layer sweep) -- NOT the full-state wiring check, which is reported for reference "
                 "only (kept in wiring_check_full_state_patch_SECONDARY) and is expected to be "
                 "~1.0 whenever the behavioral gap is non-degenerate, exactly as in condition 1.",
        "description": "Tests length_parity (sequence length mod 2) causally via length_parity_pairs "
                       "(clean = odd-length genuine positive; corrupt = same content + 1 appended "
                       "token, even length; marker_count=1 in both). Predicted: rnn large gap and "
                       "high layer-specific restored fraction at its identified layer (1 for seed3, "
                       "4 for seed7); lstm/mamba/transformer minimal effect UNLESS the seed-split "
                       "hypothesis below holds for a given lstm seed.",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "length_parity_pairs_audit": audit,
        "sample_pairs": pairs[:3],
        "cells": cells,
        "lstm_seed10_vs_seed1_mechanism_split": lstm_seed_split,
    }
    out_path = RESULTS / "phase3_causal_patching_condition2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
