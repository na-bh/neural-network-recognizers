"""Cycle-navigation Phase 3, condition 2 of 2: true_position (null test).
Tests whether the residual true_position signal found in Phase 2 (selectivity
0.037-0.072 for rnn/lstm/transformer, MLP-confirmed genuine, near-zero for
mamba) causally drives the accept/reject decision, or is a representationally-
detectable byproduct that never routes to the readout decision.

CONSTRUCTION AUDIT (required before patching, per standard discipline):
cyclenav_position_pairs (single late-move-flip) changes true_position AND
symbol_counts SIMULTANEOUSLY, unlike marked-reversal's confounds (which were
FIXABLE construction bugs). This one is NOT fixable by better input editing:
true_position = (count('>') - count('<')) mod 5 is a DETERMINISTIC function
of symbol_counts for this task, so no token-level edit can change
true_position while holding symbol_counts exactly fixed (verified in
phase3_counterfactual_design.py's audit_cyclenav_pairs, "symbol_counts_also_
necessarily_differ" field). Because the confound is a mathematical necessity
of the task, not a bug, this condition uses the escape hatch already specced
in the original Phase 3 design: a PROBE-DIRECTION SUBSPACE PATCH, projecting
onto the row-space of each cell's persisted true_position probe (fit on the
confound-free negatives-only subset, see phase2_p2_cyclenav.json), patching
ONLY that linear direction and leaving the orthogonal complement (which
carries symbol_counts-relevant information) at the corrupt run's own value.
Implementation validated against P=identity (matches full-state patch
exactly) and P=zero (exact no-op) for all 3 harness types before use --
see inline validation in the session, not re-run here.

Because position_pairs' own behavioral gap is near-zero BY DESIGN (Phase 1's
own finding: the model doesn't distinguish clean/corrupt), restored_fraction
is numerically degenerate for EVERY cell in this condition (denominator ~0).
Primary metric is therefore the ABSOLUTE logit shift from the subspace patch
(|patched - corrupt|), reported against condition 1's grammar_pairs logit-
swing scale (3.5-9.4) as the reference for "meaningful effect" -- exactly as
specced in the original design doc.

Saves analysis_outputs/final_results/phase3_cyclenav_condition2.json.

PYTHONPATH=src:analysis python analysis/phase3_cyclenav_condition2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_counterfactual_design import cyclenav_position_pairs, audit_cyclenav_pairs
from phase3_subspace_patch_utils import (
    load_probe_subspace, mamba_patch_relative_subspace, transformer_patch_relative_subspace,
)

RESULTS = Path("analysis_outputs/final_results")
CKPT_DIR = RESULTS / "phase2_checkpoints"
TASK = "cycle-navigation"
SEED_PAIRS = {"rnn": [1, 4], "lstm": [2, 3], "transformer": [6, 9], "mamba": [0, 7]}
M_MOVES = 19
N_PAIRS = 25
DEGENERATE_GAP_THRESHOLD = 0.05
SURPRISE_ABS_DELTA_THRESHOLD = 0.05
CANONICAL_LAYER = {"rnn": 0, "lstm": 0, "mamba": 4, "transformer": 4}

# reference scale from condition 1 (grammar_pairs behavioral gaps, per cell)
CONDITION1_GAP_REF = {
    ("rnn", 1): 5.4309, ("rnn", 4): 3.5508, ("lstm", 2): 4.4858, ("lstm", 3): 4.5609,
    ("transformer", 6): 7.7331, ("transformer", 9): 6.4320, ("mamba", 0): 9.1288, ("mamba", 7): 9.4247,
}


def run_rnn_lstm(arch, seed, pairs, layer_idx, P):
    from rnn_patching import RNNPatchingHarness, RecurrentScan, SITE_RNN, SITE_LSTM_FULL
    h = RNNPatchingHarness(arch, f"rec+ns/validation-short/{seed}", task=TASK)
    sc = RecurrentScan(h)
    site = SITE_RNN if arch == "rnn" else SITE_LSTM_FULL

    wiring, subspace, behavioral = [], [], []
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
        patched_sub = sc.patch_logit_at_layer_subspace(corrupt, t_ro, layer_idx, clean_layer_state, P)
        subspace.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched_sub,
                          "abs_delta": abs(patched_sub - lo)})
    return wiring, subspace, behavioral, f"data/models/{TASK}/{arch}/rec+ns/validation-short/{seed}"


def run_mamba(seed, pairs, layer_idx, P):
    from patching_harness import PatchingHarness
    from phase3_condition2_length_parity import mamba_patch_relative
    h = PatchingHarness(task=TASK, model_subdir=f"mamba/rec+ns/validation-short/{seed}")

    wiring, subspace, behavioral = [], [], []
    for p in pairs:
        clean, corrupt = p["clean"], p["corrupt"]
        t_ro = len(clean) - 1
        assert t_ro == len(corrupt) - 1
        lc = h.logit_diff(clean)
        lo = h.logit_diff(corrupt)
        behavioral.append(abs(lc - lo))

        patched_full = mamba_patch_relative(h, clean, corrupt, layer_idx, t_ro, t_ro, dims=None)
        wiring.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})

        patched_sub = mamba_patch_relative_subspace(h, clean, corrupt, layer_idx, t_ro, t_ro, P)
        subspace.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched_sub,
                          "abs_delta": abs(patched_sub - lo)})
    return wiring, subspace, behavioral, f"models/{TASK}/mamba/rec+ns/validation-short/{seed}"


def run_transformer(seed, pairs, layer_idx, P):
    from modk_transformer_harness import TransformerPatchingHarness
    from phase3_condition2_length_parity import transformer_patch_relative
    h = TransformerPatchingHarness(TASK, f"rec+ns/validation-short/{seed}")

    wiring, subspace, behavioral = [], [], []
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

        patched_sub = transformer_patch_relative_subspace(h, clean, corrupt, layer_idx, t_ro, t_ro, P)
        subspace.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched_sub,
                          "abs_delta": abs(patched_sub - lo)})
    return wiring, subspace, behavioral, f"data/models/{TASK}/transformer/rec+ns/validation-short/{seed}"


def summarize_rf(results, key):
    rfs = np.array([r["restored_fraction"] for r in results if not np.isnan(r["restored_fraction"])])
    n = len(rfs)
    mean = float(np.mean(rfs)) if n else float("nan")
    ci95 = float(1.96 * np.std(rfs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {f"{key}_restored_fraction_mean": mean, f"{key}_restored_fraction_ci95": ci95, f"{key}_n": n}


def summarize_behavioral(behavioral):
    arr = np.array(behavioral)
    return {
        "mean_abs_clean_corrupt_logit_gap": float(np.mean(arr)),
        "ci95_abs_clean_corrupt_logit_gap": float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan"),
        "n": len(arr),
    }


def summarize_subspace(subspace):
    arr = np.array([r["abs_delta"] for r in subspace])
    n = len(arr)
    mean = float(np.mean(arr))
    ci95 = float(1.96 * np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean_abs_delta_logit_subspace_patch": mean, "ci95_abs_delta_logit_subspace_patch": ci95, "n": n}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    pairs = cyclenav_position_pairs(M_MOVES, N_PAIRS, rng)
    audit = audit_cyclenav_pairs(pairs, "position")
    print("=== position_pairs (true_position) audit ===")
    print(json.dumps(audit, indent=2, default=str))
    assert audit["same_length"] and audit["differs_only_at_flip_idx"] and audit["true_position_differs"]
    confound_confirmed = audit["symbol_counts_also_necessarily_differ"] is not None
    print(f"\nCONSTRUCTION CONFOUND CONFIRMED (mathematically necessary, not fixable via input editing): "
          f"{confound_confirmed}")
    print("Resolution: probe-direction subspace patch (see module docstring).")

    cells = []
    for arch, seeds in SEED_PAIRS.items():
        layer_idx = CANONICAL_LAYER[arch]
        for seed in seeds:
            P, probe_meta = load_probe_subspace(CKPT_DIR / f"phase2_p2_{arch}_seed{seed}_true_position.pt")
            print(f"\n--- {arch} seed{seed} (canonical layer={layer_idx}, probe subspace rank={np.linalg.matrix_rank(P)}) ---", flush=True)
            if arch in ("rnn", "lstm"):
                wiring, subspace, behavioral, ckpt = run_rnn_lstm(arch, seed, pairs, layer_idx, P)
            elif arch == "mamba":
                wiring, subspace, behavioral, ckpt = run_mamba(seed, pairs, layer_idx, P)
            else:
                wiring, subspace, behavioral, ckpt = run_transformer(seed, pairs, layer_idx, P)

            beh = summarize_behavioral(behavioral)
            wiring_s = summarize_rf(wiring, "wiring")
            sub_s = summarize_subspace(subspace)
            gap = beh["mean_abs_clean_corrupt_logit_gap"]
            rf_meaningful = gap >= DEGENERATE_GAP_THRESHOLD

            ref_scale = CONDITION1_GAP_REF[(arch, seed)]
            sub_mean = sub_s["mean_abs_delta_logit_subspace_patch"]
            sub_ci = sub_s["ci95_abs_delta_logit_subspace_patch"]
            lower_bound = sub_mean - sub_ci
            surprising = lower_bound > SURPRISE_ABS_DELTA_THRESHOLD
            ratio_to_ref = sub_mean / ref_scale

            cell = {
                "arch": arch, "seed": seed, "checkpoint": ckpt, "canonical_layer": layer_idx,
                "probe_checkpoint": probe_meta["checkpoint"], "probe_subspace_rank": int(np.linalg.matrix_rank(P)),
                "PRIMARY_behavioral_logit_gap_full_edit": beh,
                "wiring_check_full_state_patch_SECONDARY": wiring_s,
                "layer_specific_rf_is_meaningful": rf_meaningful,
                "true_position_subspace_patch_PRIMARY_METRIC": sub_s,
                "condition1_grammar_pairs_reference_scale": ref_scale,
                "subspace_effect_ratio_to_condition1_reference": ratio_to_ref,
                "SURPRISING_above_noise_true_position_causal_effect": surprising,
            }
            if not rf_meaningful:
                cell["restored_fraction_caveat"] = (
                    "position_pairs' behavioral gap is near-zero BY DESIGN (this is what makes it a "
                    "valid null-test construction, matching Phase 1's near-zero hard-negative accuracy) "
                    "-- restored_fraction is therefore degenerate for every cell in this condition; use "
                    "the subspace patch's absolute |delta_logit| as the causal-effect metric instead."
                )
            cells.append(cell)
            flag = "  <<< SURPRISING -- ABOVE-NOISE true_position CAUSAL EFFECT" if surprising else ""
            print(f"  behavioral gap (full edit): mean={gap:.4f} (expected near-zero by construction)", flush=True)
            print(f"  subspace patch |delta_logit|: mean={sub_mean:.4f} (ci95={sub_ci:.4f}) "
                  f"ratio_to_cond1_ref={ratio_to_ref:.4f}{flag}", flush=True)

    any_surprising = [c for c in cells if c["SURPRISING_above_noise_true_position_causal_effect"]]

    out = {
        "task": "cycle-navigation",
        "condition": "condition_2_true_position_null_test",
        "status": "Construction audit CONFIRMED a mathematically-necessary confound (true_position is "
                 "a deterministic function of symbol_counts, so input-level editing cannot isolate one "
                 "from the other) -- resolved via probe-direction SUBSPACE patching (projecting onto "
                 "each cell's persisted true_position probe direction, raw-activation-space-corrected "
                 "for the StandardScaler), leaving the orthogonal complement (symbol_counts-relevant "
                 "information) at the corrupt run's own value. Validated against P=identity (exact "
                 "match to full-state patch) and P=zero (exact no-op) for all 3 harness types before "
                 "use. PRIMARY metric is |delta_logit| from the subspace patch (restored_fraction is "
                 "degenerate here since position_pairs' own behavioral gap is near-zero by design).",
        "description": "Tests whether the residual true_position probe signal (Phase 2: 0.037-0.072 "
                       "selectivity for rnn/lstm/transformer, MLP-confirmed genuine; near-zero for "
                       "mamba) causally drives accept/reject, or is a decodable-but-not-causally-used "
                       "byproduct. Predicted: near-zero |delta_logit| relative to condition 1's "
                       "grammar_pairs reference scale, for all 4 architectures, both seeds.",
        "m_moves": M_MOVES, "n_pairs_generated": N_PAIRS,
        "canonical_layer_by_arch": CANONICAL_LAYER,
        "position_pairs_audit": audit,
        "construction_confound_note": (
            "position_pairs (single late-move-flip) necessarily changes symbol_counts whenever it "
            "changes true_position -- this is a mathematical property of the task (true_position = "
            "net displacement mod 5, a deterministic function of move counts), NOT a fixable "
            "construction bug. Confirmed via audit_cyclenav_pairs' symbol_counts_also_necessarily_"
            "differ field. Resolved via probe-direction subspace patching rather than further input-"
            "level editing attempts."
        ),
        "sample_pairs": pairs[:3],
        "surprise_threshold_abs_delta_logit": SURPRISE_ABS_DELTA_THRESHOLD,
        "cells": cells,
        "any_surprising_cells": [f"{c['arch']}_seed{c['seed']}" for c in any_surprising],
        "prediction_confirmed": len(any_surprising) == 0,
    }
    out_path = RESULTS / "phase3_cyclenav_condition2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")
    print(f"prediction_confirmed (no surprising cells): {len(any_surprising) == 0}")


if __name__ == "__main__":
    main()
