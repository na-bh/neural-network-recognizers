"""Binary-addition Phase 3, condition 1 of 3: structural feature causal
patching (operator_position, segment_length [via u_y/equals-position],
digit_count). Tests whether the structural features Stage 2 found
represented at very different strengths across architectures (Transformer
0.58 mean best-selectivity >> RNN/LSTM ~0.26-0.35 >> Mamba ~0.16-0.29,
with segment-length selectivity showing the SHARPEST architecture split:
Transformer 0.67-0.71 vs RNN/LSTM 0.05-0.17) are ALSO causally load-bearing,
and whether causal strength tracks representation strength (the marker-
family pilot found this relationship messy -- not assumed to hold here).

Pair construction (analysis/phase3_binaryaddition_counterfactuals.py):
operator_position_pairs and segment_length_pairs use VALUE-PRESERVING
zero-padding (clean/corrupt have IDENTICAL u_x, u_y, u_z content and
correctness -- only length/position of one segment differs), so any
behavioral gap here is caused PURELY by the structural/positional
perturbation, not by any change in arithmetic content. digit_count_pairs
holds length/position fixed and varies u_x's Hamming weight (low vs high
ones-count) while keeping both examples genuinely correct sums.

PRIMARY evidence is the behavioral logit gap (unpatched clean vs corrupt).
Full-state patch at the canonical readout site is a SECONDARY wiring
check -- flagged unreliable when the behavioral gap is near zero (patching
a degenerate/near-zero denominator produces meaningless restored-fraction
values).

Saves analysis_outputs/final_results/phase3_binaryaddition_condition1.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_binaryaddition_condition1.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_binaryaddition_counterfactuals import (
    operator_position_pairs, segment_length_pairs, digit_count_pairs,
    audit_padding_pairs, audit_digit_count_pairs,
)

RESULTS = Path("analysis_outputs/final_results")
TASK = "binary-addition"
CELLS = [
    ("rnn", 9, "primary"), ("rnn", 5, "secondary_comparable"),
    ("lstm", 9, "primary"), ("lstm", 10, "secondary_comparable"),
    ("transformer", 8, "primary"), ("transformer", 6, "secondary_comparable"),
    ("mamba", 5, "primary"), ("mamba", 4, "secondary_mild_gap"),
]
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
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        # clean and corrupt may differ in length (padding pairs): the readout
        # position and cached state must come from EACH sequence's own end,
        # so we record clean's own state and patch onto corrupt's own readout.
        t_ro_corrupt = h.readout_position(p["corrupt"])
        _, states = sc.record(clean_idx, [t_ro])
        patched = sc.patch_logit(corrupt_idx, t_ro_corrupt, states[t_ro], site=site, dims=None)
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
        t_ro_clean = h.readout_position(p["clean"])
        t_ro_corrupt = h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
        cv = clean_cache[layer].to(h.device)

        def hook(module, inputs, output, cv=cv, t_ro_corrupt=t_ro_corrupt, t_ro_clean=t_ro_clean):
            out = output.clone()
            out[0, t_ro_corrupt, :] = cv[t_ro_clean, :]
            return out

        handles = [(layer, hook)]
        patched = h._forward(corrupt_idx, hooks=handles)
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
        t_ro_clean = len(p["clean"]) - 1
        t_ro_corrupt = len(p["corrupt"]) - 1
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
        clean_row = clean_cache[layer][t_ro_clean + h.bos_off]
        patched = h.patch_logit(corrupt_idx, layer, t_ro_corrupt, clean_row, dims=None)
        rf = (patched - lo) / (lc - lo) if abs(lc - lo) > 1e-9 else float("nan")
        results.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched,
                         "restored_fraction": rf, "behavioral_gap": abs(lc - lo)})
    return results, f"data/models/{TASK}/transformer/rec+ns/validation-short/{seed}"


def summarize(results, pair_type, arch, seed, role, checkpoint):
    gaps = np.array([r["behavioral_gap"] for r in results])
    gap_mean = float(np.mean(gaps))
    gap_ci95 = float(1.96 * np.std(gaps, ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else float("nan")
    degenerate = gap_mean < DEGENERATE_GAP_THRESHOLD
    rfs = np.array([r["restored_fraction"] for r in results if not np.isnan(r["restored_fraction"])])
    n = len(rfs)
    rf_mean = float(np.mean(rfs)) if n else float("nan")
    rf_ci95 = float(1.96 * np.std(rfs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "arch": arch, "seed": seed, "role": role, "pair_type": pair_type, "checkpoint": checkpoint,
        "PRIMARY_behavioral_logit_gap": {
            "mean_abs_clean_corrupt_logit_gap": gap_mean, "ci95": gap_ci95, "n": len(gaps),
        },
        "causally_effective": bool(gap_mean >= DEGENERATE_GAP_THRESHOLD),
        "wiring_check_full_state_patch_SECONDARY": {
            "restored_fraction_mean": rf_mean, "restored_fraction_ci95": rf_ci95,
            "n_nondegenerate_denom": n,
            "UNRELIABLE_near_zero_behavioral_gap": degenerate,
        },
        "raw_restored_fractions": [float(x) for x in rfs],
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    op_pairs = operator_position_pairs(N_PAIRS, rng)
    op_audit = audit_padding_pairs(op_pairs, "clean_len_x", "corrupt_len_x", 8)
    sl_pairs = segment_length_pairs(N_PAIRS, rng)
    sl_audit = audit_padding_pairs(sl_pairs, "clean_len_y", "corrupt_len_y", 8)
    dc_pairs = digit_count_pairs(N_PAIRS, rng)
    dc_audit = audit_digit_count_pairs(dc_pairs)
    print("=== operator_position_pairs audit ===")
    print(json.dumps(op_audit, indent=2))
    print("=== segment_length_pairs audit ===")
    print(json.dumps(sl_audit, indent=2))
    print("=== digit_count_pairs audit ===")
    print(json.dumps(dc_audit, indent=2))
    assert op_audit["target_property_isolated"] and sl_audit["target_property_isolated"] and dc_audit["target_property_isolated"]

    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    pair_sets = [("operator_position", op_pairs), ("segment_length", sl_pairs), ("digit_count", dc_pairs)]

    cells = []
    for arch, seed, role in CELLS:
        print(f"\n--- {arch} seed{seed} ({role}) ---", flush=True)
        for pair_type, pairs in pair_sets:
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
            print(f"  [{pair_type}] PRIMARY behavioral gap={gap:.4f}  causal={cell['causally_effective']}  |  "
                  f"wiring rf_mean={rf:.4f}", flush=True)

    # ------------------------------------------------------------------
    # representation-strength vs causal-strength cross-check (this
    # condition's specific instruction)
    # ------------------------------------------------------------------
    stage2 = json.loads((RESULTS / "phase2_binaryaddition_structural_probes.json").read_text())
    rep_strength = stage2["structural_representation_strength_summary"]

    def get(arch, seed, pt):
        return next(c for c in cells if c["arch"] == arch and c["seed"] == seed and c["pair_type"] == pt)

    cross_check = []
    for arch, seed, role in CELLS:
        key = f"{arch}_seed{seed}"
        rep = rep_strength[key]["mean_best_selectivity_across_targets"]
        causal_gaps = {pt: get(arch, seed, pt)["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
                       for pt, _ in pair_sets}
        mean_causal = float(np.mean(list(causal_gaps.values())))
        cross_check.append({
            "arch": arch, "seed": seed, "role": role,
            "stage2_mean_representation_selectivity": rep,
            "condition1_causal_gaps": causal_gaps,
            "condition1_mean_causal_gap": mean_causal,
        })
    # rank correlation between representation strength and causal strength across cells
    reps = np.array([c["stage2_mean_representation_selectivity"] for c in cross_check])
    causals = np.array([c["condition1_mean_causal_gap"] for c in cross_check])
    rep_rank = reps.argsort().argsort()
    causal_rank = causals.argsort().argsort()
    spearman_like = float(np.corrcoef(rep_rank, causal_rank)[0, 1]) if len(reps) > 1 else float("nan")

    print("\n=== representation-strength vs causal-strength cross-check ===")
    for c in cross_check:
        print(f"  {c['arch']}_seed{c['seed']}: rep_sel={c['stage2_mean_representation_selectivity']:.3f} "
              f"causal_gap={c['condition1_mean_causal_gap']:.3f}")
    print(f"  rank correlation (representation vs causal strength, 8 cells): {spearman_like:.3f}")

    out = {
        "condition": "condition_1_structural_features",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (canonical readout site per architecture) is SECONDARY, a wiring "
                 "check only, flagged UNRELIABLE when the behavioral gap is near zero.",
        "description": "Tests whether operator_position, segment_length (via u_y padding, isolating "
                       "equals-position/length from operator position), and digit_count causally "
                       "drive the accept/reject decision, all 8 cells. Prediction from Stage 2: "
                       "Transformer strongest (mean sel 0.58), RNN/LSTM moderate (0.26-0.35), Mamba "
                       "weakest (0.16-0.29) -- but representation strength is NOT assumed to predict "
                       "causal strength (the marker-family pilot found this relationship messy).",
        "n_pairs": N_PAIRS,
        "degenerate_gap_threshold": DEGENERATE_GAP_THRESHOLD,
        "pair_construction_audits": {
            "operator_position": op_audit, "segment_length": sl_audit, "digit_count": dc_audit,
        },
        "sample_pairs": {"operator_position": op_pairs[:2], "segment_length": sl_pairs[:2],
                         "digit_count": dc_pairs[:2]},
        "cells": cells,
        "representation_vs_causal_strength_cross_check": {
            "per_cell": cross_check,
            "rank_correlation_8_cells": spearman_like,
            "interpretation": (
                f"Rank correlation between Stage 2 mean structural-representation selectivity and "
                f"Condition 1 mean causal (behavioral-gap) strength across all 8 cells is "
                f"{spearman_like:.3f}. " +
                ("Strong positive association -- representation strength DOES predict causal "
                 "strength for this task's structural features, unlike the marker family's messier "
                 "pattern." if spearman_like >= 0.6 else
                 "Weak/moderate association -- representation strength is only a partial predictor "
                 "of causal strength here, echoing the marker family's messier pattern rather than "
                 "a clean correspondence." if spearman_like >= 0.2 else
                 "Little to no association (or negative) -- representation strength does NOT "
                 "predict causal strength for this task's structural features, replicating the "
                 "marker family's finding that these two measures can decouple.")
            ),
        },
    }
    out_path = RESULTS / "phase3_binaryaddition_condition1.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
