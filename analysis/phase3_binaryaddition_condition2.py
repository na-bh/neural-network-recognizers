"""Binary-addition Phase 3, condition 2 of 3: arithmetic feature causal
patching (carry_state_pos1-4). Tests whether the TRUE carry-in bit at each
position -- computed from u_x/u_y only, never from the claimed u_z -- is
causally load-bearing, independent of Stage 3's (mostly weak/absent) probe
signal for these targets.

Pair construction (analysis/phase3_binaryaddition_counterfactuals.py's
carry_state_pairs): rejection-sampled at FIXED lengths (matching condition
1's LEN_X=LEN_Y=8, LEN_Z=10), BOTH clean and corrupt are genuine correct
sums (target_computation=True for both, so a behavioral gap here cannot be
explained by an overall-correctness confound) -- they differ specifically
in the true carry-in bit at position k.

Two specific things this run watches for, per this condition's approval:
  (1) Mamba seed4 anomaly follow-up (Condition 1 found seed4 causally
      OUTPERFORMING every other cell including Transformer on structural
      features, despite only middling representation strength): does this
      generalize to ALL FOUR carry positions uniformly (generic reactivity/
      hypersensitivity to any activation perturbation) or concentrate at
      specific positions (genuine partial arithmetic computation)?
  (2) Probe-blind-spot replication check: Stage 3 found carry_state_pos1-4
      selectivity was near-zero for RNN/LSTM/mamba-secondary and only
      weakly positive (0.03-0.16, concentrated at pos3/pos4) for Transformer
      and mamba-primary. Any cell showing an above-noise CAUSAL gap despite
      near-zero Stage 3 PROBE selectivity replicates the probe-blind-spot
      pattern already seen elsewhere in this project.

PRIMARY evidence is the behavioral logit gap; full-state patch is SECONDARY
(wiring check), flagged unreliable when the gap is near zero.

Saves analysis_outputs/final_results/phase3_binaryaddition_condition2.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_binaryaddition_condition2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_binaryaddition_counterfactuals import carry_state_pairs, audit_carry_state_pairs
from phase3_binaryaddition_condition1 import run_rnn_lstm, run_mamba, run_transformer, summarize

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
CARRY_POSITIONS = [1, 2, 3, 4]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    pairs_by_k = {}
    audits_by_k = {}
    for k in CARRY_POSITIONS:
        pairs = carry_state_pairs(k, N_PAIRS, rng)
        audit = audit_carry_state_pairs(pairs, k)
        pairs_by_k[k] = pairs
        audits_by_k[k] = audit
        print(f"=== carry_state_pos{k}_pairs audit ===")
        print(json.dumps(audit, indent=2))
        assert audit["target_property_isolated"], f"pos{k} pairs failed to isolate the target property"

    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    cells = []
    for arch, seed, role in CELLS:
        print(f"\n--- {arch} seed{seed} ({role}) ---", flush=True)
        for k in CARRY_POSITIONS:
            pair_type = f"carry_state_pos{k}"
            pairs = pairs_by_k[k]
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

    def get(arch, seed, k):
        return next(c for c in cells if c["arch"] == arch and c["seed"] == seed
                    and c["pair_type"] == f"carry_state_pos{k}")

    # ------------------------------------------------------------------
    # (1) Mamba seed4 anomaly follow-up: generic reactivity vs partial computation
    # ------------------------------------------------------------------
    m4_gaps = {k: get("mamba", 4, k)["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
               for k in CARRY_POSITIONS}
    m5_gaps = {k: get("mamba", 5, k)["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
               for k in CARRY_POSITIONS}
    m4_all_large = all(g >= DEGENERATE_GAP_THRESHOLD for g in m4_gaps.values())
    m4_none_large = not any(g >= DEGENERATE_GAP_THRESHOLD for g in m4_gaps.values())
    m4_spread = float(np.std(list(m4_gaps.values())))
    m4_mean = float(np.mean(list(m4_gaps.values())))
    mamba_anomaly_check = {
        "mamba_seed4_gaps_by_position": m4_gaps,
        "mamba_seed5_gaps_by_position": m5_gaps,
        "mamba_seed4_mean_gap": m4_mean, "mamba_seed4_gap_std_across_positions": m4_spread,
        "verdict": (
            "GENERIC REACTIVITY -- mamba seed4 shows large, roughly uniform causal effects at "
            "EVERY carry position, consistent with the Condition 1 pattern: this cell appears "
            "broadly hypersensitive to activation perturbation at the readout site, not "
            "selectively encoding specific arithmetic content."
            if m4_all_large else
            "NULL ON ARITHMETIC FEATURES -- despite Condition 1's large structural-feature effects, "
            "mamba seed4 shows no above-noise causal effect on ANY carry position. The Condition 1 "
            "anomaly does NOT generalize to arithmetic content; it may be specific to structural/"
            "positional perturbations."
            if m4_none_large else
            "FEATURE-SPECIFIC (PARTIAL ARITHMETIC COMPUTATION) -- mamba seed4 shows large causal "
            "effects at SOME carry positions but not others, inconsistent with generic reactivity. "
            "This supports a genuine (if incomplete) causal role for specific carry positions, "
            "not indiscriminate sensitivity to any perturbation."
        ),
    }
    print("\n=== Mamba seed4 anomaly follow-up: reactivity vs partial computation ===")
    print(json.dumps(mamba_anomaly_check, indent=2, default=str))

    # ------------------------------------------------------------------
    # (2) probe-blind-spot replication check: cross-reference Stage 3 probe
    # selectivity against this condition's causal gaps, per cell x position
    # ------------------------------------------------------------------
    stage3 = json.loads((RESULTS / "phase2_binaryaddition_arithmetic_probes.json").read_text())
    stage3_interp = stage3["per_cell_interpretation"]

    blind_spot_cases = []
    all_cross = []
    for arch, seed, role in CELLS:
        key = f"{arch}_seed{seed}"
        for k in CARRY_POSITIONS:
            probe_sel = stage3_interp[key][f"carry_state_pos{k}_best_selectivity"]
            probe_has_signal = stage3_interp[key][f"carry_state_pos{k}_has_signal"]
            causal_gap = get(arch, seed, k)["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            causal_effective = get(arch, seed, k)["causally_effective"]
            row = {
                "arch": arch, "seed": seed, "role": role, "k": k,
                "stage3_probe_best_selectivity": probe_sel, "stage3_probe_has_signal": probe_has_signal,
                "condition2_causal_gap": causal_gap, "condition2_causally_effective": causal_effective,
            }
            all_cross.append(row)
            if causal_effective and not probe_has_signal:
                blind_spot_cases.append(row)

    print("\n=== probe-blind-spot replication check (Stage 3 probe selectivity vs Condition 2 causal gap) ===")
    for r in all_cross:
        flag = "  <-- PROBE-BLIND-SPOT" if r in blind_spot_cases else ""
        print(f"  {r['arch']}_seed{r['seed']} pos{r['k']}: probe_sel={r['stage3_probe_best_selectivity']:.3f} "
              f"(signal={r['stage3_probe_has_signal']})  causal_gap={r['condition2_causal_gap']:.3f} "
              f"(causal={r['condition2_causally_effective']}){flag}")

    blind_spot_summary = {
        "n_blind_spot_cases": len(blind_spot_cases),
        "n_total_cell_position_combinations": len(all_cross),
        "blind_spot_cases": blind_spot_cases,
        "interpretation": (
            f"{len(blind_spot_cases)}/{len(all_cross)} (cell, carry-position) combinations show a "
            f"causally-effective behavioral gap despite Stage 3 finding NO probe signal (selectivity "
            f"< 0.05) for that exact target -- " +
            ("replicating the probe-blind-spot pattern seen repeatedly elsewhere in this project: "
             "large causal effects can coexist with near-zero linear-probe selectivity, meaning "
             "Stage 3's mostly-null probing results should NOT be read as 'no arithmetic "
             "representation exists' for these cells/positions."
             if blind_spot_cases else
             "no blind-spot cases found here -- causal effectiveness and probe signal are "
             "consistent with each other for every cell/position tested in this condition.")
        ),
    }
    print("\n=== blind-spot summary ===")
    print(json.dumps(blind_spot_summary, indent=2, default=str))

    out = {
        "condition": "condition_2_arithmetic_features_carry_state",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (canonical readout site per architecture) is SECONDARY, a wiring "
                 "check only, flagged UNRELIABLE when the behavioral gap is near zero.",
        "description": "Tests whether the TRUE carry-in bit at positions 1-4 (computed from u_x/u_y "
                       "only, independent of the claimed u_z) causally drives the accept/reject "
                       "decision, all 8 cells. Both clean and corrupt in every pair are genuine "
                       "correct sums (target_computation=True for both), isolating carry-state "
                       "sensitivity from overall-correctness confounds. Prediction: Transformer "
                       "(both seeds) and mamba primary may show effects at pos3/pos4 (Stage 3's "
                       "weak signal); RNN/LSTM near-zero.",
        "n_pairs_per_position": N_PAIRS,
        "degenerate_gap_threshold": DEGENERATE_GAP_THRESHOLD,
        "carry_positions_tested": CARRY_POSITIONS,
        "pair_construction_audits": audits_by_k,
        "sample_pairs": {f"carry_state_pos{k}": pairs_by_k[k][:2] for k in CARRY_POSITIONS},
        "cells": cells,
        "mamba_seed4_anomaly_followup": mamba_anomaly_check,
        "probe_blind_spot_replication_check": {
            "all_cell_position_rows": all_cross,
            "summary": blind_spot_summary,
        },
    }
    out_path = RESULTS / "phase3_binaryaddition_condition2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
