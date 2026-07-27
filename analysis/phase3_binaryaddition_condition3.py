"""Binary-addition Phase 3, condition 3 of 3: target_computation causal
patching with position-swept single-bit-flip counterfactuals in u_z. This
is the load-bearing test for the carrier account's cross-family prediction
on this task -- durable-carrier tasks (marked-reversal, and by hypothesis
binary-addition) should show uniform ABSENCE of causal target_computation
verification, unlike locally-verifiable tasks (bucket-sort) where some
cells showed comprehensive or position-specific causal use.

Pair construction (analysis/phase3_binaryaddition_counterfactuals.py's
target_computation_bitflip_pairs, per Phase 1 P2's already-specified
design): clean = genuine correct sum at FIXED lengths (LEN_X=LEN_Y=8,
LEN_Z=10, matching conditions 1-2). corrupt = SAME u_x, u_y, SAME u_z
EXCEPT one bit flipped at position k -- guarantees an incorrect sum,
preserves len_x/y/z and structural validity exactly. Does NOT preserve
ones_count_z (documented in the Phase 1 design, expected and re-audited
here, not assumed).

Three positions swept within u_z (LEN_Z=10): low=1, mid=5, high=8 --
mirroring bucket-sort's position-sweep discipline of testing early/mid/
late locations rather than trusting one fixed position.

Three readings distinguished per this condition's approval:
  (1) UNIFORM ABSENCE across cells and positions -> carrier account
      CONFIRMED for this task (matches marked-reversal).
  (2) COMPREHENSIVE causal use in some cells (all 3 positions) -> would be
      SURPRISING under the carrier account -- flagged for investigation
      (possibly local carry-state checks approximating full verification,
      testable against Condition 2's carry-state results for that cell).
  (3) POSITION-SPECIFIC causal effects (some but not all positions) ->
      partial carry-verification, not comprehensive verification (matches
      bucket-sort's RNN seed4 / LSTM seed3 pattern).

RNN seed9 specific interpretability test (per this condition's approval):
Condition 2 found RNN seed9 causally uses carry_state at ALL FOUR positions
despite near-zero Stage 3 probe signal (a blind-spot case). If seed9 ALSO
shows comprehensive (all-position) causal effects on target_computation
here, that supports interpretation (a) -- RNN seed9 built a genuine
(if partial) carrier, since carry-state use compounds into full-sum
verification. If seed9's target_computation effects are position-specific
or absent, that supports interpretation (b) -- Condition 2's carry-state
effects reflect CORRELATE use (the carry bit correlates with something
else driving the decision) rather than genuine carrier construction that
composes into sum verification.

Standard discipline: behavioral gap PRIMARY, full-state patch SECONDARY
(wiring check, flagged unreliable when the gap is near zero).

Saves analysis_outputs/final_results/phase3_binaryaddition_condition3.json.
STOP after this condition.

PYTHONPATH=src:analysis python analysis/phase3_binaryaddition_condition3.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_binaryaddition_counterfactuals import (
    target_computation_bitflip_pairs, audit_target_computation_bitflip_pairs,
)
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
FLIP_POSITIONS = {"low": 1, "mid": 5, "high": 8}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    position_pairs, position_audits = {}, {}
    for label, k in FLIP_POSITIONS.items():
        rng = np.random.default_rng(hash(label) % (2**31))
        pairs = target_computation_bitflip_pairs(k, N_PAIRS, rng)
        audit = audit_target_computation_bitflip_pairs(pairs, k)
        position_pairs[label] = pairs
        position_audits[label] = audit
        print(f"=== flip_k={k} ({label}) audit ===")
        print(json.dumps(audit, indent=2))
        assert audit["target_property_isolated"], f"{label} (flip_k={k}) failed to isolate the target property"

    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    results_by_position = {}
    for label, k in FLIP_POSITIONS.items():
        pairs = position_pairs[label]
        cells = []
        for arch, seed, role in CELLS:
            print(f"\n--- [{label}, flip_k={k}] {arch} seed{seed} ({role}) ---", flush=True)
            if arch in ("rnn", "lstm"):
                results, ckpt = run_rnn_lstm(arch, seed, pairs, tok2idx)
            elif arch == "mamba":
                results, ckpt = run_mamba(seed, pairs, tok2idx)
            else:
                results, ckpt = run_transformer(seed, pairs, tok2idx)
            cell = summarize(results, f"target_computation_flip_{label}", arch, seed, role, ckpt)
            cell["flip_k"] = k
            cell["position_label"] = label
            cells.append(cell)
            gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
            print(f"  PRIMARY behavioral gap={gap:.4f} causal={cell['causally_effective']}  |  "
                  f"wiring rf_mean={rf:.4f}", flush=True)
        results_by_position[label] = cells

    # ------------------------------------------------------------------
    # per-cell cross-position summary: comprehensive / null / position-specific
    # ------------------------------------------------------------------
    per_cell_summary = {}
    for arch, seed, role in CELLS:
        key = f"{arch}_seed{seed}"
        gaps_by_position, causal_by_position = {}, {}
        for label in FLIP_POSITIONS:
            c = next(x for x in results_by_position[label] if x["arch"] == arch and x["seed"] == seed)
            gaps_by_position[label] = c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            causal_by_position[label] = c["causally_effective"]
        all_causal = all(causal_by_position.values())
        none_causal = not any(causal_by_position.values())
        verdict = (
            "COMPREHENSIVE CAUSAL USE (all 3 positions) -- would be SURPRISING under the carrier "
            "account; flagged for investigation."
            if all_causal else
            "UNIFORM ABSENCE (null at all 3 positions) -- consistent with the carrier account's "
            "prediction that this task's target_computation requires durable carriers this "
            "architecture/seed does not have."
            if none_causal else
            "POSITION-SPECIFIC (causal at some, not all, positions) -- partial carry-verification, "
            "not comprehensive sum verification."
        )
        per_cell_summary[key] = {
            "arch": arch, "seed": seed, "role": role,
            "gap_by_position": gaps_by_position, "causal_by_position": causal_by_position,
            "verdict": verdict,
        }
        print(f"{key}: " + " ".join(f"{l}={gaps_by_position[l]:.3f}" for l in FLIP_POSITIONS) + f"  [{verdict.split(' -- ')[0]}]")

    n_comprehensive = sum(1 for v in per_cell_summary.values() if v["verdict"].startswith("COMPREHENSIVE"))
    n_null = sum(1 for v in per_cell_summary.values() if v["verdict"].startswith("UNIFORM ABSENCE"))
    n_position_specific = len(CELLS) - n_comprehensive - n_null

    overall_conclusion = {
        "n_cells_comprehensive": n_comprehensive, "n_cells_null": n_null,
        "n_cells_position_specific": n_position_specific,
        "carrier_account_reading": (
            f"{n_null}/{len(CELLS)} cells show UNIFORM ABSENCE across all 3 flip positions, "
            f"{n_comprehensive}/{len(CELLS)} show COMPREHENSIVE causal use, "
            f"{n_position_specific}/{len(CELLS)} show POSITION-SPECIFIC effects. " +
            ("Reading (1): uniform absence -- the carrier account is CONFIRMED for this task, "
             "matching marked-reversal: no architecture causally verifies full sum correctness."
             if n_comprehensive == 0 and n_position_specific == 0 else
             f"Readings (2)/(3) apply to {n_comprehensive + n_position_specific}/{len(CELLS)} cells -- "
             f"the carrier account is NOT uniformly confirmed; see per-cell verdicts and the RNN "
             f"seed9 interpretability test below for the specific mechanism(s) involved."),
        ),
    }
    print("\n=== overall conclusion ===")
    print(json.dumps(overall_conclusion, indent=2, default=str))

    # ------------------------------------------------------------------
    # RNN seed9 interpretability test (this condition's specific instruction)
    # ------------------------------------------------------------------
    rnn9 = per_cell_summary["rnn_seed9"]
    rnn9_all_causal = all(rnn9["causal_by_position"].values())
    rnn9_interpretation = {
        "condition2_carry_state_finding": "RNN seed9 was causally effective on carry_state at ALL "
                                          "FOUR positions (Condition 2), despite near-zero Stage 3 "
                                          "probe signal -- a probe-blind-spot case.",
        "condition3_target_computation_gaps": rnn9["gap_by_position"],
        "condition3_causal_by_position": rnn9["causal_by_position"],
        "interpretation": (
            "(a) RNN seed9 built a genuine (if partial) CARRIER -- comprehensive causal effects on "
            "target_computation at ALL 3 flip positions here, compounding with Condition 2's "
            "carry-state effects, supports carry-tracking that composes into actual sum "
            "verification, not a mere correlate."
            if rnn9_all_causal else
            "(b) CORRELATE USE, not carrier construction -- Condition 2's carry-state causal "
            "effects do NOT compose into comprehensive target_computation verification here "
            "(position-specific or absent effects on the full-sum test). The carry-state "
            "sensitivity found in Condition 2 more likely reflects the carry bit correlating with "
            "some OTHER decision-relevant quantity (e.g. a length/parity/local-pattern heuristic) "
            "rather than genuine incremental carrier construction toward full arithmetic "
            "verification."
        ),
    }
    print("\n=== RNN seed9 interpretability test ===")
    print(json.dumps(rnn9_interpretation, indent=2, default=str))

    out = {
        "condition": "condition_3_target_computation_position_sweep",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (canonical readout site per architecture) is SECONDARY, a wiring "
                 "check only, flagged UNRELIABLE when the behavioral gap is near zero.",
        "description": "Load-bearing test for the carrier account's cross-family prediction on "
                       "binary-addition: clean = genuine correct sum at fixed lengths (LEN_X=LEN_Y=8, "
                       "LEN_Z=10); corrupt = single bit of u_z flipped at a swept position "
                       "(low=1, mid=5, high=8), guaranteeing an incorrect sum while preserving "
                       "len_x/y/z and structural validity exactly (NOT preserving ones_count_z, "
                       "documented and re-audited). All 8 cells x 3 positions.",
        "n_pairs_per_position": N_PAIRS,
        "degenerate_gap_threshold": DEGENERATE_GAP_THRESHOLD,
        "flip_positions_tested": FLIP_POSITIONS,
        "position_construction_audits": position_audits,
        "sample_pairs": {label: position_pairs[label][:2] for label in FLIP_POSITIONS},
        "results_by_position": results_by_position,
        "per_cell_summary": per_cell_summary,
        "overall_conclusion": overall_conclusion,
        "rnn_seed9_interpretability_test": rnn9_interpretation,
    }
    out_path = RESULTS / "phase3_binaryaddition_condition3.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
