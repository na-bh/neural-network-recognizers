"""Assembles phase3_dyck_counterfactual_design.json: builds N=25 pairs per
condition x position/depth (matching the pilot's standard N=25/cell/
condition convention) using analysis/phase3_dyck_counterfactuals.py's
constructions, runs every audit, and saves the full design document (no
patching run here -- that is Conditions 1-3's own job, once approved).

PYTHONPATH=src:analysis python analysis/phase3_dyck_build_counterfactual_design.py
"""

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "analysis")
from phase3_dyck_counterfactuals import (
    parity_pairs, audit_parity_pairs,
    per_type_pairs, audit_per_type_pairs,
    target_computation_type_mismatch_pairs, audit_type_mismatch_pairs,
    target_computation_depth_exceeded_pairs, audit_depth_exceeded_pairs,
)

RESULTS = Path("analysis_outputs/final_results")
N_PAIRS = 25
POSITIONS = {"early": 0.2, "mid": 0.5, "late": 0.8}
DEPTHS = [1, 2, 3]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260715)

    condition1 = {}
    for label, frac in POSITIONS.items():
        pairs = parity_pairs(N_PAIRS, rng, frac)
        audit = audit_parity_pairs(pairs)
        condition1[label] = {"position_frac": frac, "pairs": pairs, "audit": audit}
        print(f"condition1 [{label}]: {json.dumps(audit)}", flush=True)
        assert audit["target_property_isolated"]

    condition2 = {}
    for label, frac in POSITIONS.items():
        pairs = per_type_pairs(N_PAIRS, rng, frac)
        audit = audit_per_type_pairs(pairs)
        condition2[label] = {"position_frac": frac, "pairs": pairs, "audit": audit}
        print(f"condition2 [{label}]: {json.dumps(audit)}", flush=True)
        assert audit["target_property_isolated"]

    condition3a = {}
    for depth in DEPTHS:
        pairs = target_computation_type_mismatch_pairs(N_PAIRS, rng, depth)
        audit = audit_type_mismatch_pairs(pairs, depth)
        condition3a[f"depth{depth}"] = {"depth_level": depth, "pairs": pairs, "audit": audit}
        print(f"condition3a [depth={depth}]: {json.dumps(audit)}", flush=True)
        assert audit["target_property_isolated"]

    condition3b = {}
    for label, frac in POSITIONS.items():
        pairs = target_computation_depth_exceeded_pairs(N_PAIRS, rng, frac)
        audit = audit_depth_exceeded_pairs(pairs)
        condition3b[label] = {"position_frac": frac, "pairs": pairs, "audit": audit}
        print(f"condition3b [{label}]: {json.dumps(audit)}", flush=True)
        assert audit["target_property_isolated"]

    out = {
        "task": "dyck-2-3",
        "description": "Counterfactual pair design for Dyck-2-3's bounded Phase 3 patching pilot, "
                       "built from phase3_dyck_task_audit.json's findings. NO PATCHING RUN HERE.",
        "n_pairs_per_position_or_depth": N_PAIRS,
        "condition_1_bracket_count_parity": {
            "construction": "Flip ONE interior closer into an opener of the same type -- preserves "
                            "length, length_parity, and first/last tokens exactly; necessarily also "
                            "breaks that bracket's own per-type balance (mathematically forced, "
                            "documented not hidden). Position swept early/mid/late.",
            "prediction": "Causally load-bearing across most cells IF the shortcut is used -- "
                          "bracket_count_parity has only a 9.5% blind spot on real negatives (Stage-"
                          "1 audit), a strong shortcut signal.",
            "by_position": condition1,
        },
        "condition_2_per_type_bracket_counts": {
            "construction": "Relabel ONE interior closer's TYPE (e.g. )0->)1) -- preserves length, "
                            "TOTAL open/close counts (bracket_count_parity UNCHANGED, isolating this "
                            "from condition 1), and first/last tokens; breaks per-type balance for "
                            "the two affected types. Position swept early/mid/late.",
            "prediction": "Possibly causally used, especially if models detect type-mismatch "
                          "shortcuts -- per_type_bracket_counts has only a 1.9% blind spot (the "
                          "strongest of the four features), so if models lean on ANY structural "
                          "shortcut this is the most likely candidate.",
            "by_position": condition2,
        },
        "condition_3a_target_computation_type_mismatch": {
            "construction": "Swap the TYPE LABELS of two nested closers at a chosen depth level -- "
                            "preserves ALL FOUR listed shortcut features exactly (same token "
                            "multiset, just two closer positions exchanged), breaks ONLY LIFO "
                            "stack-order matching. Mirrors the task audit's own load-bearing "
                            "counterexample, generalized across depth levels 1/2/3.",
            "prediction": "Uniform absence across cells under the carrier account (matching "
                          "marker-family durable-carrier tasks) -- any cell showing a causal effect "
                          "here is genuine stack-verification, a specific new finding.",
            "by_depth": condition3a,
        },
        "condition_3b_target_computation_depth_exceeded": {
            "construction": "Insert a matched (open,close) pair of a random type at a point where "
                            "nesting depth is already 3 -- momentarily reaches depth 4 (invalid), "
                            "even though the inserted pair is itself properly matched and the string "
                            "ends balanced. Preserves all four shortcut features (adds 1 to that "
                            "type's open AND close count together; length changes by +2, an EVEN "
                            "delta, so length_parity is unchanged; insertion is interior, so first/"
                            "last unchanged). Architecturally DISTINCT from 3a: a bounded-COUNTER "
                            "violation, not a LIFO-order violation -- kept as a separate sub-"
                            "condition per this stage's approval, since depth-checking requires "
                            "tracking a counter against a threshold, a different computational "
                            "primitive from stack-order matching. Position (of the insertion) swept "
                            "early/mid/late.",
            "prediction": "Uniform absence across cells under the carrier account, same as 3a -- "
                          "but a positive finding here would implicate DEPTH-COUNTING specifically, "
                          "distinguishable from LIFO-verification (3a) by which sub-condition shows "
                          "the effect.",
            "by_position": condition3b,
        },
    }
    out_path = RESULTS / "phase3_dyck_counterfactual_design.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
