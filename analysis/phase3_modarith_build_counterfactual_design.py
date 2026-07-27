"""Assembles phase3_modarith_counterfactual_design.json: builds N=25 pairs
per condition x position (matching the pilot's standard N=25/cell/
condition convention) using analysis/phase3_modarith_counterfactuals.py's
constructions, runs every audit, and saves the full design document. No
patching run here -- that is Conditions 1-3's own job, once approved.

PYTHONPATH=src:analysis python analysis/phase3_modarith_build_counterfactual_design.py
"""

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "analysis")
from phase3_modarith_counterfactuals import (
    structural_break_pairs, audit_structural_break_pairs,
    cumulative_sum_pairs, audit_cumulative_sum_pairs,
    target_computation_permuted_pairs, audit_permuted_pairs,
)

RESULTS = Path("analysis_outputs/final_results")
N_PAIRS = 25
N_PAIRS_PER_EXPR = 8
POSITIONS = {"early": 0.2, "mid": 0.5, "late": 0.8}
COND2_POSITIONS = {"early": 1, "mid": 4, "late": 7}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260716)

    condition1 = {}
    for label, frac in POSITIONS.items():
        pairs = structural_break_pairs(N_PAIRS_PER_EXPR, N_PAIRS, rng, frac)
        audit = audit_structural_break_pairs(pairs)
        condition1[label] = {"position_frac": frac, "pairs": pairs, "audit": audit}
        print(f"condition1 [{label}]: {json.dumps(audit)}", flush=True)
        assert audit["target_property_isolated"]

    condition2 = {}
    for label, k in COND2_POSITIONS.items():
        pairs = cumulative_sum_pairs(N_PAIRS_PER_EXPR, k, N_PAIRS, rng)
        audit = audit_cumulative_sum_pairs(pairs, k)
        condition2[label] = {"pair_index_k": k, "pairs": pairs, "audit": audit}
        print(f"condition2 [{label}, k={k}]: {json.dumps(audit)}", flush=True)
        assert audit["target_property_isolated"]

    condition3 = {}
    for label, frac in POSITIONS.items():
        pairs = target_computation_permuted_pairs(N_PAIRS_PER_EXPR, N_PAIRS, rng, frac)
        audit = audit_permuted_pairs(pairs)
        condition3[label] = {"position_frac": frac, "pairs": pairs, "audit": audit}
        print(f"condition3 [{label}]: {json.dumps(audit)}", flush=True)
        assert audit["target_property_isolated"]

    out = {
        "task": "modular-arithmetic-simple",
        "description": "Counterfactual pair design for modular-arithmetic-simple's bounded Phase 3 "
                       "patching pilot, built from phase3_modarith_task_audit.json's findings. NO "
                       "PATCHING RUN HERE.",
        "n_pairs_per_position": N_PAIRS, "n_pairs_per_expression": N_PAIRS_PER_EXPR,
        "condition_1_structural_features": {
            "construction": "Insert an EXTRA digit token at a chosen (op,digit)-pair boundary -- "
                            "creates an illegal digit-digit adjacency, breaking alternation_pattern_"
                            "valid, digit_operator_count_relation, and sequence_length_parity_odd "
                            "simultaneously (a tightly-coupled cluster, analogous to how Dyck-2-3's "
                            "parity condition necessarily also broke per-type balance). Position "
                            "(of the insertion) swept early/mid/late.",
            "prediction": "Causally load-bearing if the shortcut is used -- these are the most "
                          "easily shortcut-exploitable structural features per the task audit "
                          "(alternation_pattern_valid alone has only a 0.6% blind spot).",
            "by_position": condition1,
        },
        "condition_2_cumulative_sum_feature": {
            "construction": "Rejection-sampled pairs, BOTH members genuine valid expressions "
                            "(correct final claimed digit) of the SAME length, differing in the "
                            "TRUE running mod-5 value at a chosen pair-index k -- isolates whether "
                            "the model represents/uses the INTERMEDIATE accumulated value "
                            "independent of overall correctness. Swept across pair-index k "
                            "(early=1, mid=4, late=7 of 8 pairs).",
            "prediction": "Under the refined carrier account: mod-5 sum requires an error-"
                          "intolerant aggregate carrier -- either models build it (majority-like, "
                          "causally load-bearing) or show absence (parity-like, uniform null). Not "
                          "content-preserving, so should NOT show marker-family-style uniform "
                          "absence by default.",
            "by_position": condition2,
        },
        "condition_3_target_computation_permuted_pairs": {
            "construction": "Reuses the task audit's load-bearing counterexample construction "
                            "exactly: permute two (op,digit) pairs -- preserves digit multiset, "
                            "operator multiset, length, and claimed final answer; changes ONLY the "
                            "true value (since +,-,* do not commute when interleaved), rejection-"
                            "sampled to guarantee a genuine change. Position (of the permuted pair) "
                            "swept early/mid/late.",
            "prediction": "Target-computation null test: uniform absence would match the marker-"
                          "family durable-carrier pattern; any cell showing a causal effect here is "
                          "genuine order-sensitive verification -- a specific new finding.",
            "by_position": condition3,
        },
    }
    out_path = RESULTS / "phase3_modarith_counterfactual_design.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
