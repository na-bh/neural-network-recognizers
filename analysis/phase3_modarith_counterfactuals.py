"""Phase 3 counterfactual construction for modular-arithmetic-simple
(bounded, patching-only pilot). Task-specific throughout -- this task's
carrier is a single accumulated mod-5 register updated by non-commuting
operators (+,-,*), structurally distinct from every other task piloted
(marker-family/durable-content, binary-addition/carry-chain, Dyck-2-3/
bounded-stack).

Condition 1 (structural features -- operator positions / digit counts):
clean = genuine valid expression. corrupted = SAME expression with an EXTRA
digit token inserted at a chosen (op,digit)-pair boundary, creating an
illegal digit-digit adjacency -- breaks alternation_pattern_valid,
digit_operator_count_relation, AND sequence_length_parity_odd
simultaneously (a natural, tightly-coupled cluster of the six structural
shortcut features, exactly analogous to how Dyck-2-3's bracket-count-
parity condition necessarily also broke per-type balance). Swept across
insertion position early/mid/late.

Condition 2 (cumulative mod-5 sum feature): rejection-sampled pairs, BOTH
members genuine valid expressions (correct final claimed digit, matching
their own true value) of the SAME length (same number of (op,digit)
pairs), differing specifically in the TRUE running mod-5 value at a chosen
pair-index k -- isolates whether the model represents/uses the
INTERMEDIATE accumulated value, independent of final correctness (both are
correct overall). Swept across which pair-index k (early/mid/late through
the chain).

Condition 3 (target-computation null test): per this stage's approval,
reuses the task audit's load-bearing construction exactly -- permute two
(op,digit) pairs, preserving digit multiset, operator multiset, length,
and the claimed final answer, changing only the TRUE value (since +,-,*
do not commute when interleaved). Swept across WHERE in the sequence the
permuted pair lands (early/mid/late).

NO PATCHING IS RUN HERE -- pair construction + audit only.

PYTHONPATH=src:analysis python analysis/phase3_modarith_counterfactuals.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_modarith_task_audit import (
    parse_and_evaluate, is_valid_example, true_cumulative_value_at_pair,
    SHORTCUT_FEATURES, DIGITS, OPS,
)

RESULTS = Path("analysis_outputs/final_results")
DIGIT_LIST = list(DIGITS)
OP_LIST = list(OPS)


def random_valid_expression(rng, n_pairs):
    """Random digit (op digit)^n_pairs = digit, with the trailing digit set
    to the TRUE computed value (always valid by construction)."""
    seq = [str(int(rng.integers(0, 5)))]
    value = int(seq[0])
    for _ in range(n_pairs):
        op = OP_LIST[int(rng.integers(0, 3))]
        x = int(rng.integers(0, 5))
        seq += [op, str(x)]
        if op == "+":
            value = (value + x) % 5
        elif op == "-":
            value = (value - x) % 5
        elif op == "*":
            value = (value * x) % 5
    seq += ["=", str(value)]
    return seq


def audit_features_match(clean, corrupted):
    return {name: (fn(clean) == fn(corrupted)) for name, fn in SHORTCUT_FEATURES.items()}


# ---------------------------------------------------------------------------
# condition 1: structural features (digit-digit adjacency break)
# ---------------------------------------------------------------------------

def structural_break_pairs(n_pairs_per_expr, n_pairs, rng, position_frac):
    """Insert an extra digit at a chosen (op,digit)-pair boundary --
    creates an illegal digit-digit adjacency."""
    pairs = []
    for _ in range(n_pairs):
        clean = random_valid_expression(rng, n_pairs_per_expr)
        # candidate insertion points: right after any digit token that is
        # followed by an operator (i.e., insertable boundaries within the
        # digit,(op,digit)* portion, excluding the final '=', digit)
        n_body = 1 + 2 * n_pairs_per_expr  # length of "digit (op digit)*" part
        candidates = list(range(1, n_body, 2))  # positions right after each digit in the body
        insert_at = min(candidates, key=lambda i: abs(i / (len(clean) - 1) - position_frac))
        extra_digit = DIGIT_LIST[int(rng.integers(0, 5))]
        corrupted = clean[:insert_at] + [extra_digit] + clean[insert_at:]
        pairs.append({"clean": clean, "corrupted": corrupted, "insert_at_index": insert_at,
                      "insert_position_relative": insert_at / (len(clean) - 1)})
    return pairs


def audit_structural_break_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["length_delta_is_plus_1"] = all(len(p["corrupted"]) - len(p["clean"]) == 1 for p in pairs)
    audit["clean_all_valid"] = all(is_valid_example(p["clean"]) for p in pairs)
    audit["corrupted_all_invalid"] = all(not is_valid_example(p["corrupted"]) for p in pairs)
    audit["corrupted_all_break_alternation"] = all(
        not SHORTCUT_FEATURES["alternation_pattern_valid"](p["corrupted"]) for p in pairs)
    audit["target_property_isolated"] = (
        audit["length_delta_is_plus_1"] and audit["clean_all_valid"]
        and audit["corrupted_all_invalid"] and audit["corrupted_all_break_alternation"]
    )
    return audit


# ---------------------------------------------------------------------------
# condition 2: cumulative mod-5 sum feature (correctness held fixed both sides)
# ---------------------------------------------------------------------------

def cumulative_sum_pairs(n_pairs_per_expr, k, n_pairs, rng, max_tries_mult=400):
    """Both clean and corrupt are genuine valid expressions of the SAME
    length (same n_pairs_per_expr), differing in the TRUE running value at
    pair-index k. Rejection-sampled by value bucket."""
    zero_group, other_group = [], []
    attempts = 0
    target_n = n_pairs * 3
    while (len(zero_group) < target_n or len(other_group) < target_n) and attempts < n_pairs * max_tries_mult:
        attempts += 1
        expr = random_valid_expression(rng, n_pairs_per_expr)
        val_at_k = true_cumulative_value_at_pair(expr, k)
        if val_at_k is None:
            continue
        (zero_group if val_at_k == 0 else other_group).append((expr, val_at_k))
    if len(zero_group) < n_pairs or len(other_group) < n_pairs:
        raise RuntimeError(f"pos{k}: only got {len(zero_group)} val=0 / {len(other_group)} val!=0 after {attempts} attempts")
    pairs = []
    for i in range(n_pairs):
        clean, clean_val = zero_group[i]
        corrupt, corrupt_val = other_group[i]
        pairs.append({"clean": clean, "corrupted": corrupt, "pair_index_k": k,
                      "clean_cumulative_value_at_k": clean_val, "corrupted_cumulative_value_at_k": corrupt_val})
    return pairs


def audit_cumulative_sum_pairs(pairs, k):
    audit = {"n_pairs": len(pairs), "k": k}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupted"]) for p in pairs)
    audit["clean_all_valid"] = all(is_valid_example(p["clean"]) for p in pairs)
    audit["corrupted_all_valid"] = all(is_valid_example(p["corrupted"]) for p in pairs)  # BOTH correct overall
    audit["cumulative_values_differ"] = all(
        p["clean_cumulative_value_at_k"] != p["corrupted_cumulative_value_at_k"] for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["clean_all_valid"] and audit["corrupted_all_valid"]
        and audit["cumulative_values_differ"]
    )
    return audit


# ---------------------------------------------------------------------------
# condition 3: target computation, permuted (op,digit) pairs
# ---------------------------------------------------------------------------

def permuted_pair_counterexample(n_pairs_per_expr, swap_position_frac, rng, max_tries=400):
    """Genuine valid expression; swap two (op,digit) pairs (positions
    chosen to land near swap_position_frac through the sequence) -- keeps
    digit multiset, operator multiset, length, and claimed final digit
    identical; changes the TRUE value (rejection-sampled to guarantee a
    genuine change, since +/-/* commute in SOME cases)."""
    for _ in range(max_tries):
        clean = random_valid_expression(rng, n_pairs_per_expr)
        if n_pairs_per_expr < 2:
            continue
        # (op,digit) pair start indices: 1, 3, 5, ..., within the body
        pair_starts = [1 + 2 * i for i in range(n_pairs_per_expr)]
        target_frac = swap_position_frac
        a_idx = min(range(len(pair_starts) - 1),
                    key=lambda i: abs(pair_starts[i] / (len(clean) - 1) - target_frac))
        b_idx = a_idx + 1
        a, b = pair_starts[a_idx], pair_starts[b_idx]
        corrupted = clean[:]
        corrupted[a], corrupted[a + 1] = clean[b], clean[b + 1]
        corrupted[b], corrupted[b + 1] = clean[a], clean[a + 1]
        clean_ok, clean_val, clean_claim = parse_and_evaluate(clean)
        corrupt_ok, corrupt_val, _ = parse_and_evaluate(corrupted)
        if corrupt_val == clean_claim:
            continue  # coincidentally still correct (commuted away) -- retry
        if not all(fn(clean) == fn(corrupted) for fn in SHORTCUT_FEATURES.values()):
            continue
        return {"clean": clean, "corrupted": corrupted, "swapped_pair_indices": [a_idx, b_idx],
                "swap_position_relative": a / (len(clean) - 1),
                "clean_true_value": clean_val, "corrupted_true_value": corrupt_val, "claimed_value": clean_claim}
    raise RuntimeError("could not construct a permuted-pair counterexample")


def target_computation_permuted_pairs(n_pairs_per_expr, n_pairs, rng, position_frac):
    return [permuted_pair_counterexample(n_pairs_per_expr, position_frac, rng) for _ in range(n_pairs)]


def audit_permuted_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupted"]) for p in pairs)
    audit["clean_all_valid"] = all(is_valid_example(p["clean"]) for p in pairs)
    audit["corrupted_all_invalid"] = all(not is_valid_example(p["corrupted"]) for p in pairs)
    feat_matches = [audit_features_match(p["clean"], p["corrupted"]) for p in pairs]
    audit["all_six_shortcut_features_preserved"] = all(all(f.values()) for f in feat_matches)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["clean_all_valid"] and audit["corrupted_all_invalid"]
        and audit["all_six_shortcut_features_preserved"]
    )
    return audit


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    N_PAIRS_PER_EXPR = 8

    print("=== condition 1: structural_break_pairs (position sweep) ===")
    for label, frac in [("early", 0.2), ("mid", 0.5), ("late", 0.8)]:
        pairs = structural_break_pairs(N_PAIRS_PER_EXPR, 10, rng, frac)
        audit = audit_structural_break_pairs(pairs)
        print(f"  [{label}] {json.dumps(audit)}")

    print("=== condition 2: cumulative_sum_pairs (position sweep over pair-index k) ===")
    for label, k in [("early", 1), ("mid", 4), ("late", 7)]:
        pairs = cumulative_sum_pairs(N_PAIRS_PER_EXPR, k, 10, rng)
        audit = audit_cumulative_sum_pairs(pairs, k)
        print(f"  [{label}, k={k}] {json.dumps(audit)}")

    print("=== condition 3: target_computation_permuted_pairs (position sweep) ===")
    for label, frac in [("early", 0.2), ("mid", 0.5), ("late", 0.8)]:
        pairs = target_computation_permuted_pairs(N_PAIRS_PER_EXPR, 10, rng, frac)
        audit = audit_permuted_pairs(pairs)
        print(f"  [{label}] {json.dumps(audit)}")
