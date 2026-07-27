"""Modular-arithmetic-simple shortcut-exploitation pilot, Phase 3 (bounded,
patching-only): task audit. Structural analysis of the FLaRe generator
(src/recognizers/hand_picked_languages/modular_arithmetic_simple.py) +
empirical verification on the actual test-set data.

Grammar (confirmed via a from-scratch left-to-right evaluator, 100% exact
match against the real training-set labels, not assumed from reading the
FSA alone): digit (op digit)* '=' digit, where digit in {0,1,2,3,4}, op in
{'+','-','*'}, evaluated LEFT TO RIGHT (no operator precedence) mod 5 at
every step; the trailing digit after '=' must equal the running value.
Alphabet: 9 tokens ('*','+','-','0','1','2','3','4','=').

This is a fundamentally different carrier type from every task piloted so
far: not a durable content-multiset (marker family), not a carry-
propagating arithmetic register (binary-addition), not a bounded stack
(Dyck-2-3) -- here the carrier is a SINGLE accumulated mod-5 REGISTER
updated by one of three non-commuting operators at each step, order-
sensitive throughout (reordering two (op,digit) pairs generally changes the
final value, confirmed via the load-bearing counterexample below).

Reproduced the cited "99.4% of negatives structurally scrambled" figure
EXACTLY on the test split (99.40%, 2468/2483 negatives fail to parse the
grammar at all; only 15/2483 = 0.60% are grammatically valid with a wrong
final digit) -- confirms the population composition and which split the
cited figure refers to.

Six candidate structural shortcut features audited (per this experiment's
design), each a TRIVIAL pattern/count check requiring no arithmetic
computation:
  n_equals_is_one            (exactly one '=' token)
  first_token_is_digit
  last_token_is_digit
  sequence_length_parity     (must be ODD: 1 + 2k + 2 tokens for k ops --
                               the parity-necessity direction is opposite
                               Dyck-2-3's, which requires EVEN)
  alternation_pattern_valid  (token TYPES, not values, follow the
                               digit,op,digit,op,...,digit,'=',digit slot
                               pattern)
  digit_operator_count_relation (n_digits == n_operators + 2)
Each is a NECESSARY (not sufficient) condition for a positive. The GENUINE
target-computation carrier (the running mod-5 value at each position) is
NOT reducible to any combination of these -- demonstrated directly via the
load-bearing counterexample (reordering two (op,digit) pairs preserves
every one of the six features exactly, changes the true value).

PYTHONPATH=src:analysis python analysis/phase3_modarith_task_audit.py
"""

import json
import sys
from pathlib import Path

RESULTS = Path("analysis_outputs/final_results")
TASK = "modular-arithmetic-simple"
DIGITS = set("01234")
OPS = set("+-*")


def load_split(split):
    d = Path(f"languages/{TASK}") if split == "train" else Path(f"languages/{TASK}/datasets/{split}")
    toks = (d / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (d / "labels.txt").read_text().splitlines()]
    seqs = [line.split() for line in toks]
    return seqs, labels


def parse_and_evaluate(seq):
    """Returns (grammatically_valid, true_value, claimed_value). Left-to-
    right evaluation mod 5, no operator precedence -- matches the FSA
    exactly (100% agreement verified against real labels)."""
    n = len(seq)
    if n == 0 or seq[0] not in DIGITS:
        return False, None, None
    value = int(seq[0])
    i = 1
    while i < n and seq[i] in OPS:
        op = seq[i]
        i += 1
        if i >= n or seq[i] not in DIGITS:
            return False, None, None
        x = int(seq[i])
        i += 1
        if op == "+":
            value = (value + x) % 5
        elif op == "-":
            value = (value - x) % 5
        elif op == "*":
            value = (value * x) % 5
    if i >= n or seq[i] != "=":
        return False, None, None
    i += 1
    if i >= n or seq[i] not in DIGITS:
        return False, None, None
    claimed = int(seq[i])
    i += 1
    if i != n:
        return False, None, None
    return True, value, claimed


def is_valid_example(seq):
    ok, true_val, claimed = parse_and_evaluate(seq)
    return ok and true_val == claimed


def true_cumulative_value_at_pair(seq, k):
    """The TRUE running mod-5 value after the k-th (op,digit) pair (k=0
    means just the leading digit, before any operator) -- computed from
    the CLAIMED tokens only up to that point, independent of whatever the
    final claimed digit says. Returns None if the prefix doesn't have k
    pairs or isn't grammatically valid up to that point."""
    n = len(seq)
    if n == 0 or seq[0] not in DIGITS:
        return None
    value = int(seq[0])
    i = 1
    pair_idx = 0
    while pair_idx < k:
        if i + 1 >= n or seq[i] not in OPS or seq[i + 1] not in DIGITS:
            return None
        op, x = seq[i], int(seq[i + 1])
        if op == "+":
            value = (value + x) % 5
        elif op == "-":
            value = (value - x) % 5
        elif op == "*":
            value = (value * x) % 5
        i += 2
        pair_idx += 1
    return value


# ---------------------------------------------------------------------------
# six structural shortcut features -- all trivial counting/pattern checks
# ---------------------------------------------------------------------------

def n_equals_is_one(seq):
    return sum(1 for t in seq if t == "=") == 1


def first_token_is_digit(seq):
    return bool(seq) and seq[0] in DIGITS


def last_token_is_digit(seq):
    return bool(seq) and seq[-1] in DIGITS


def sequence_length_parity_odd(seq):
    return len(seq) % 2 == 1


def alternation_pattern_valid(seq):
    """Token TYPE (digit/op/equals) follows digit,(op,digit)*,=,digit --
    checked WITHOUT verifying any arithmetic, pure slot-pattern matching."""
    n = len(seq)
    if n == 0 or seq[0] not in DIGITS:
        return False
    i = 1
    while i < n and seq[i] in OPS:
        if i + 1 >= n or seq[i + 1] not in DIGITS:
            return False
        i += 2
    if i >= n or seq[i] != "=":
        return False
    i += 1
    return i < n and seq[i] in DIGITS and i + 1 == n


def digit_operator_count_relation(seq):
    n_digits = sum(1 for t in seq if t in DIGITS)
    n_ops = sum(1 for t in seq if t in OPS)
    return n_digits == n_ops + 2


SHORTCUT_FEATURES = {
    "n_equals_is_one": n_equals_is_one,
    "first_token_is_digit": first_token_is_digit,
    "last_token_is_digit": last_token_is_digit,
    "sequence_length_parity_odd": sequence_length_parity_odd,
    "alternation_pattern_valid": alternation_pattern_valid,
    "digit_operator_count_relation": digit_operator_count_relation,
}


def audit_shortcut_features(seqs, labels):
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    per_feature = {}
    for name, fn in SHORTCUT_FEATURES.items():
        pos_satisfy = sum(1 for i in pos_idx if fn(seqs[i]))
        neg_satisfy = sum(1 for i in neg_idx if fn(seqs[i]))
        per_feature[name] = {
            "positive_satisfy_fraction": pos_satisfy / len(pos_idx) if pos_idx else float("nan"),
            "negative_satisfy_fraction": neg_satisfy / len(neg_idx) if neg_idx else float("nan"),
        }

    def passes_all(seq):
        return all(fn(seq) for fn in SHORTCUT_FEATURES.values())

    joint_pos = sum(1 for i in pos_idx if passes_all(seqs[i]))
    joint_neg_idx = [i for i in neg_idx if passes_all(seqs[i])]
    joint = {
        "positive_satisfy_all_six_fraction": joint_pos / len(pos_idx) if pos_idx else float("nan"),
        "negative_satisfy_all_six_fraction (shortcut-blind fraction)": len(joint_neg_idx) / len(neg_idx) if neg_idx else float("nan"),
        "n_shortcut_blind_negatives": len(joint_neg_idx), "n_total_negatives": len(neg_idx),
    }
    # cross-check: shortcut-blind negatives should equal the grammatically-valid-but-wrong population
    n_grammatically_valid_neg = sum(1 for i in neg_idx if parse_and_evaluate(seqs[i])[0])
    joint["matches_grammatically_valid_negative_population"] = (len(joint_neg_idx) == n_grammatically_valid_neg)
    joint["n_grammatically_valid_negatives"] = n_grammatically_valid_neg
    return per_feature, joint


def build_counterexample():
    """Reorder two (op,digit) pairs -- preserves digit multiset, operator
    multiset, length, first/last token type, alternation pattern, and
    equals-count exactly; changes the TRUE value because +/-/* do not
    commute when interleaved."""
    clean = ["1", "*", "2", "+", "3", "=", "0"]
    corrupted = ["1", "+", "3", "*", "2", "=", "0"]
    clean_ok, clean_val, clean_claim = parse_and_evaluate(clean)
    corrupt_ok, corrupt_val, corrupt_claim = parse_and_evaluate(corrupted)
    features_match = {name: (fn(clean) == fn(corrupted)) for name, fn in SHORTCUT_FEATURES.items()}
    return {
        "clean": clean, "corrupted": corrupted,
        "clean_valid": clean_ok and clean_val == clean_claim,
        "corrupted_valid": corrupt_ok and corrupt_val == corrupt_claim,
        "clean_true_value": clean_val, "corrupted_true_value": corrupt_val,
        "claimed_value_both": clean_claim,
        "all_six_shortcut_features_identical": all(features_match.values()),
        "per_feature_match": features_match,
        "note": ("Same digit multiset {1,2,3}, same operator multiset {*,+}, same length, same "
                "first/last token type, same alternation pattern, same claimed final digit (0) -- "
                "but reordering the two (op,digit) pairs changes the TRUE left-to-right value from "
                "0 to 3, since '*' and '+' do not commute when interleaved. No combination of the "
                "six structural features can detect this."),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    train_seqs, train_labels = load_split("train")
    test_seqs, test_labels = load_split("test")

    print("=== confirming grammar via from-scratch evaluator (train split) ===", flush=True)
    n_match = sum(1 for s, l in zip(train_seqs, train_labels) if is_valid_example(s) == bool(l))
    grammar_confirm = {"n_total": len(train_seqs), "n_match": n_match, "fraction": n_match / len(train_seqs)}
    print(json.dumps(grammar_confirm, indent=2))
    assert grammar_confirm["fraction"] == 1.0, "expected exact match confirming the grammar"

    print("\n=== reproducing cited '99.4% structurally scrambled' figure (test split) ===", flush=True)
    test_neg_idx = [i for i, l in enumerate(test_labels) if l == 0]
    n_scrambled = sum(1 for i in test_neg_idx if not parse_and_evaluate(test_seqs[i])[0])
    n_hard = len(test_neg_idx) - n_scrambled
    scrambled_report = {
        "n_negatives": len(test_neg_idx), "n_structurally_scrambled": n_scrambled,
        "fraction_scrambled": n_scrambled / len(test_neg_idx),
        "n_grammatically_valid_but_wrong (hard)": n_hard,
        "fraction_hard": n_hard / len(test_neg_idx),
        "matches_cited_99_4_percent": abs(n_scrambled / len(test_neg_idx) - 0.994) < 0.002,
    }
    print(json.dumps(scrambled_report, indent=2))

    print("\n=== six shortcut features: per-feature + joint audit (test split) ===", flush=True)
    per_feature, joint = audit_shortcut_features(test_seqs, test_labels)
    print(json.dumps(per_feature, indent=2, default=str))
    print(json.dumps(joint, indent=2, default=str))
    assert joint["matches_grammatically_valid_negative_population"]

    print("\n=== load-bearing counterexample (reordered op,digit pairs) ===", flush=True)
    counterexample = build_counterexample()
    print(json.dumps(counterexample, indent=2, default=str))
    assert counterexample["all_six_shortcut_features_identical"]
    assert counterexample["clean_valid"] and not counterexample["corrupted_valid"]

    out = {
        "task": TASK,
        "description": "Phase 3 (bounded) task audit for modular-arithmetic-simple -- a single "
                       "accumulated mod-5 register carrier, order-sensitive via non-commuting "
                       "+/-/* operators, structurally distinct from every task piloted so far "
                       "(marker/durable-content, binary-addition/carry-chain, Dyck-2-3/bounded-stack).",
        "grammar": "digit (op digit)* '=' digit, digit in {0..4}, op in {+,-,*}, left-to-right, mod 5, "
                  "no operator precedence -- confirmed via 100% exact match of a from-scratch "
                  "evaluator against real training labels.",
        "grammar_confirmation": grammar_confirm,
        "cited_scrambled_fraction_reproduction": scrambled_report,
        "shortcut_feature_audit": {
            "per_feature": per_feature, "joint_all_six": joint,
            "all_features_trivial_counting_or_pattern_no_arithmetic_required": True,
            "target_computation_requires_order_sensitive_accumulation": (
                "The genuine target computation is a running mod-5 register updated by non-"
                "commuting operators -- NOT reducible to any combination of the six structural "
                "features audited above, demonstrated directly by the counterexample below."
            ),
        },
        "load_bearing_counterexample": counterexample,
    }
    out_path = RESULTS / "phase3_modarith_task_audit.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
