"""Repeat-01 Phase 3 (bounded, patching-only): task audit.

GRAMMAR, confirmed directly from the generator source (src/recognizers/
hand_picked_languages/repeat_01.py) -- a DFA-based task (unlike every other
task audited in this pilot, which used a custom WeightedLanguage.sample()):
a 2-state finite automaton, q0 (initial AND accepting) --'0'--> q1 --'1'-->
q0. Undefined transitions (q0 on '1', q1 on '0') have no successor, so any
deviation is an immediate dead-state reject. This DFA accepts exactly
(01)* -- strings that are the empty string or repeated '01' blocks.

Equivalently (and exactly matching flare_a1_task_audit.py's existing viol_
repeat_01 classifier, reused here): s[i] == '0' if i even else '1', for
EVERY position i, AND len(s) even. Verified directly against 5026 real
positives -- 100% match, zero exceptions -- and confirmed that 0/4974 real
negatives accidentally satisfy this pattern.

Alphabet: {'0','1'} (2 symbols, no marker or other structural token) -- the
simplest alphabet of any task in this pilot.

CANDIDATE STRUCTURAL SHORTCUTS (four features, all trivial O(1) checks):
  length_parity_even   -- len(s) % 2 == 0
  first_symbol_is_0    -- s[0] == '0'
  last_symbol_is_1     -- s[-1] == '1' (for even length, index len-1 is odd)
  count_balance         -- count('0') == count('1') (every genuine '01'
                            block contributes exactly one of each)

CRITICAL AUDIT FINDING (not assumed, discovered by comparing the two
candidate "hard" definitions directly): flare_a1_task_audit.py's viol_
repeat_01-based "hard" classification (frac>=0.8, first-mismatch-position-
based) and the TRUE shortcut-blind population (passes all four structural
features, requires genuine per-position verification) are LARGELY DISJOINT
for this task -- unlike every other task audited in this pilot, where the
two definitions closely coincided. Of the 151 negatives viol_repeat_01
classifies "hard," only 5 are actually shortcut-blind (96.7% of the
nominal "hard" population is trivially catchable, confirming the user's
prediction). But the shortcut-blind population itself is LARGER (115
negatives, 4.6% of all negatives) and SPREAD ACROSS ALL POSITIONS (mean
first-violation-fraction 0.25, NOT concentrated late -- 58/115 have their
first violation in the very first quintile of the string). This happens
because expected[i] is a PURE FUNCTION OF POSITION i (not history-
dependent), so a count-preserving corruption (e.g. an adjacent swap) can
have its first detectable violation at ANY position, not preferentially
late -- unlike the marker/count-family tasks elsewhere in this pilot, where
genuinely-hard violations were structurally forced to be late-detectable
(you need to see the whole marker section, or the whole answer field,
before a mismatch is provable). CONSEQUENCE: for this task, viol_repeat_
01's "hard negative" statistic is a POOR proxy for "genuinely requires
verification" -- the correct genuinely-hard population for Condition 3 is
the 115-example shortcut-blind set defined directly, not the 151-example
viol-hard set.

PYTHONPATH=src:analysis python analysis/phase3_repeat01_task_audit.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from flare_a1_task_audit import viol_repeat_01, HARD_THRESHOLD

RESULTS = Path("analysis_outputs/final_results")
TASK = "repeat-01"


def load_split(split):
    d = Path(f"languages/{TASK}") if split == "train" else Path(f"languages/{TASK}/datasets/{split}")
    toks = (d / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (d / "labels.txt").read_text().splitlines()]
    seqs = [line.split() for line in toks]
    return seqs, labels


def classify_violation(seq):
    n = len(seq)
    mismatches = [i for i, t in enumerate(seq) if t != ("0" if i % 2 == 0 else "1")]
    if not mismatches:
        return "length_parity_only" if n % 2 == 1 else "ACTUALLY_POSITIVE"
    return "alternation_mismatch"


def length_parity_even(seq):
    return len(seq) % 2 == 0


def first_symbol_is_0(seq):
    return bool(seq) and seq[0] == "0"


def last_symbol_is_1(seq):
    return bool(seq) and seq[-1] == "1"


def count_balance(seq):
    return seq.count("0") == seq.count("1")


SHORTCUT_FEATURES = {
    "length_parity_even": length_parity_even,
    "first_symbol_is_0": first_symbol_is_0,
    "last_symbol_is_1": last_symbol_is_1,
    "count_balance": count_balance,
}


def confirm_grammar(train_seqs, train_labels):
    def is_alternating_01(s):
        if len(s) % 2 != 0:
            return False
        return all(s[i] == ("0" if i % 2 == 0 else "1") for i in range(len(s)))

    pos = [s for s, l in zip(train_seqs, train_labels) if l == 1]
    neg = [s for s, l in zip(train_seqs, train_labels) if l == 0]
    n_pos_match = sum(1 for s in pos if is_alternating_01(s))
    n_neg_accidentally_match = sum(1 for s in neg if is_alternating_01(s))
    return {
        "alphabet": ["0", "1"],
        "dfa_based": True,
        "target": "(01)* -- s[i]=='0' if i even else '1', for all i, AND len(s) even",
        "n_positives_checked": len(pos),
        "positive_match_fraction": n_pos_match / len(pos) if pos else float("nan"),
        "n_negatives_checked": len(neg),
        "n_negatives_accidentally_matching": n_neg_accidentally_match,
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

    joint_pos_satisfy = sum(1 for i in pos_idx if passes_all(seqs[i]))
    shortcut_blind_neg_idx = [i for i in neg_idx if passes_all(seqs[i])]
    kind_counts = Counter(classify_violation(seqs[i]) for i in shortcut_blind_neg_idx)

    joint = {
        "positive_satisfy_all_four_fraction": joint_pos_satisfy / len(pos_idx) if pos_idx else float("nan"),
        "n_shortcut_blind_negatives (pass ALL FOUR checks, still invalid)": len(shortcut_blind_neg_idx),
        "n_total_negatives": len(neg_idx),
        "shortcut_blind_fraction_of_negatives": len(shortcut_blind_neg_idx) / len(neg_idx) if neg_idx else float("nan"),
        "shortcut_blind_violation_kind_breakdown": dict(kind_counts),
    }
    return per_feature, joint, shortcut_blind_neg_idx


def build_counterexample():
    """Genuine positive (01)^4 = 0101 0101. Adjacent swap at positions 2,3
    (values '0','1' -- differing, matching the alternating pattern's own
    adjacent-pair structure) -- preserves length, first/last symbol, AND
    count balance (a swap can't change a multiset), while creating a
    DOUBLE alternation violation at positions 2 and 3."""
    clean = ["0", "1", "0", "1", "0", "1", "0", "1"]
    corrupted = clean[:]
    corrupted[2], corrupted[3] = corrupted[3], corrupted[2]  # 0,1 -> 1,0
    features_match = {name: (fn(clean) == fn(corrupted)) for name, fn in SHORTCUT_FEATURES.items()}
    return {
        "clean": clean, "corrupted": corrupted,
        "clean_violation": classify_violation(clean), "corrupted_violation": classify_violation(corrupted),
        "all_four_shortcut_features_identical_between_clean_and_corrupted": all(features_match.values()),
        "per_feature_match": features_match,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    train_seqs, train_labels = load_split("train")
    test_seqs, test_labels = load_split("test")

    print("=== confirming grammar from generator + real train data ===", flush=True)
    grammar = confirm_grammar(train_seqs, train_labels)
    print(json.dumps(grammar, indent=2, default=str))
    assert grammar["positive_match_fraction"] == 1.0
    assert grammar["n_negatives_accidentally_matching"] == 0

    print("\n=== reproducing cited hard-negative fraction (viol_repeat_01, "
          "HARD_THRESHOLD=0.8, TEST split) ===", flush=True)
    test_neg_idx = [i for i, l in enumerate(test_labels) if l == 0]
    fracs = {i: viol_repeat_01(test_seqs[i]) for i in test_neg_idx}
    hard_idx = [i for i in test_neg_idx if fracs[i] >= HARD_THRESHOLD]
    hard_fraction = len(hard_idx) / len(test_neg_idx)
    print(f"  n_neg={len(test_neg_idx)} n_hard={len(hard_idx)} fraction={hard_fraction*100:.2f}% "
          f"(smallest of any task audited in this pilot so far)", flush=True)

    print("\n=== CRITICAL CHECK: does viol-hard coincide with genuinely-shortcut-blind? ===", flush=True)
    hard_kind_counts = Counter(classify_violation(test_seqs[i]) for i in hard_idx)

    def passes_all(seq):
        return all(fn(seq) for fn in SHORTCUT_FEATURES.values())

    n_hard_shortcut_blind = sum(1 for i in hard_idx if passes_all(test_seqs[i]))
    divergence_check = {
        "n_hard": len(hard_idx),
        "hard_population_violation_kind_breakdown": dict(hard_kind_counts),
        "n_hard_that_is_ALSO_shortcut_blind": n_hard_shortcut_blind,
        "fraction_hard_that_is_trivially_catchable": 1 - n_hard_shortcut_blind / len(hard_idx),
        "interpretation": (
            "CONFIRMED: 96.7% of the nominally-'hard' population is trivially catchable by the four "
            "structural shortcuts (matching the user's prediction). But this does NOT mean the task "
            "has almost no genuinely-hard population -- see the full shortcut-blind audit below, "
            "which finds a LARGER (115-example) shortcut-blind population spread across ALL "
            "positions, not concentrated late the way viol_repeat_01's classification would suggest."
        ),
    }
    print(json.dumps(divergence_check, indent=2, default=str))

    print("\n=== four candidate shortcut features: per-feature + joint audit (TEST split) ===", flush=True)
    per_feature, joint, shortcut_blind_neg_idx = audit_shortcut_features(test_seqs, test_labels)
    print(json.dumps(per_feature, indent=2, default=str))
    print(json.dumps(joint, indent=2, default=str))

    blind_fracs = [viol_repeat_01(test_seqs[i]) for i in shortcut_blind_neg_idx]
    arr = np.array(blind_fracs)
    hist, _ = np.histogram(arr, bins=5)
    position_distribution = {
        "n_shortcut_blind": len(shortcut_blind_neg_idx),
        "first_violation_fraction_mean": float(arr.mean()),
        "first_violation_fraction_histogram_5bins_0to1": hist.tolist(),
        "n_classified_hard_by_viol_among_shortcut_blind (frac>=0.8)": int((arr >= HARD_THRESHOLD).sum()),
        "interpretation": (
            "The 115 shortcut-blind negatives' first-violation positions are spread across the "
            "WHOLE string (mean fraction 0.25, most concentrated EARLY not late) -- because "
            "expected[i] is a pure function of position i, not history-dependent, a count-"
            "preserving corruption (e.g. adjacent swap) can be detected at any position, unlike the "
            "marker/count-family tasks elsewhere in this pilot where genuine hardness was "
            "structurally forced to be late-detectable."
        ),
    }
    print("\n=== position distribution of the TRUE shortcut-blind population ===", flush=True)
    print(json.dumps(position_distribution, indent=2, default=str))

    print("\n=== load-bearing counterexample (adjacent swap, same shortcut features) ===", flush=True)
    counterexample = build_counterexample()
    print(json.dumps(counterexample, indent=2, default=str))
    assert counterexample["all_four_shortcut_features_identical_between_clean_and_corrupted"]
    assert counterexample["clean_violation"] == "ACTUALLY_POSITIVE" and counterexample["corrupted_violation"] == "alternation_mismatch"

    out = {
        "task": TASK,
        "description": (
            "Phase 3 (bounded) task audit for repeat-01, the structurally simplest task in this "
            "pilot's shortcut-vulnerable coverage (DFA-based, 2-symbol alphabet, no structural "
            "tokens). Confirms the (01)* target exactly. KEY FINDING: viol_repeat_01's 'hard "
            "negative' classification and the TRUE shortcut-blind population are largely disjoint "
            "for this task (only 5/151 overlap) -- the correct genuinely-hard population for "
            "Condition 3 is the 115-example shortcut-blind set, defined directly via the four "
            "structural features, not the viol-based classification."
        ),
        "grammar_confirmation": grammar,
        "cited_hard_negative_fraction_reproduction": {
            "test_set_n_negatives": len(test_neg_idx), "n_hard": len(hard_idx), "fraction": hard_fraction,
        },
        "hard_vs_shortcut_blind_divergence_check": divergence_check,
        "shortcut_feature_audit": {"per_feature": per_feature, "joint_all_four": joint},
        "genuinely_hard_population_position_distribution": position_distribution,
        "load_bearing_counterexample": counterexample,
        "genuinely_hard_population_definition_for_later_conditions": (
            "length_parity_even AND first_symbol_is_0 AND last_symbol_is_1 AND count_balance AND "
            f"alternation_mismatch (NOT the viol_repeat_01 'hard' classification) -- "
            f"{joint['n_shortcut_blind_negatives (pass ALL FOUR checks, still invalid)']} real "
            "test-set examples, spread across all positions."
        ),
    }
    out_path = RESULTS / "phase3_repeat01_task_audit.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
