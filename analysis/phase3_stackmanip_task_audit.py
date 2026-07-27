"""Stack-manipulation Phase 3 (bounded, patching-only): task audit.

GRAMMAR, confirmed directly from the generator source (src/recognizers/
hand_picked_languages/stack_manipulation.py) and cross-checked against real
train/test data:

  Alphabet (5 symbols): '0', '1' (stack values), PUSH, POP (operations),
  '#' (MARKER, separates the operation program from its answer).

  Structure: [initial_stack: 0/1 tokens]* [operations: (PUSH bit | POP)]*
  MARKER [answer: reversed(final_stack)].

  _is_positive (verified against the real class): parse leading 0/1 tokens
  as the initial stack (push order = left to right, so the LAST bit read is
  the initial TOP of stack). Simulate the operations left to right: PUSH
  must be followed by a 0/1 bit (else reject); POP must have a non-empty
  stack (else reject). Exactly one MARKER must follow the operations
  (missing/absent = reject). The remaining suffix must EXACTLY equal
  reversed(final_stack) (both length and, per position, value).

STRUCTURAL RECLASSIFICATION (important, not assumed from the task's
resemblance to Dyck-2-3): despite both being "stack-carrier" tasks, stack-
manipulation is structurally in the MARKER-TRANSFORM family (bucket-sort,
marked-reversal, marked-copy, odds-first -- explicit MARKER separating
input from a deterministic transform of it) -- NOT Dyck-2-3's family
(implicit LIFO checking throughout, no marker, single global accept/
reject with no designated "answer" substring). The "transform" here
(simulate push/pop, emit reversed final stack) is more complex than sort/
reverse/copy, but the STRUCTURAL SHAPE matches the marker-transform tasks
directly, so marker-family-style candidate shortcuts (marker count/
position, answer-length arithmetic) apply here just as they did for
bucket-sort, plus operation-specific validity checks.

CANDIDATE STRUCTURAL SHORTCUTS (each a trivial counting/parsing check, no
real stack simulation of VALUES needed):
  marker_count_is_one           -- necessary, TRIVIAL O(1) count
  operations_well_formed        -- every PUSH followed by a bit, every POP
                                    has a non-empty stack at that point --
                                    EARLY detectable (provable at the exact
                                    violating token), not a late-only signal
  stack_size_arithmetic_consistent -- len(answer) == n_stack + n_push -
                                    n_pop (pure counting, no value-tracking;
                                    the CRITICAL confound check below)
  first_token_never_pop         -- necessary, single-token check
  token_before_marker_never_push -- necessary, single-token check
  answer_bit_counts_subset_of_available -- weak NECESSARY (not sufficient)
                                    condition: count of each bit value in
                                    the answer cannot exceed the count of
                                    that value ever available (initial
                                    stack + pushed values) -- analogous to
                                    Dyck-2-3's per-type bracket counts, but
                                    weaker here since WHICH values survive
                                    depends on LIFO order, not just totals

NO fixed length-parity constraint exists (verified: total length = 2*
n_stack + 3*n_push + 1, independent of n_pop and of n_push's parity --
both even AND odd total lengths occur among genuine positives, confirmed
below) -- unlike every marker-family task audited so far in this pilot.

CRITICAL METHODOLOGICAL CHECK (per the missing-duplicate-string lesson):
is flare_a2's hard-negative population confounded by a trivial O(1)
shortcut mislabeled "hard" by the incremental-detectability metric (viol_
stack_manipulation), the same way missing-duplicate-string's was? YES, and
MORE SEVERELY: of the 582 real test-set negatives classified "hard"
(frac>=0.8), 87.1% are pure stack_size_arithmetic_consistent violations
(length_mismatch -- a trivial running-counter check, PUSH-count minus POP-
count plus n_stack vs. observed answer length, requiring NO value
simulation) that the metric mislabels "hard" only because the mismatch is
strictly unconfirmable before EOS (you must see the whole suffix to know
its length). Only 9.5% of the nominally-"hard" population is genuinely
value_mismatch_only (structurally perfect -- correct marker, well-formed
operations, correct answer LENGTH -- but wrong VALUES, the only category
that actually requires real stack simulation to catch).

PYTHONPATH=src:analysis python analysis/phase3_stackmanip_task_audit.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "analysis")
from flare_a1_task_audit import viol_stack_manipulation, HARD_THRESHOLD

RESULTS = Path("analysis_outputs/final_results")
TASK = "stack-manipulation"
PUSH, POP, MARKER = "PUSH", "POP", "#"


def load_split(split):
    d = Path(f"languages/{TASK}") if split == "train" else Path(f"languages/{TASK}/datasets/{split}")
    toks = (d / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (d / "labels.txt").read_text().splitlines()]
    seqs = [line.split() for line in toks]
    return seqs, labels


def classify_violation(seq):
    """Mirrors _is_positive's exact parse, returning a fine-grained category
    instead of a bool. 'ACTUALLY_POSITIVE' means the sequence is genuinely
    valid (used as a self-check, not expected among the negative population)."""
    n = len(seq)
    i = 0
    stack = []
    while i < n and seq[i] in ("0", "1"):
        stack.append(seq[i])
        i += 1
    while i < n and seq[i] in (PUSH, POP):
        if seq[i] == PUSH:
            i += 1
            if i < n and seq[i] in ("0", "1"):
                stack.append(seq[i])
                i += 1
            else:
                return "malformed_push_no_bit"
        else:
            if not stack:
                return "malformed_pop_empty"
            stack.pop()
            i += 1
    if i < n and seq[i] == MARKER:
        i += 1
    else:
        return "no_marker_reached" if i >= n else "unexpected_token_before_marker"
    expected = list(reversed(stack))
    suffix = seq[i:]
    if len(suffix) != len(expected):
        return "length_mismatch"
    if suffix != expected:
        return "value_mismatch_only"
    return "ACTUALLY_POSITIVE"


def confirm_grammar(train_seqs, train_labels):
    pos = [s for s, l in zip(train_seqs, train_labels) if l == 1]
    first_never_pop = all(s[0] != "POP" for s in pos if s)
    any_empty = any(len(s) == 0 for s in pos)

    def tok_before_marker(s):
        i = s.index("#")
        return s[i - 1] if i > 0 else None

    tb_dist = Counter(tok_before_marker(s) for s in pos)
    never_push_before_marker = tb_dist.get("PUSH", 0) == 0
    lens = [len(s) for s in pos]
    n_even = sum(1 for l in lens if l % 2 == 0)
    n_odd = len(lens) - n_even

    return {
        "alphabet": ["0", "1", "PUSH", "POP", "#"],
        "n_positives_checked": len(pos),
        "first_token_never_pop": {"holds": first_never_pop, "any_empty_positive": any_empty},
        "token_before_marker_distribution": {str(k): v for k, v in tb_dist.items()},
        "token_before_marker_never_push": never_push_before_marker,
        "length_parity_both_occur (no fixed parity constraint)": {
            "n_even": n_even, "n_odd": n_odd, "both_occur": n_even > 0 and n_odd > 0,
        },
        "task_family_reclassification": (
            "MARKER-TRANSFORM family (bucket-sort/marked-reversal/marked-copy/odds-first), NOT "
            "Dyck-2-3's implicit-throughout-LIFO family, despite both being 'stack-carrier' tasks -- "
            "explicit MARKER separates the operation program from its deterministic answer (simulate "
            "push/pop, emit reversed final stack)."
        ),
    }


def marker_count_is_one(seq):
    return seq.count(MARKER) == 1


def operations_well_formed(seq):
    return classify_violation(seq) not in ("malformed_push_no_bit", "malformed_pop_empty",
                                            "unexpected_token_before_marker", "no_marker_reached")


def stack_size_arithmetic_consistent(seq):
    return classify_violation(seq) not in (
        "length_mismatch", "malformed_push_no_bit", "malformed_pop_empty",
        "unexpected_token_before_marker", "no_marker_reached",
    )


def first_token_never_pop(seq):
    return bool(seq) and seq[0] != "POP"


def token_before_marker_never_push(seq):
    if MARKER not in seq:
        return False
    i = seq.index(MARKER)
    return i == 0 or seq[i - 1] != "PUSH"


def answer_bit_counts_subset_of_available(seq):
    if seq.count(MARKER) != 1:
        return False
    i = seq.index(MARKER)
    prefix, suffix = seq[:i], seq[i + 1:]
    available = [t for t in prefix if t in ("0", "1")]
    avail_counts = Counter(available)
    ans_counts = Counter(t for t in suffix if t in ("0", "1"))
    if len(suffix) != sum(ans_counts.values()):
        return False  # suffix contains a non-bit token -- not a valid answer shape at all
    return ans_counts["0"] <= avail_counts["0"] and ans_counts["1"] <= avail_counts["1"]


SHORTCUT_FEATURES = {
    "marker_count_is_one": marker_count_is_one,
    "operations_well_formed": operations_well_formed,
    "stack_size_arithmetic_consistent": stack_size_arithmetic_consistent,
    "first_token_never_pop": first_token_never_pop,
    "token_before_marker_never_push": token_before_marker_never_push,
    "answer_bit_counts_subset_of_available": answer_bit_counts_subset_of_available,
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
        "positive_satisfy_all_six_fraction": joint_pos_satisfy / len(pos_idx) if pos_idx else float("nan"),
        "n_shortcut_blind_negatives (pass ALL SIX checks, still invalid)": len(shortcut_blind_neg_idx),
        "n_total_negatives": len(neg_idx),
        "shortcut_blind_fraction_of_negatives": len(shortcut_blind_neg_idx) / len(neg_idx) if neg_idx else float("nan"),
        "shortcut_blind_violation_kind_breakdown": dict(kind_counts),
        "interpretation": (
            "The shortcut-blind population is exactly the set of negatives all six counting/parsing "
            "checks CANNOT distinguish from a genuine positive -- a purely structural/counting "
            "baseline is chance-level on this subset by construction. Expected (and confirmed below) "
            "to be dominated by value_mismatch_only, the only category requiring genuine stack "
            "simulation."
        ),
    }
    return per_feature, joint, shortcut_blind_neg_idx


def build_counterexample():
    """Same marker count/position, same well-formed operations, same
    arithmetically-consistent answer length, same bit-count availability --
    but the answer's INTERNAL ORDER is wrong (a genuine LIFO violation:
    swaps two adjacent answer positions with differing values). Passes
    every listed shortcut feature, fails true stack-simulation
    verification."""
    # program: push 0, push 1 (stack bottom->top: 0,1) -> true answer (reversed): 1, 0
    clean = ["PUSH", "0", "PUSH", "1", "#", "1", "0"]
    corrupted = ["PUSH", "0", "PUSH", "1", "#", "0", "1"]  # answer order swapped
    features_match = {name: (fn(clean) == fn(corrupted)) for name, fn in SHORTCUT_FEATURES.items()}
    return {
        "clean": clean, "corrupted": corrupted,
        "clean_is_valid": classify_violation(clean) == "ACTUALLY_POSITIVE",
        "corrupted_is_valid": classify_violation(corrupted) == "ACTUALLY_POSITIVE",
        "corrupted_violation_kind": classify_violation(corrupted),
        "all_six_shortcut_features_identical_between_clean_and_corrupted": all(features_match.values()),
        "per_feature_match": features_match,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    train_seqs, train_labels = load_split("train")
    test_seqs, test_labels = load_split("test")

    print("=== confirming grammar from generator + real train data ===", flush=True)
    grammar = confirm_grammar(train_seqs, train_labels)
    print(json.dumps(grammar, indent=2, default=str))
    assert grammar["first_token_never_pop"]["holds"]
    assert grammar["token_before_marker_never_push"]
    assert grammar["length_parity_both_occur (no fixed parity constraint)"]["both_occur"]

    print("\n=== reproducing cited hard-negative fraction (viol_stack_manipulation, "
          "HARD_THRESHOLD=0.8, TEST split) ===", flush=True)
    test_neg_idx = [i for i, l in enumerate(test_labels) if l == 0]
    fracs = {i: viol_stack_manipulation(test_seqs[i]) for i in test_neg_idx}
    hard_idx = [i for i in test_neg_idx if fracs[i] >= HARD_THRESHOLD]
    hard_fraction = len(hard_idx) / len(test_neg_idx)
    print(f"  n_neg={len(test_neg_idx)} n_hard={len(hard_idx)} fraction={hard_fraction*100:.2f}% (cited: 22.7%)", flush=True)

    print("\n=== CRITICAL CONFOUND CHECK: is the 'hard' population dominated by a trivial "
          "O(1) shortcut, per the missing-duplicate-string lesson? ===", flush=True)
    hard_kind_counts = Counter(classify_violation(test_seqs[i]) for i in hard_idx)
    n_length_mismatch = hard_kind_counts.get("length_mismatch", 0)
    n_genuine_value_mismatch = hard_kind_counts.get("value_mismatch_only", 0)
    confound_check = {
        "n_hard": len(hard_idx),
        "hard_population_violation_kind_breakdown": dict(hard_kind_counts),
        "fraction_hard_that_is_length_mismatch_CONFOUND (trivial counting check)": n_length_mismatch / len(hard_idx),
        "fraction_hard_that_is_genuinely_value_mismatch_only": n_genuine_value_mismatch / len(hard_idx),
        "confound_confirmed": n_length_mismatch / len(hard_idx) > 0.5,
        "interpretation": (
            "CONFIRMED, and MORE SEVERE than missing-duplicate-string's confound: 87%+ of the "
            "nominally-'hard' population is length_mismatch (stack_size_arithmetic_consistent "
            "violations) -- a pure counting check (push-count minus pop-count plus initial-stack-"
            "size vs. observed answer length), requiring NO value simulation. It is classified "
            "'hard' by viol_stack_manipulation ONLY because the answer's total length cannot be "
            "confirmed before EOS (must see the whole suffix), not because it requires deep "
            "computation. Only ~9-10% of the 'hard' population is genuinely value_mismatch_only, "
            "the sole category requiring real LIFO simulation to catch. Any Phase-3 conclusion must "
            "be drawn from the value_mismatch_only-isolated population, NOT flare_a2's raw "
            "hard-negative accuracy statistic."
        ),
    }
    print(json.dumps(confound_check, indent=2, default=str))
    assert confound_check["confound_confirmed"]

    print("\n=== six candidate shortcut features: per-feature + joint audit (TEST split) ===", flush=True)
    per_feature, joint, shortcut_blind_neg_idx = audit_shortcut_features(test_seqs, test_labels)
    print(json.dumps(per_feature, indent=2, default=str))
    print(json.dumps(joint, indent=2, default=str))

    print("\n=== load-bearing counterexample (same shortcut features, wrong answer order) ===", flush=True)
    counterexample = build_counterexample()
    print(json.dumps(counterexample, indent=2, default=str))
    assert counterexample["all_six_shortcut_features_identical_between_clean_and_corrupted"]
    assert counterexample["clean_is_valid"] and not counterexample["corrupted_is_valid"]

    out = {
        "task": TASK,
        "description": (
            "Phase 3 (bounded) task audit for stack-manipulation. Reclassifies the task's family "
            "(marker-transform, like bucket-sort -- NOT Dyck-2-3's implicit-LIFO family, despite "
            "surface resemblance as a 'stack-carrier' task). Confirms and QUANTIFIES a confound in "
            "flare_a2's hard-negative metric analogous to (and more severe than) missing-duplicate-"
            "string's: 87%+ of the nominally-hard population is a trivial O(1) counting check "
            "(answer-length arithmetic), not genuine stack simulation."
        ),
        "grammar_confirmation": grammar,
        "cited_hard_negative_fraction_reproduction": {
            "test_set_n_negatives": len(test_neg_idx), "n_hard": len(hard_idx), "fraction": hard_fraction,
            "matches_cited_22_7_percent": abs(hard_fraction - 0.227) < 0.005,
        },
        "hard_negative_confound_check": confound_check,
        "shortcut_feature_audit": {
            "per_feature": per_feature, "joint_all_six": joint,
        },
        "load_bearing_counterexample": counterexample,
        "genuinely_hard_population_definition_for_later_conditions": (
            "marker_count_is_one AND operations_well_formed AND stack_size_arithmetic_consistent "
            "AND (value_mismatch_only for negatives) -- i.e., exclude both malformed-operation "
            "negatives (early-detectable) and length_mismatch negatives (the confound) before "
            "drawing conclusions from Condition 3 patching."
        ),
    }
    out_path = RESULTS / "phase3_stackmanip_task_audit.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
