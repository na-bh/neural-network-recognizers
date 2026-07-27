"""Missing-duplicate-string Phase 3 (bounded, patching-only): task audit.

Structural analysis of the FLaRe generator (src/recognizers/hand_picked_
languages/missing_duplicate_string.py) + empirical verification on the
actual train/test data.

GRAMMAR, confirmed directly from the generator source (not assumed from the
task name):
  alphabet_size() == 3: symbols '0', '1', and the blank marker '_'. There is
  NO richer symbol alphabet -- content is strictly BINARY (0/1); '_' is a
  redaction marker, not a content symbol.

  Generation (_sample_w / _w_to_string): draw a random bit-string w of
  length n (min_n=max(1,ceil(min_len/2)), max_n=floor(max_len/2)), force one
  position of w to '1' (so w always contains >=1 one). Form s = w + w (exact
  duplication, length 2n). Choose UNIFORMLY AT RANDOM among all positions in
  s that currently hold '1' (there are always >=2, one in each half-copy of
  the forced '1') and replace that one position with '_'. So every genuine
  positive is "two identical copies of a bit-string, with exactly one
  occurrence of '1' redacted to a blank."

  is_positive(s): reject if len(s) is odd. Reject if the number of '_' in s
  is not exactly 1. Otherwise let i be the blank's index, n=len(s)//2; set
  ss=list(s), ss[i]='1' (HARDCODED to 1, never tried as '0'); accept iff
  ss[:n] == ss[n:] (elementwise, ORDER-SENSITIVE).

KEY STRUCTURAL FINDING (corrects the pilot's initial framing): this is NOT
a "which of several symbols is the duplicate" task (cycle-navigation-style
aggregate counting over a rich alphabet). The alphabet is binary and the
blank ALWAYS stands for a redacted '1' by construction -- confirmed
empirically below that 0/5098 real positives would ALSO validate if the
blank were instead filled with '0' (zero ambiguity: the fill value is a
FIXED CONSTANT, not a computed quantity). The actual target computation is
an UNMARKED, MIDPOINT-IMPLICIT DUPLICATE-STRING CHECK: verify ss[j] ==
ss[half+j] for every j in [0,half) after hardcoding the blank to '1' -- the
same computational class as marked-copy's "does the suffix equal the
prefix" check (content-preserving carry across up to half the sequence
length), NOT an aggregate/majority-count task like cycle-navigation or
missing-parity tasks. There is no marker separating "string" from "answer";
recognition is single-shot accept/reject over the whole corrupted-copy
string.

Candidate structural shortcuts audited (each a trivial counting/positional
check, no content-preserving carry needed):
  sequence_length_parity   (len(s) even)              -- TRIVIAL, necessary
  blank_count_is_one                                   -- TRIVIAL, necessary
  aggregate_count_per_half (count('1' in each half, after
                             hardcoded blank fill, are equal)          -- the
                             "aggregate route" hypothesized by the pilot
  first_last_symbol_match  (s[0] == s[half] under blank-fill)          -- weak
                             positional proxy, single-position only

Empirically, the aggregate-count-per-half feature is INSUFFICIENT: among the
488 real train negatives that pass BOTH other necessary checks (even length,
exactly one blank -- i.e. the population any purely-structural shortcut
cannot already reject), 88 (18.0%) would be WRONGLY accepted by the
aggregate-count shortcut, since it ignores order entirely -- decisive
evidence the true recognizer requires genuine positional (order-sensitive)
comparison, not just symbol tallying.

PYTHONPATH=src:analysis python analysis/phase3_missdup_task_audit.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "analysis")
from flare_a1_task_audit import viol_missing_duplicate_string, HARD_THRESHOLD

RESULTS = Path("analysis_outputs/final_results")
TASK = "missing-duplicate-string"


def load_split(split):
    d = Path(f"languages/{TASK}") if split == "train" else Path(f"languages/{TASK}/datasets/{split}")
    toks = (d / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (d / "labels.txt").read_text().splitlines()]
    seqs = [line.split() for line in toks]
    return seqs, labels


def confirm_grammar(seqs, labels):
    alphabet = sorted({t for s in seqs for t in s})
    lens = [len(s) for s in seqs]
    pos_blank_counts = {}
    neg_blank_counts = {}
    for s, l in zip(seqs, labels):
        c = s.count("_")
        d = pos_blank_counts if l == 1 else neg_blank_counts
        d[c] = d.get(c, 0) + 1
    pos_lens_all_even = all(len(s) % 2 == 0 for s, l in zip(seqs, labels) if l == 1)
    pos_blank_all_one = all(s.count("_") == 1 for s, l in zip(seqs, labels) if l == 1)

    # zero-ambiguity check: would filling the blank with '0' instead of '1'
    # EVER also satisfy the halves-match condition for a genuine positive?
    n_fill0_also_matches = 0
    n_pos_checked = 0
    for s, l in zip(seqs, labels):
        if l != 1:
            continue
        n_pos_checked += 1
        i = s.index("_")
        n = len(s)
        half = n // 2
        ss = s[:]
        ss[i] = "0"
        if ss[:half] == ss[half:]:
            n_fill0_also_matches += 1

    return {
        "alphabet_confirmed": alphabet,
        "alphabet_is_binary_plus_blank": alphabet == ["0", "1", "_"],
        "length_range_seen": {"min": min(lens), "max": max(lens)},
        "positive_lengths_always_even": pos_lens_all_even,
        "positive_blank_count_always_exactly_one": pos_blank_all_one,
        "positive_blank_fill_value_is_fixed_constant_one_never_ambiguous": {
            "n_positives_checked": n_pos_checked,
            "n_where_filling_with_zero_would_also_match": n_fill0_also_matches,
            "conclusion": (
                "0 ambiguous cases confirms the blank deterministically stands for a redacted "
                "'1' by construction -- the fill value requires NO computation (constant), only "
                "the subsequent positional halves-match check is a genuine target computation."
            ) if n_fill0_also_matches == 0 else "AMBIGUITY FOUND -- framing above may be wrong",
        },
        "negative_blank_count_distribution": {str(k): v for k, v in sorted(neg_blank_counts.items())},
        "positive_blank_count_distribution": {str(k): v for k, v in sorted(pos_blank_counts.items())},
        "no_marker_separating_string_from_answer": True,
        "task_family_reclassification": (
            "Structurally an UNMARKED MIDPOINT-IMPLICIT DUPLICATE-STRING check (compare s[j] vs "
            "s[half+j] for all j, after a zero-computation constant blank-fill), same "
            "content-preserving-carry computational class as marked-copy -- NOT an aggregate/"
            "majority-counting task despite the 'which symbol is duplicated' framing suggested by "
            "the task name. The alphabet is binary, so 'which symbol' is trivial/constant; the "
            "real difficulty is WHETHER the two halves genuinely match position-by-position."
        ),
    }


# ---------------------------------------------------------------------------
# four candidate shortcut features
# ---------------------------------------------------------------------------

def sequence_length_parity(seq):
    return len(seq) % 2 == 0


def blank_count_is_one(seq):
    return seq.count("_") == 1


def aggregate_count_per_half(seq):
    """The 'aggregate route' hypothesized by the pilot: count of '1's in
    each half (after hardcoded blank->'1' fill) are equal. Only meaningful
    (computable) when length is even and there is exactly one blank; for
    other sequences this trivially returns False (they already fail a
    stronger necessary check)."""
    if len(seq) % 2 != 0 or seq.count("_") != 1:
        return False
    i = seq.index("_")
    half = len(seq) // 2
    ss = list(seq)
    ss[i] = "1"
    return ss[:half].count("1") == ss[half:].count("1")


def first_last_symbol_match(seq):
    """Weak single-position positional proxy: does the very first symbol of
    each half match (after blank-fill)? A necessary but very weak condition
    for genuine positional equality."""
    if len(seq) % 2 != 0 or seq.count("_") != 1:
        return False
    i = seq.index("_")
    half = len(seq) // 2
    ss = list(seq)
    ss[i] = "1"
    return ss[0] == ss[half]


SHORTCUT_FEATURES = {
    "sequence_length_parity": sequence_length_parity,
    "blank_count_is_one": blank_count_is_one,
    "aggregate_count_per_half": aggregate_count_per_half,
    "first_last_symbol_match": first_last_symbol_match,
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

    # the two TRIVIAL necessary checks (length parity, blank-count) isolate
    # the population any purely-structural pre-filter would already reject;
    # the remaining "genuinely hard" candidates are what the aggregate-route
    # feature is tested against
    def passes_trivial(seq):
        return sequence_length_parity(seq) and blank_count_is_one(seq)

    hard_candidate_neg_idx = [i for i in neg_idx if passes_trivial(seqs[i])]
    agg_wrongly_accepts = sum(1 for i in hard_candidate_neg_idx if aggregate_count_per_half(seqs[i]))

    joint = {
        "n_total_negatives": len(neg_idx),
        "n_genuinely_hard_candidates (pass length-parity AND blank-count-is-one)": len(hard_candidate_neg_idx),
        "fraction_of_negatives_genuinely_hard_candidates": len(hard_candidate_neg_idx) / len(neg_idx) if neg_idx else float("nan"),
        "n_hard_candidates_aggregate_route_would_wrongly_accept": agg_wrongly_accepts,
        "fraction_hard_candidates_aggregate_route_wrongly_accepts": agg_wrongly_accepts / len(hard_candidate_neg_idx) if hard_candidate_neg_idx else float("nan"),
        "interpretation": (
            "Among negatives that pass BOTH trivial structural checks (even length, exactly one "
            "blank -- i.e. everything a shallow structural shortcut cannot already reject), the "
            "aggregate-count-per-half 'route' still wrongly accepts a substantial minority as "
            "positive, since it ignores WITHIN-half symbol order entirely. This is decisive "
            "empirical evidence that aggregate symbol counting alone cannot solve the task -- "
            "true positional (order-sensitive) comparison is required for full accuracy."
        ),
    }
    return per_feature, joint, hard_candidate_neg_idx


def build_counterexample():
    """Same length, same blank count (1), same aggregate count-of-1s in each
    half (after blank-fill) -- but the SECOND half's internal order is
    permuted relative to the first half, so it fails the true positional
    check while passing the aggregate-route shortcut. This is the
    load-bearing demonstration that aggregate counting != the target
    computation."""
    # first half (after fill): 1 1 0 1  (two 1s, one 0... wait need matching counts)
    first_half = ["1", "1", "0", "1"]  # three 1s, one 0
    second_half_genuine = ["1", "1", "0", "1"]  # identical -> genuine positive (before blanking)
    second_half_permuted = ["1", "0", "1", "1"]  # same multiset (three 1s, one 0), different order
    clean = first_half + second_half_genuine
    corrupted = first_half + second_half_permuted
    # blank the same position (index 4, first '1' of second half) in both
    blank_idx = 4
    clean_s = clean[:]
    clean_s[blank_idx] = "_"
    corrupted_s = corrupted[:]
    corrupted_s[blank_idx] = "_"

    def is_positive_local(seq):
        if len(seq) % 2 != 0 or seq.count("_") != 1:
            return False
        i = seq.index("_")
        half = len(seq) // 2
        ss = list(seq)
        ss[i] = "1"
        return ss[:half] == ss[half:]

    features_match = {name: (fn(clean_s) == fn(corrupted_s)) for name, fn in SHORTCUT_FEATURES.items()}
    return {
        "clean": clean_s, "corrupted": corrupted_s,
        "clean_is_valid": is_positive_local(clean_s),
        "corrupted_is_valid": is_positive_local(corrupted_s),
        "aggregate_route_and_trivial_features_identical_between_clean_and_corrupted": all(features_match.values()),
        "per_feature_match": features_match,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    train_seqs, train_labels = load_split("train")
    test_seqs, test_labels = load_split("test")

    print("=== confirming grammar from generator + real train data ===", flush=True)
    grammar = confirm_grammar(train_seqs, train_labels)
    print(json.dumps(grammar, indent=2, default=str))
    assert grammar["positive_lengths_always_even"]
    assert grammar["positive_blank_count_always_exactly_one"]
    assert grammar["positive_blank_fill_value_is_fixed_constant_one_never_ambiguous"]["n_where_filling_with_zero_would_also_match"] == 0

    print("\n=== reproducing cited hard-negative fraction (viol_missing_duplicate_string, "
          "HARD_THRESHOLD=0.8, TEST split) ===", flush=True)
    test_neg_idx = [i for i, l in enumerate(test_labels) if l == 0]
    fracs = [viol_missing_duplicate_string(test_seqs[i]) for i in test_neg_idx]
    n_hard = sum(1 for f in fracs if f >= HARD_THRESHOLD)
    hard_fraction = n_hard / len(test_neg_idx)
    print(f"  n_neg={len(test_neg_idx)} n_hard={n_hard} fraction={hard_fraction*100:.2f}% (cited: ~26.7%)", flush=True)

    print("\n=== four candidate shortcut features: per-feature + joint audit (TRAIN split, larger n) ===", flush=True)
    per_feature, joint, hard_candidate_neg_idx = audit_shortcut_features(train_seqs, train_labels)
    print(json.dumps(per_feature, indent=2, default=str))
    print(json.dumps(joint, indent=2, default=str))

    print("\n=== load-bearing counterexample (same length/blank-count/aggregate-counts, permuted order) ===", flush=True)
    counterexample = build_counterexample()
    print(json.dumps(counterexample, indent=2, default=str))
    assert counterexample["aggregate_route_and_trivial_features_identical_between_clean_and_corrupted"]
    assert counterexample["clean_is_valid"] and not counterexample["corrupted_is_valid"]

    out = {
        "task": TASK,
        "description": (
            "Phase 3 (bounded) task audit for missing-duplicate-string. Corrects the pilot's "
            "initial framing (hypothesized as an aggregate/majority 'which symbol is duplicated' "
            "task akin to cycle-navigation): the real generator is BINARY-alphabet, and the "
            "correct task is an unmarked, midpoint-implicit duplicate-of-halves check with a "
            "single, semantically-constant ('always fill with 1') redacted position -- same "
            "content-preserving-carry computational class as marked-copy, not an aggregate-count "
            "task."
        ),
        "grammar_confirmation": grammar,
        "cited_hard_negative_fraction_reproduction": {
            "test_set_n_negatives": len(test_neg_idx),
            "n_hard": n_hard,
            "fraction": hard_fraction,
            "matches_cited_26_7_percent": abs(hard_fraction - 0.267) < 0.01,
        },
        "shortcut_feature_audit": {
            "per_feature": per_feature, "joint_hard_candidates_vs_aggregate_route": joint,
            "aggregate_route_alone_is_insufficient": joint["fraction_hard_candidates_aggregate_route_wrongly_accepts"] > 0,
        },
        "load_bearing_counterexample": counterexample,
        "condition_design_implications": {
            "condition1_structural_features": ["sequence_length_parity", "blank_count_is_one"],
            "condition2_aggregate_route_feature": "aggregate_count_per_half (count of '1's in each half after blank-fill, order-blind)",
            "condition3_target_identity_null_test_caveat": (
                "IMPORTANT: because the blank's fill value is a FIXED CONSTANT ('1', never '0', "
                "0/5098 ambiguous cases above), 'the identity of the duplicated symbol' as "
                "originally envisioned by the pilot brief (varying WHICH symbol is duplicated) is "
                "not constructible from genuine positives in this task -- there is no genuine "
                "positive where the answer is anything other than '1'. Condition 3 is REDEFINED "
                "here as a target-computation null test over WHICH HALF-POSITION PAIR is mismatched "
                "(i.e., whether models causally verify the SPECIFIC position of the halves-mismatch "
                "vs. some position-blind aggregate/parity signal) -- see "
                "phase3_missdup_counterfactual_design.json for the concrete pair construction."
            ),
        },
    }
    out_path = RESULTS / "phase3_missdup_task_audit.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
