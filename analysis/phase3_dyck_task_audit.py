"""Dyck-2-3 shortcut-exploitation pilot, Phase 3 (bounded, patching-only):
task audit. Structural analysis of the FLaRe generator (src/recognizers/
hand_picked_languages/dyck_k_m.py) + empirical verification on the actual
test-set data.

Dyck-2-3 = dyck_k_m_dfa(k=2, m=3): TWO bracket types ('(0'/')0', '(1'/')1'),
maximum nesting DEPTH 3 -- confirmed empirically below (NOT assumed from the
name), by comparing a from-scratch stack simulator against the real test
labels at several candidate max-depth values; max_depth=3 gives 100% exact
agreement (max_depth=2 only 59%, max_depth=4 only 99.9%), decisively
confirming both k and m and that DEPTH is a real, enforced constraint, not
just LIFO type-matching. This means Dyck-2-3's target computation combines
TWO stack-carrier requirements: (a) proper LIFO type-matching (a closer
must match the most-recently-opened, still-unclosed type) and (b) a bounded
DEPTH counter (nesting beyond 3 is invalid even if eventually balanced) --
a finer-grained "durable carrier" structure than the marker family (single
content-multiset carrier) or binary-addition (carry-propagation carrier).

Reused/extended from prior pilot infrastructure: flare_a1_task_audit.py's
viol_dyck() (first-provable-violation-position fraction, HARD_THRESHOLD=0.8)
reproduces the "11.7% hard-negative fraction" figure cited from Part 6B
EXACTLY on the TEST split (11.67%, not the training split's 21.26% -- the
cited number is test-set-specific, verified here rather than assumed).
FLAGGED GAP found and reported: viol_dyck() does NOT check max-depth at all
-- it only detects "closing with nothing open" and "mismatched closing
type," meaning a negative whose ONLY violation is exceeding max-depth-3
(otherwise perfectly LIFO-matched and eventually balanced) gets frac=1.0
("only provable at EOS") even though the true first-violation position is
the moment the 4th nested bracket opens, often much earlier. A corrected
violation function (viol_dyck_with_depth) is added here and the two are
compared directly on the actual hard-negative population to quantify this.

Four candidate shortcut features audited (per this experiment's design),
each classified as TRIVIAL COUNTING/POSITIONAL (no stack needed) vs
requiring genuine stack simulation:
  bracket_count_parity   (n_open == n_close, TOTAL)             -- TRIVIAL
  per_type_bracket_counts (n_open_t == n_close_t, PER TYPE)      -- TRIVIAL
  sequence_length_parity (len(seq) % 2 == 0)                     -- TRIVIAL
  first_last_check       (starts with an opener, ends with a
                           closer)                               -- TRIVIAL
  TARGET COMPUTATION (LIFO type-matching + depth<=3)             -- STACK

Each of the four is a NECESSARY (not sufficient) condition for a positive,
verified empirically (100% of positives satisfy each). The residual
"shortcut-blind" population -- negatives that pass ALL FOUR checks yet are
still invalid -- is exactly the population any purely-structural/counting
shortcut baseline CANNOT distinguish from a genuine positive, broken down
by whether the violation is type-mismatch, depth-exceeded, or both.

PYTHONPATH=src:analysis python analysis/phase3_dyck_task_audit.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "analysis")
from flare_a1_task_audit import viol_dyck, HARD_THRESHOLD

RESULTS = Path("analysis_outputs/final_results")
TASK = "dyck-2-3"
K, M = 2, 3


def load_split(split):
    d = Path(f"languages/{TASK}") if split == "train" else Path(f"languages/{TASK}/datasets/{split}")
    toks = (d / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (d / "labels.txt").read_text().splitlines()]
    seqs = [line.split() for line in toks]
    return seqs, labels


def is_valid_dyck(seq, max_depth=None):
    stack = []
    for t in seq:
        if t.startswith("("):
            stack.append(t[1:])
            if max_depth is not None and len(stack) > max_depth:
                return False
        elif t.startswith(")"):
            if not stack or stack.pop() != t[1:]:
                return False
        else:
            return False
    return len(stack) == 0


def confirm_k_m(seqs, labels):
    results = {}
    for max_depth in [2, 3, 4, None]:
        n_match = sum(1 for s, l in zip(seqs, labels) if is_valid_dyck(s, max_depth) == bool(l))
        results[str(max_depth)] = {"n_match": n_match, "n_total": len(seqs), "fraction": n_match / len(seqs)}
    return results


def viol_dyck_with_depth(seq, max_depth=M):
    """Corrected version of flare_a1_task_audit.py's viol_dyck: ALSO detects
    the moment nesting depth first exceeds max_depth, in addition to
    'closing with nothing open' and 'mismatched closing type'. Returns
    (violation_fraction, violation_kind)."""
    n = len(seq)
    stack = []
    for i, tok in enumerate(seq):
        if tok.startswith("("):
            stack.append(tok[1:])
            if len(stack) > max_depth:
                return i / n, "depth_exceeded"
        elif tok.startswith(")"):
            t = tok[1:]
            if not stack:
                return i / n, "close_with_nothing_open"
            top = stack.pop()
            if top != t:
                return i / n, "type_mismatch"
        else:
            return i / n, "invalid_token"
    return (1.0, "unclosed_at_eos") if stack else (0.0, "none")


# ---------------------------------------------------------------------------
# four candidate shortcut features -- all trivial counting/positional
# ---------------------------------------------------------------------------

OPENERS = {f"({k}" for k in range(K)}
CLOSERS = {f"){k}" for k in range(K)}


def bracket_count_parity(seq):
    n_open = sum(1 for t in seq if t in OPENERS)
    n_close = sum(1 for t in seq if t in CLOSERS)
    return n_open == n_close


def per_type_bracket_counts(seq):
    for k in range(K):
        n_open_k = sum(1 for t in seq if t == f"({k}")
        n_close_k = sum(1 for t in seq if t == f"){k}")
        if n_open_k != n_close_k:
            return False
    return True


def sequence_length_parity(seq):
    return len(seq) % 2 == 0


def first_last_check(seq):
    if not seq:
        return False
    return seq[0] in OPENERS and seq[-1] in CLOSERS


SHORTCUT_FEATURES = {
    "bracket_count_parity": bracket_count_parity,
    "per_type_bracket_counts": per_type_bracket_counts,
    "sequence_length_parity": sequence_length_parity,
    "first_last_check": first_last_check,
}


def audit_shortcut_features(seqs, labels):
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    per_feature = {}
    for name, fn in SHORTCUT_FEATURES.items():
        pos_satisfy = sum(1 for i in pos_idx if fn(seqs[i]))
        neg_satisfy = sum(1 for i in neg_idx if fn(seqs[i]))
        per_feature[name] = {
            "classification": "TRIVIAL_COUNTING_OR_POSITIONAL",
            "positive_satisfy_fraction": pos_satisfy / len(pos_idx) if pos_idx else float("nan"),
            "negative_satisfy_fraction": neg_satisfy / len(neg_idx) if neg_idx else float("nan"),
            "note": (
                "necessary-condition check: should be ~1.0 for positives (a real violation in "
                "this feature would make a positive fail, which should never happen); the "
                "negative_satisfy_fraction is this feature's OWN blind spot size."
            ),
        }

    # joint: passes ALL FOUR checks
    def passes_all(seq):
        return all(fn(seq) for fn in SHORTCUT_FEATURES.values())

    joint_pos_satisfy = sum(1 for i in pos_idx if passes_all(seqs[i]))
    joint_neg_satisfy = sum(1 for i in neg_idx if passes_all(seqs[i]))
    shortcut_blind_neg_idx = [i for i in neg_idx if passes_all(seqs[i])]

    # break down the shortcut-blind population by TRUE violation kind
    kind_counts = {}
    for i in shortcut_blind_neg_idx:
        _, kind = viol_dyck_with_depth(seqs[i])
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    joint = {
        "positive_satisfy_all_four_fraction": joint_pos_satisfy / len(pos_idx) if pos_idx else float("nan"),
        "negative_satisfy_all_four_fraction (shortcut-blind fraction)": joint_neg_satisfy / len(neg_idx) if neg_idx else float("nan"),
        "n_shortcut_blind_negatives": len(shortcut_blind_neg_idx),
        "n_total_negatives": len(neg_idx),
        "shortcut_blind_violation_kind_breakdown": kind_counts,
        "interpretation": (
            "The shortcut-blind population is exactly the set of negatives that four independent "
            "counting/positional checks (bracket-count parity, per-type counts, length parity, "
            "first/last check) CANNOT distinguish from a genuine positive -- a purely structural "
            "baseline is chance-level on this subset by construction, regardless of how many "
            "counting features it has access to, since none of them encode LIFO order or nesting "
            "depth."
        ),
    }
    return per_feature, joint, shortcut_blind_neg_idx


def build_counterexample():
    """Same bracket counts/positions as a genuine positive, but the closer at
    position 2 matches the WRONG (non-topmost) open type -- passes every
    listed shortcut feature, fails true stack-verification. Mirrors the
    classic ([)] confound, adapted to this task's '(k'/')k' token alphabet."""
    clean = ["(0", "(1", ")1", ")0"]  # genuine positive: (0(1)1)0
    corrupted = ["(0", "(1", ")0", ")1"]  # (0(1)0)1 -- closer at pos 2 matches wrong type
    clean_valid = is_valid_dyck(clean, M)
    corrupted_valid = is_valid_dyck(corrupted, M)
    features_match = {
        name: (fn(clean) == fn(corrupted)) for name, fn in SHORTCUT_FEATURES.items()
    }
    return {
        "clean": clean, "corrupted": corrupted,
        "clean_is_valid": clean_valid, "corrupted_is_valid": corrupted_valid,
        "all_four_shortcut_features_identical_between_clean_and_corrupted": all(features_match.values()),
        "per_feature_match": features_match,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    train_seqs, train_labels = load_split("train")
    test_seqs, test_labels = load_split("test")

    print("=== confirming k, m from the generator + real data (train split) ===", flush=True)
    km_confirm = confirm_k_m(train_seqs, train_labels)
    print(json.dumps(km_confirm, indent=2, default=str))
    assert km_confirm["3"]["fraction"] == 1.0, "expected max_depth=3 to give 100% exact match"

    print("\n=== reproducing cited hard-negative fraction (viol_dyck, HARD_THRESHOLD=0.8, TEST split) ===", flush=True)
    test_neg_idx = [i for i, l in enumerate(test_labels) if l == 0]
    fracs = [viol_dyck(test_seqs[i]) for i in test_neg_idx]
    n_hard_original = sum(1 for f in fracs if f >= HARD_THRESHOLD)
    original_hard_fraction = n_hard_original / len(test_neg_idx)
    print(f"  n_neg={len(test_neg_idx)} n_hard(viol_dyck)={n_hard_original} "
          f"fraction={original_hard_fraction*100:.2f}% (cited: 11.7%)", flush=True)

    print("\n=== depth-blindness gap in viol_dyck (flare_a1_task_audit.py) ===", flush=True)
    depth_gap = {"n_checked": 0, "n_reclassified_as_earlier_violation": 0, "examples": []}
    for i in test_neg_idx:
        seq = test_seqs[i]
        old_frac = viol_dyck(seq)
        if old_frac < HARD_THRESHOLD:
            continue  # only re-examine cases viol_dyck called "hard" (late/EOS-only detectable)
        depth_gap["n_checked"] += 1
        new_frac, kind = viol_dyck_with_depth(seq)
        if kind == "depth_exceeded" and new_frac < old_frac - 1e-9:
            depth_gap["n_reclassified_as_earlier_violation"] += 1
            if len(depth_gap["examples"]) < 3:
                depth_gap["examples"].append({
                    "seq_len": len(seq), "viol_dyck_frac": old_frac,
                    "viol_dyck_with_depth_frac": new_frac, "true_kind": kind,
                })
    depth_gap["fraction_of_viol_dyck_hard_negatives_actually_depth_exceeded_earlier"] = (
        depth_gap["n_reclassified_as_earlier_violation"] / depth_gap["n_checked"] if depth_gap["n_checked"] else float("nan")
    )
    print(json.dumps(depth_gap, indent=2, default=str))

    print("\n=== four shortcut features: per-feature + joint audit (TEST split) ===", flush=True)
    per_feature, joint, shortcut_blind_neg_idx = audit_shortcut_features(test_seqs, test_labels)
    print(json.dumps(per_feature, indent=2, default=str))
    print(json.dumps(joint, indent=2, default=str))

    print("\n=== load-bearing counterexample (same counts/positions, wrong closer type) ===", flush=True)
    counterexample = build_counterexample()
    print(json.dumps(counterexample, indent=2, default=str))
    assert counterexample["all_four_shortcut_features_identical_between_clean_and_corrupted"]
    assert counterexample["clean_is_valid"] and not counterexample["corrupted_is_valid"]

    out = {
        "task": TASK, "k_bracket_types": K, "m_max_nesting_depth": M,
        "description": "Phase 3 (bounded) task audit for Dyck-2-3, a stack-carrier formal language "
                       "-- structurally distinct from the marker-family (durable content carrier) "
                       "and arithmetic (carry-propagation carrier) tasks already piloted.",
        "k_m_confirmation": km_confirm,
        "cited_hard_negative_fraction_reproduction": {
            "test_set_n_negatives": len(test_neg_idx),
            "n_hard_by_viol_dyck": n_hard_original,
            "fraction": original_hard_fraction,
            "matches_cited_11_7_percent": abs(original_hard_fraction - 0.117) < 0.002,
        },
        "viol_dyck_depth_blindness_gap": depth_gap,
        "shortcut_feature_audit": {
            "per_feature": per_feature, "joint_all_four": joint,
            "all_features_trivial_counting_or_positional_no_stack_required": True,
            "target_computation_requires_stack": (
                "TRUE target computation requires LIFO type-matching (a closer must match the most "
                "recently opened, still-unclosed type) AND a bounded depth counter (nesting beyond "
                "m=3 is invalid) -- neither is reducible to any combination of the four counting/"
                "positional features audited above, demonstrated directly by the counterexample below."
            ),
        },
        "load_bearing_counterexample": counterexample,
    }
    out_path = RESULTS / "phase3_dyck_task_audit.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
