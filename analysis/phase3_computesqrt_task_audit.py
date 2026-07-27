"""Compute-sqrt Phase 3 (bounded, patching-only): task audit.

GRAMMAR, confirmed directly from the generator source (src/recognizers/
hand_picked_languages/compute_sqrt.py + binary_util.py) and cross-checked
against real train/test data:

  Alphabet (3 symbols): '0', '1' (binary digits), '=' (EQUALS, single
  separator marker).

  Structure: [u_x: n_x bits] '=' [u_z: n_z bits]. BOTH u_x and u_z are
  LSB-FIRST (little-endian) fixed-width binary encodings -- confirmed
  directly via binary_util.decode_binary's bit-shift loop (mask starts at
  1, shifts LEFT each position) and binary_encoding's zero-padding always
  appended at the END (correct for LSB-first fixed-width padding). This
  means the LEFTMOST token of each field is the LEAST significant bit and
  the RIGHTMOST token is the MOST significant bit -- the OPPOSITE of
  standard MSB-first reading order. Confirmed empirically: decoding 2000
  real positives this way gives math.isqrt(x) == z for 2000/2000.

  EXACT target computation (confirmed via _is_positive AND cross-checked
  against math.isqrt): z must equal floor(sqrt(x)) EXACTLY -- the INTEGER
  square root (z**2 <= x < (z+1)**2), NOT an exact-perfect-square check
  (x need not be a perfect square).

  IMPORTANT: unlike stack-manipulation/missing-duplicate-string, the
  answer field's WIDTH (n_z) does NOT need to exactly equal the true
  value's bit length -- decode_binary treats any EXTRA high-order bits as
  additional value bits, so u_z may have trailing (high-order, since LSB-
  first) zero-padding of ANY length beyond what z truly needs, and still
  decode to the same integer. The only TRUE necessary length constraint is
  n_z >= bitlength(floor(sqrt(x))) (the field must be WIDE ENOUGH to even
  represent the true answer) -- confirmed mathematically and empirically:
  bitlength(isqrt(x)) == ceil(bitlength(x)/2) exactly for all x>=1 (spot-
  checked x in {3,4,8,15,16}), meaning this length-sufficiency check is a
  TRIVIAL O(1) computation from x's bit-length alone -- no square-root
  VALUE computation needed, matching the user's "answer digit length is
  about half the input's" hypothesis, refined to an exact inequality (n_z
  >= ceil(bitlength(x)/2)), not an exact equality (extra width is legal).

  n_x and n_z are otherwise sampled INDEPENDENTLY (a Dirichlet split of a
  shared length budget) -- x is directly CAPPED at generation time to
  min(2**n_x-1, 2**(2*n_z)-1), which is exactly what guarantees the above
  length-sufficiency necessary condition holds for every genuine positive.

  NO fixed parity (mod-2) relationship exists between x and z -- checked
  explicitly per the user's request: all four (x_parity, z_parity)
  combinations occur with substantial frequency among real positives
  (925-1532 each out of 5077), ruling out a parity-based shortcut.

CANDIDATE STRUCTURAL SHORTCUTS (each a trivial counting/parsing check, no
real square-root VALUE computation needed):
  marker_count_is_one           -- necessary, TRIVIAL O(1) count
  fields_well_formed            -- u_x and u_z both non-empty
  answer_length_sufficient      -- n_z >= ceil(bitlength(x)/2) -- the
                                    CRITICAL length-based shortcut,
                                    computable without ever computing sqrt

CRITICAL METHODOLOGICAL CHECK (per the missing-duplicate-string / stack-
manipulation lesson): is flare_a2's hard-negative population confounded by
a trivial O(1) shortcut mislabeled "hard" by the incremental-detectability
metric (viol_compute_sqrt)? PARTIALLY, and LESS SEVERELY than the prior two
tasks: of the 477 real test-set negatives classified "hard" (frac>=0.8;
reproduces the A1 audit's cited 19.18% exactly), 46.5% are multiple_markers
(a second '=' that happens to occur late for THIS negative-generation
distribution -- still a trivial O(1) count, just late-confirmable here),
25.0% are length_insufficient (the trivial length-sufficiency check), 4.8%
are no_marker/empty-field -- 76.9% total confound. The REMAINING 23.1% is
genuinely value_mismatch_only (well-formed, sufficient length, but the
decoded VALUE is wrong) -- requiring real square-root verification to
catch. This genuinely-hard fraction (23.1% of the "hard" bucket, and 16.5%
of ALL negatives -- 411/2487) is SUBSTANTIALLY LARGER than missing-
duplicate-string's (~9.5% of "hard", 2.4% of all negatives) or stack-
manipulation's (~9.5% of "hard", 2.4% of all negatives) -- compute-sqrt's
confound is real but much less totalizing.

PYTHONPATH=src:analysis python analysis/phase3_computesqrt_task_audit.py
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "analysis")
from flare_a1_task_audit import viol_compute_sqrt, HARD_THRESHOLD

RESULTS = Path("analysis_outputs/final_results")
TASK = "compute-sqrt"
EQUALS = "="


def load_split(split):
    d = Path(f"languages/{TASK}") if split == "train" else Path(f"languages/{TASK}/datasets/{split}")
    toks = (d / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (d / "labels.txt").read_text().splitlines()]
    seqs = [line.split() for line in toks]
    return seqs, labels


def decode_binary(bits):
    x = 0
    for i, b in enumerate(bits):
        if b == "1":
            x |= (1 << i)
    return x


def bitlength(v):
    return v.bit_length() if v > 0 else 1


def classify_violation(seq):
    eq_idx = [i for i, t in enumerate(seq) if t == EQUALS]
    if len(eq_idx) == 0:
        return "no_marker"
    if len(eq_idx) >= 2:
        return "multiple_markers"
    i = eq_idx[0]
    u_x, u_z = seq[:i], seq[i + 1:]
    if not u_x:
        return "empty_ux"
    if not u_z:
        return "empty_uz"
    x = decode_binary(u_x)
    z = decode_binary(u_z)
    z_true = math.isqrt(x)
    need_bits = bitlength(z_true)
    if len(u_z) < need_bits:
        return "length_insufficient"
    if z != z_true:
        return "value_mismatch_only"
    return "ACTUALLY_POSITIVE"


def confirm_grammar(train_seqs, train_labels):
    pos = [s for s, l in zip(train_seqs, train_labels) if l == 1]
    n_checked = 0
    n_match = 0
    for s in pos:
        i = s.index(EQUALS)
        x = decode_binary(s[:i])
        z = decode_binary(s[i + 1:])
        n_checked += 1
        if math.isqrt(x) == z:
            n_match += 1

    length_ok = 0
    for s in pos:
        i = s.index(EQUALS)
        u_x, u_z = s[:i], s[i + 1:]
        x = decode_binary(u_x)
        need = bitlength(math.isqrt(x))
        if len(u_z) >= need:
            length_ok += 1

    parity_dist = Counter()
    for s in pos:
        i = s.index(EQUALS)
        x = decode_binary(s[:i])
        z = decode_binary(s[i + 1:])
        parity_dist[(x % 2, z % 2)] += 1

    return {
        "alphabet": ["0", "1", "="],
        "encoding": "LSB-first (little-endian) fixed-width binary for both u_x and u_z",
        "n_positives_checked": n_checked,
        "floor_sqrt_exact_match_fraction": n_match / n_checked if n_checked else float("nan"),
        "answer_length_sufficient_always_holds": length_ok / n_checked if n_checked else float("nan"),
        "bitlength_isqrt_equals_ceil_half_bitlength_x": (
            "mathematically exact identity, spot-checked: bitlength(isqrt(x)) == ceil(bitlength(x)/2) "
            "for x in {3,4,8,15,16} -- a trivial O(1) shortcut requiring no sqrt VALUE computation."
        ),
        "x_z_parity_joint_distribution_on_positives": {str(k): v for k, v in parity_dist.items()},
        "no_fixed_mod2_relationship_between_x_and_z": len(parity_dist) == 4,
    }


def marker_count_is_one(seq):
    return seq.count(EQUALS) == 1


def fields_well_formed(seq):
    if seq.count(EQUALS) != 1:
        return False
    i = seq.index(EQUALS)
    return bool(seq[:i]) and bool(seq[i + 1:])


def answer_length_sufficient(seq):
    if not fields_well_formed(seq):
        return False
    i = seq.index(EQUALS)
    u_x, u_z = seq[:i], seq[i + 1:]
    x = decode_binary(u_x)
    need = bitlength(math.isqrt(x))
    return len(u_z) >= need


SHORTCUT_FEATURES = {
    "marker_count_is_one": marker_count_is_one,
    "fields_well_formed": fields_well_formed,
    "answer_length_sufficient": answer_length_sufficient,
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
        "positive_satisfy_all_three_fraction": joint_pos_satisfy / len(pos_idx) if pos_idx else float("nan"),
        "n_shortcut_blind_negatives (pass ALL THREE checks, still invalid)": len(shortcut_blind_neg_idx),
        "n_total_negatives": len(neg_idx),
        "shortcut_blind_fraction_of_negatives": len(shortcut_blind_neg_idx) / len(neg_idx) if neg_idx else float("nan"),
        "shortcut_blind_violation_kind_breakdown": dict(kind_counts),
        "interpretation": (
            "The shortcut-blind population is exactly the set of negatives all three counting/"
            "parsing checks CANNOT distinguish from a genuine positive -- SUBSTANTIALLY LARGER here "
            "(16.5% of all negatives) than for missing-duplicate-string/stack-manipulation (~2.4% "
            "each), meaning compute-sqrt's structural shortcuts leave much more of the real "
            "negative distribution requiring genuine sqrt verification to catch."
        ),
    }
    return per_feature, joint, shortcut_blind_neg_idx


def build_counterexample():
    """x=9 (u_x LSB-first: 1,0,0,1 -- 9=0b1001), floor(sqrt(9))=3 (u_z LSB-
    first: 1,1 -- 3=0b11). corrupted: same x, same answer LENGTH (n_z=2,
    answer_length_sufficient holds: bitlength(3)=2<=2), same marker count/
    well-formedness, but z=2 (u_z: 0,1) instead of 3 -- passes every listed
    shortcut feature, fails true sqrt verification."""
    clean = ["1", "0", "0", "1", "=", "1", "1"]  # x=9, z=3 (correct: isqrt(9)=3)
    corrupted = ["1", "0", "0", "1", "=", "0", "1"]  # x=9, z=2 (wrong: isqrt(9)=3, not 2)
    features_match = {name: (fn(clean) == fn(corrupted)) for name, fn in SHORTCUT_FEATURES.items()}
    return {
        "clean": clean, "corrupted": corrupted,
        "clean_x": decode_binary(clean[:4]), "clean_z": decode_binary(clean[5:]),
        "corrupted_z": decode_binary(corrupted[5:]),
        "clean_violation": classify_violation(clean), "corrupted_violation": classify_violation(corrupted),
        "all_three_shortcut_features_identical_between_clean_and_corrupted": all(features_match.values()),
        "per_feature_match": features_match,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    train_seqs, train_labels = load_split("train")
    test_seqs, test_labels = load_split("test")

    print("=== confirming grammar from generator + real train data ===", flush=True)
    grammar = confirm_grammar(train_seqs, train_labels)
    print(json.dumps(grammar, indent=2, default=str))
    assert grammar["floor_sqrt_exact_match_fraction"] == 1.0
    assert grammar["answer_length_sufficient_always_holds"] == 1.0
    assert grammar["no_fixed_mod2_relationship_between_x_and_z"]

    print("\n=== reproducing cited hard-negative fraction (viol_compute_sqrt, "
          "HARD_THRESHOLD=0.8, TEST split) ===", flush=True)
    test_neg_idx = [i for i, l in enumerate(test_labels) if l == 0]
    fracs = {i: viol_compute_sqrt(test_seqs[i]) for i in test_neg_idx}
    hard_idx = [i for i in test_neg_idx if fracs[i] >= HARD_THRESHOLD]
    hard_fraction = len(hard_idx) / len(test_neg_idx)
    print(f"  n_neg={len(test_neg_idx)} n_hard={len(hard_idx)} fraction={hard_fraction*100:.2f}% "
          f"(A1 audit cited: 19.18%)", flush=True)

    print("\n=== CONFOUND CHECK: is the 'hard' population dominated by trivial O(1) shortcuts, "
          "per the missing-duplicate-string/stack-manipulation lesson? ===", flush=True)
    hard_kind_counts = Counter(classify_violation(test_seqs[i]) for i in hard_idx)
    n_confound = sum(hard_kind_counts.get(k, 0) for k in ("multiple_markers", "length_insufficient", "no_marker", "empty_ux", "empty_uz"))
    n_genuine = hard_kind_counts.get("value_mismatch_only", 0)
    confound_check = {
        "n_hard": len(hard_idx),
        "hard_population_violation_kind_breakdown": dict(hard_kind_counts),
        "fraction_hard_that_is_trivial_confound (marker/length/empty checks)": n_confound / len(hard_idx),
        "fraction_hard_that_is_genuinely_value_mismatch_only": n_genuine / len(hard_idx),
        "confound_present_but_less_severe_than_prior_two_tasks": True,
        "interpretation": (
            "CONFIRMED but LESS SEVERE than missing-duplicate-string (87%) or stack-manipulation "
            "(87%): 76.9% of the nominally-'hard' population is a trivial structural shortcut "
            "(dominated by multiple_markers at 46.5%, not length as in the prior two tasks -- here "
            "the second '=' simply happens to occur late for this task's negative-generation "
            "distribution). The remaining 23.1% is genuinely value_mismatch_only. Unlike the prior "
            "two tasks, this leaves a SUBSTANTIAL genuinely-hard population (16.5% of ALL negatives, "
            "411/2487) -- large enough that a meaningful direct-accuracy check and Condition 3 test "
            "are both well-powered here, not just a token residual."
        ),
    }
    print(json.dumps(confound_check, indent=2, default=str))

    print("\n=== three candidate shortcut features: per-feature + joint audit (TEST split) ===", flush=True)
    per_feature, joint, shortcut_blind_neg_idx = audit_shortcut_features(test_seqs, test_labels)
    print(json.dumps(per_feature, indent=2, default=str))
    print(json.dumps(joint, indent=2, default=str))
    assert dict(Counter(classify_violation(test_seqs[i]) for i in shortcut_blind_neg_idx)) == {"value_mismatch_only": len(shortcut_blind_neg_idx)}

    print("\n=== load-bearing counterexample (x=9, correct z=3 vs wrong z=2, same shortcut features) ===", flush=True)
    counterexample = build_counterexample()
    print(json.dumps(counterexample, indent=2, default=str))
    assert counterexample["all_three_shortcut_features_identical_between_clean_and_corrupted"]
    assert counterexample["clean_violation"] == "ACTUALLY_POSITIVE" and counterexample["corrupted_violation"] == "value_mismatch_only"

    out = {
        "task": TASK,
        "description": (
            "Phase 3 (bounded) task audit for compute-sqrt. Confirms LSB-first binary encoding for "
            "both x and z (opposite of naive MSB-first assumption -- critical for Condition 3's "
            "'early vs late bit' position sweep: leftmost/early sequence positions are LOW-ORDER "
            "bits, rightmost/late positions are HIGH-ORDER bits). Confirms integer-floor-sqrt "
            "semantics (not exact-perfect-square). No mod-2 parity shortcut exists. Confound check: "
            "present but markedly less severe than missing-duplicate-string/stack-manipulation, "
            "leaving a substantial (16.5% of all negatives) genuinely-hard population."
        ),
        "grammar_confirmation": grammar,
        "cited_hard_negative_fraction_reproduction": {
            "test_set_n_negatives": len(test_neg_idx), "n_hard": len(hard_idx), "fraction": hard_fraction,
            "matches_a1_audit_cited_19_18_percent": abs(hard_fraction - 0.1918) < 0.005,
        },
        "hard_negative_confound_check": confound_check,
        "shortcut_feature_audit": {"per_feature": per_feature, "joint_all_three": joint},
        "load_bearing_counterexample": counterexample,
        "genuinely_hard_population_definition_for_later_conditions": (
            "marker_count_is_one AND fields_well_formed AND answer_length_sufficient AND "
            "value_mismatch_only (decoded z != true isqrt(x)) -- 411 real test-set examples, "
            "SUBSTANTIALLY larger than the prior two tasks' residuals."
        ),
        "bit_order_note_for_condition3_position_sweep": (
            "LSB-first encoding means 'early' sequence positions within u_z are LOW-ORDER (small "
            "magnitude) bits and 'late' positions are HIGH-ORDER (large magnitude) bits -- flipping "
            "a late/high-order bit changes z's VALUE by a much larger amount than flipping an early/"
            "low-order bit. Any Condition 3 gap that scales with position must be checked against "
            "this magnitude confound (per the user's explicit binary-addition-precedent concern) "
            "before being read as 'more comprehensive verification at high positions' -- it may just "
            "be magnitude sensitivity."
        ),
    }
    out_path = RESULTS / "phase3_computesqrt_task_audit.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
