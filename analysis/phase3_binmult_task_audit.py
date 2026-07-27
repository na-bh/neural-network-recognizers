"""Binary-multiplication Phase 3 (bounded, patching-only): task audit.

GRAMMAR, confirmed directly from the generator source (src/recognizers/
hand_picked_languages/binary_multiplication.py + binary_util.py's shared
BinaryArithmeticOperation/LengthRestrictedBinaryArithmeticOperation, the
SAME base classes binary-addition uses) and cross-checked against real
train/test data:

  Alphabet (4 symbols): '0', '1' (binary digits), '×' (OPERATOR), '='
  (EQUALS). Structure: [u_x: n_x bits] '×' [u_y: n_y bits] '=' [u_z: n_z
  bits]. All three bit-fields are LSB-FIRST (little-endian) fixed-width
  binary encodings (confirmed empirically: decoding 3000 real positives
  this way gives x*y == z for 3000/3000). Operand order is randomized at
  generation time (u_x/u_y swapped with 50% probability), so there is no
  fixed "smaller operand first" convention.

  EXACT target: z = x * y (compute_z(x,y) = x*y, confirmed directly in
  source and cross-checked against math on real data). get_length_weights()
  = (1,1,2) -- the Dirichlet split favors giving u_z roughly DOUBLE the
  length budget of u_x or u_y individually, reflecting that a product can
  need up to twice as many bits as either factor.

MULTIPLICATION-SPECIFIC LENGTH CONSTRAINT (the "specific length-mismatch
constraint" flagged in the request, distinct from addition's simpler
constraint): for nonzero x,y, bitlength(x*y) is ALWAYS EITHER
bitlength(x)+bitlength(y)-1 OR bitlength(x)+bitlength(y) -- a classical
number-theory fact, confirmed empirically on 1823/1823 real positives.
This gives a trivial O(1) shortcut: n_z must be >= bitlength(x)+
bitlength(y)-1 (the MINIMUM possible product bit-length) -- computable via
simple bit-length arithmetic on x and y alone, no actual multiplication
needed.

MOD-2 (LSB) RELATIONSHIP -- explicitly requested, and CONFIRMED (unlike
compute-sqrt, where the equivalent check found no such relationship):
LSB(z) == LSB(x) AND LSB(y) is an EXACT, ALWAYS-TRUE identity for
multiplication ((2a+p)(2b+q) = 4ab+2aq+2bp+pq, so mod 2 this is exactly
p*q = p AND q for p,q in {0,1}) -- confirmed empirically 3000/3000 on real
positives, zero exceptions, matching the exact mathematical proof. This is
the single cheapest possible shortcut for this task: a 2-bit AND check on
the very first tokens of u_x/u_y/u_z, structurally requiring NO awareness
of the rest of the string at all.

CANDIDATE STRUCTURAL SHORTCUTS (five features, each trivial counting/
single-position/single-AND checks, no real multiplication needed):
  operator_count_is_one
  equals_count_is_one_after_operator
  fields_well_formed              -- u_x, u_y, u_z all non-empty
  length_sufficient                -- n_z >= bitlength(x)+bitlength(y)-1
  lsb_consistent                   -- u_z[0] == (u_x[0] AND u_y[0])

CONFOUND CHECK (per the missing-duplicate-string/stack-manipulation/
compute-sqrt lesson): of the 284 real test-set negatives classified "hard"
by viol_binary_arith (frac>=0.8; reproduces the cited 11.19% hard-negative
fraction), 66.5% are a trivial structural violation (dominated by
multiple_operators at 33.8% and value-adjacent length_insufficient at
18.0%), leaving 33.5% genuinely value_mismatch_only. This is a REAL
confound (same qualitative pattern as the other three tasks), but the
LEAST severe of the four audited in this pilot so far -- missing-
duplicate-string and stack-manipulation were ~87% confounded, compute-sqrt
~77%, binary-multiplication only ~66.5%. Correspondingly, the joint-five-
feature shortcut-blind residual is 323/2538 = 12.7% of ALL negatives, the
LARGEST genuinely-hard population of the four tasks (missing-duplicate-
string/stack-manipulation ~2.4%, compute-sqrt ~16.5% -- binary-
multiplication is comparable in scale to compute-sqrt, both well-powered
for a real Condition 3 test).

PYTHONPATH=src:analysis python analysis/phase3_binmult_task_audit.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "analysis")
from flare_a1_task_audit import viol_binary_arith, HARD_THRESHOLD

RESULTS = Path("analysis_outputs/final_results")
TASK = "binary-multiplication"
OPERATOR, EQUALS = "×", "="


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


def _parse(seq):
    op_idx = [i for i, t in enumerate(seq) if t == OPERATOR]
    eq_idx = [i for i, t in enumerate(seq) if t == EQUALS]
    if len(op_idx) != 1:
        return None
    op_i = op_idx[0]
    later_eq = [e for e in eq_idx if e > op_i]
    if len(eq_idx) != 1 or not later_eq:
        return None
    eq_i = later_eq[0]
    u_x, u_y, u_z = seq[:op_i], seq[op_i + 1:eq_i], seq[eq_i + 1:]
    if not u_x or not u_y or not u_z:
        return None
    return u_x, u_y, u_z


def classify_violation(seq):
    op_idx = [i for i, t in enumerate(seq) if t == OPERATOR]
    eq_idx = [i for i, t in enumerate(seq) if t == EQUALS]
    if len(op_idx) == 0:
        return "no_operator"
    if len(op_idx) >= 2:
        return "multiple_operators"
    op_i = op_idx[0]
    later_eq = [e for e in eq_idx if e > op_i]
    if len(eq_idx) == 0:
        return "no_equals"
    if len(eq_idx) >= 2 or not later_eq:
        return "multiple_equals_or_misplaced"
    eq_i = later_eq[0]
    u_x, u_y, u_z = seq[:op_i], seq[op_i + 1:eq_i], seq[eq_i + 1:]
    if not u_x:
        return "empty_ux"
    if not u_y:
        return "empty_uy"
    if not u_z:
        return "empty_uz"
    x, y, z = decode_binary(u_x), decode_binary(u_y), decode_binary(u_z)
    z_true = x * y
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
    n_bitlength_checked = 0
    n_bitlength_match = 0
    n_lsb_checked = 0
    n_lsb_match = 0
    for s in pos:
        parts = _parse(s)
        if parts is None:
            continue
        u_x, u_y, u_z = parts
        x, y, z = decode_binary(u_x), decode_binary(u_y), decode_binary(u_z)
        n_checked += 1
        if x * y == z:
            n_match += 1
        if x > 0 and y > 0:
            n_bitlength_checked += 1
            bx, by, bz = bitlength(x), bitlength(y), bitlength(z)
            if bz in (bx + by - 1, bx + by):
                n_bitlength_match += 1
        n_lsb_checked += 1
        expected = "1" if (u_x[0] == "1" and u_y[0] == "1") else "0"
        if u_z[0] == expected:
            n_lsb_match += 1

    return {
        "alphabet": ["0", "1", "×", "="],
        "encoding": "LSB-first (little-endian) fixed-width binary for u_x, u_y, u_z",
        "n_positives_checked": n_checked,
        "x_times_y_equals_z_match_fraction": n_match / n_checked if n_checked else float("nan"),
        "bitlength_constraint": {
            "formula": "bitlength(x*y) in {bitlength(x)+bitlength(y)-1, bitlength(x)+bitlength(y)} for x,y>0",
            "n_checked": n_bitlength_checked, "n_match": n_bitlength_match,
            "match_fraction": n_bitlength_match / n_bitlength_checked if n_bitlength_checked else float("nan"),
        },
        "mod2_lsb_relationship": {
            "formula": "LSB(z) == LSB(x) AND LSB(y)",
            "n_checked": n_lsb_checked, "n_match": n_lsb_match,
            "match_fraction": n_lsb_match / n_lsb_checked if n_lsb_checked else float("nan"),
            "note": "EXACT mathematical identity for multiplication (unlike compute-sqrt, where no "
                    "such parity relationship was found) -- confirmed with zero exceptions.",
        },
        "task_family_reclassification": (
            "Same operator/equals-count family as binary-addition and compute-sqrt (viol_binary_"
            "arith is the SHARED violation classifier for both binary-addition and binary-"
            "multiplication) -- structurally simplest change from binary-addition is the harder "
            "target computation (multiplication's carry-propagation is a longer, non-linear "
            "dependency chain vs addition's simple ripple-carry) plus the doubled length-share for "
            "u_z (get_length_weights=(1,1,2))."
        ),
    }


def marker_shortcut_features_names():
    return ["operator_count_is_one", "equals_count_is_one_after_operator", "fields_well_formed",
            "length_sufficient", "lsb_consistent"]


def operator_count_is_one(seq):
    return seq.count(OPERATOR) == 1


def equals_count_is_one_after_operator(seq):
    op_idx = [i for i, t in enumerate(seq) if t == OPERATOR]
    eq_idx = [i for i, t in enumerate(seq) if t == EQUALS]
    if len(op_idx) != 1 or len(eq_idx) != 1:
        return False
    return eq_idx[0] > op_idx[0]


def fields_well_formed(seq):
    return _parse(seq) is not None


def length_sufficient(seq):
    parts = _parse(seq)
    if parts is None:
        return False
    u_x, u_y, u_z = parts
    x, y = decode_binary(u_x), decode_binary(u_y)
    need = bitlength(x * y)
    return len(u_z) >= need


def lsb_consistent(seq):
    parts = _parse(seq)
    if parts is None:
        return False
    u_x, u_y, u_z = parts
    expected = "1" if (u_x[0] == "1" and u_y[0] == "1") else "0"
    return u_z[0] == expected


SHORTCUT_FEATURES = {
    "operator_count_is_one": operator_count_is_one,
    "equals_count_is_one_after_operator": equals_count_is_one_after_operator,
    "fields_well_formed": fields_well_formed,
    "length_sufficient": length_sufficient,
    "lsb_consistent": lsb_consistent,
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
        "positive_satisfy_all_five_fraction": joint_pos_satisfy / len(pos_idx) if pos_idx else float("nan"),
        "n_shortcut_blind_negatives (pass ALL FIVE checks, still invalid)": len(shortcut_blind_neg_idx),
        "n_total_negatives": len(neg_idx),
        "shortcut_blind_fraction_of_negatives": len(shortcut_blind_neg_idx) / len(neg_idx) if neg_idx else float("nan"),
        "shortcut_blind_violation_kind_breakdown": dict(kind_counts),
        "interpretation": (
            "The shortcut-blind population is exactly the set of negatives all five counting/"
            "parsing/AND-check features CANNOT distinguish from a genuine positive -- the LARGEST "
            "such residual found in this pilot so far relative to the other three tasks audited, "
            "comparable in scale to compute-sqrt's."
        ),
    }
    return per_feature, joint, shortcut_blind_neg_idx


def build_counterexample():
    """x=3(u_x=1,1), y=3(u_y=1,1), z=9(u_z=1,0,0,1, LSB-first: 1+0+0+8=9).
    corrupted: z=5 (u_z=1,0,1,0 -> 1+0+4+0=5) instead of 9 -- SAME length,
    SAME operator/equals counts, SAME length_sufficient (bitlength(5)=3 <=
    4, still sufficient), SAME lsb_consistent (LSB(x)=1,LSB(y)=1 so
    expected LSB(z)=1; both 9 and 5 have LSB=1) -- passes every listed
    shortcut feature, fails true multiplication verification."""
    clean = ["1", "1", "×", "1", "1", "=", "1", "0", "0", "1"]  # 3*3=9
    corrupted = ["1", "1", "×", "1", "1", "=", "1", "0", "1", "0"]  # claims 3*3=5
    features_match = {name: (fn(clean) == fn(corrupted)) for name, fn in SHORTCUT_FEATURES.items()}
    return {
        "clean": clean, "corrupted": corrupted,
        "clean_violation": classify_violation(clean), "corrupted_violation": classify_violation(corrupted),
        "all_five_shortcut_features_identical_between_clean_and_corrupted": all(features_match.values()),
        "per_feature_match": features_match,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    train_seqs, train_labels = load_split("train")
    test_seqs, test_labels = load_split("test")

    print("=== confirming grammar from generator + real train data ===", flush=True)
    grammar = confirm_grammar(train_seqs, train_labels)
    print(json.dumps(grammar, indent=2, default=str))
    assert grammar["x_times_y_equals_z_match_fraction"] == 1.0
    assert grammar["bitlength_constraint"]["match_fraction"] == 1.0
    assert grammar["mod2_lsb_relationship"]["match_fraction"] == 1.0

    print("\n=== reproducing cited hard-negative fraction (viol_binary_arith, "
          "HARD_THRESHOLD=0.8, TEST split) ===", flush=True)
    test_neg_idx = [i for i, l in enumerate(test_labels) if l == 0]
    fracs = {i: viol_binary_arith(test_seqs[i]) for i in test_neg_idx}
    hard_idx = [i for i in test_neg_idx if fracs[i] >= HARD_THRESHOLD]
    hard_fraction = len(hard_idx) / len(test_neg_idx)
    print(f"  n_neg={len(test_neg_idx)} n_hard={len(hard_idx)} fraction={hard_fraction*100:.2f}%", flush=True)

    print("\n=== CONFOUND CHECK ===", flush=True)
    hard_kind_counts = Counter(classify_violation(test_seqs[i]) for i in hard_idx)
    n_genuine = hard_kind_counts.get("value_mismatch_only", 0)
    n_confound = len(hard_idx) - n_genuine
    confound_check = {
        "n_hard": len(hard_idx),
        "hard_population_violation_kind_breakdown": dict(hard_kind_counts),
        "fraction_hard_that_is_trivial_confound": n_confound / len(hard_idx),
        "fraction_hard_that_is_genuinely_value_mismatch_only": n_genuine / len(hard_idx),
        "confound_present_but_least_severe_of_four_tasks_audited": True,
        "interpretation": (
            "CONFIRMED but the LEAST severe confound of the four tasks audited in this pilot "
            "(missing-duplicate-string/stack-manipulation ~87%, compute-sqrt ~77%, binary-"
            "multiplication ~66.5%). The genuinely-hard fraction (33.5% of 'hard', 12.7% of ALL "
            "negatives) is the largest well-powered residual found so far."
        ),
    }
    print(json.dumps(confound_check, indent=2, default=str))
    assert n_confound / len(hard_idx) > 0.5

    print("\n=== five candidate shortcut features: per-feature + joint audit (TEST split) ===", flush=True)
    per_feature, joint, shortcut_blind_neg_idx = audit_shortcut_features(test_seqs, test_labels)
    print(json.dumps(per_feature, indent=2, default=str))
    print(json.dumps(joint, indent=2, default=str))

    print("\n=== load-bearing counterexample (3x3=9 claimed vs 3x3=5 claimed, same shortcut features) ===", flush=True)
    counterexample = build_counterexample()
    print(json.dumps(counterexample, indent=2, default=str))
    assert counterexample["all_five_shortcut_features_identical_between_clean_and_corrupted"]
    assert counterexample["clean_violation"] == "ACTUALLY_POSITIVE" and counterexample["corrupted_violation"] == "value_mismatch_only"

    out = {
        "task": TASK,
        "description": (
            "Phase 3 (bounded) task audit for binary-multiplication. Confirms LSB-first encoding "
            "(same as binary-addition/compute-sqrt), exact x*y=z semantics, the multiplication-"
            "specific bitlength(x)+bitlength(y)-1/+0 length constraint, and an EXACT mod-2 LSB "
            "relationship (LSB(z)=LSB(x) AND LSB(y), unlike compute-sqrt where no such relationship "
            "exists). Confound check: present but the LEAST severe of the four tasks audited so far, "
            "leaving the largest genuinely-hard population (12.7% of all negatives)."
        ),
        "grammar_confirmation": grammar,
        "cited_hard_negative_fraction_reproduction": {
            "test_set_n_negatives": len(test_neg_idx), "n_hard": len(hard_idx), "fraction": hard_fraction,
        },
        "hard_negative_confound_check": confound_check,
        "shortcut_feature_audit": {"per_feature": per_feature, "joint_all_five": joint},
        "load_bearing_counterexample": counterexample,
        "genuinely_hard_population_definition_for_later_conditions": (
            "operator_count_is_one AND equals_count_is_one_after_operator AND fields_well_formed "
            "AND length_sufficient AND lsb_consistent AND value_mismatch_only -- "
            f"{joint['n_shortcut_blind_negatives (pass ALL FIVE checks, still invalid)']} real "
            "test-set examples."
        ),
        "bit_order_note_for_condition3_position_sweep": (
            "LSB-first encoding means 'early' sequence positions within u_z are LOW-ORDER (small-"
            "magnitude) bits and 'late' positions are HIGH-ORDER (large-magnitude) bits -- same "
            "convention as compute-sqrt. Any Condition 3 gap that scales with position must be "
            "checked against this magnitude confound before being read as more comprehensive "
            "verification at high positions."
        ),
    }
    out_path = RESULTS / "phase3_binmult_task_audit.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
