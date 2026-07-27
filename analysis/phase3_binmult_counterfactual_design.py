"""Binary-multiplication Phase 3 (bounded) pilot: counterfactual pair
construction + audit for the three approved conditions. Mirrors the
compute-sqrt/stack-manipulation construction discipline: each pair type
isolates ONE candidate feature, audited explicitly, no patching here.

Uses phase3_binmult_task_audit.py's confirmed grammar (LSB-first binary for
u_x/u_y/u_z, x*y=z semantics, the bitlength(x)+bitlength(y)-1/+0 length
constraint, the EXACT LSB(z)=LSB(x) AND LSB(y) identity) and its five
shortcut features (operator_count_is_one, equals_count_is_one_after_
operator, fields_well_formed, length_sufficient, lsb_consistent), plus the
323-example value_mismatch_only shortcut-blind residual.

CONDITION 1 (structural: operator count, equals count, well-formedness,
TRUE null control):
  operator_count_pairs: clean=valid positive. corrupt=SAME LENGTH, an
    EARLY token (first bit of u_x) replaced by a second '×' -- operator_
    count 1->2.
  equals_count_pairs: clean=valid positive. corrupt=SAME LENGTH, first bit
    of u_y replaced by a second '=' -- equals_count 1->2 (placed inside
    u_y, still after the operator, so equals_count_is_one_after_operator's
    own "after operator" sub-check stays satisfied for the ORIGINAL equals
    but the COUNT check fails; audited explicitly, not assumed).
  fields_well_formed_pairs: clean=valid positive with n_y>=1. corrupt=SAME
    TOKEN MULTISET, rearranged so the operator is IMMEDIATELY followed by
    the equals (u_y moved to just after the equals instead) -- makes u_y
    EMPTY while operator_count and equals_count-after-operator both stay
    satisfied (exactly 1 each, equals still after operator).
  null_control_pairs (TRUE minimal pair, multiplication-specific math):
    exploits COMMUTATIVITY (x*y == y*x) -- for x != y with the SAME field
    width, clean = u_x(x) × u_y(y) = u_z, corrupt = u_x(y) × u_y(x) = u_z
    (operands swapped, IDENTICAL u_z). Since z is unchanged and x*y=y*x
    always holds, BOTH orderings are genuinely valid with the SAME true
    label -- tests whether models spuriously depend on WHICH position
    (first/second operand) holds which value, despite commutativity making
    that irrelevant.

CONDITION 2 (task-specific structural -- the two cheapest-possible
shortcuts, length_sufficient and lsb_consistent):
  length_insufficient_pairs: clean=valid positive with u_z built at
    EXACTLY the minimum sufficient width (n_z == bitlength(x*y), so the
    last/highest-order token of u_z is necessarily '1', the MSB).
    corrupt=that last token REMOVED (field one token shorter) -- breaks
    length_sufficient, x/y and all remaining u_z bits unchanged.
  lsb_flip_pairs: clean=valid positive. corrupt=u_z[0] (the LSB) flipped --
    breaks lsb_consistent directly (and, as an expected/documented
    coupling matching missing-duplicate-string/compute-sqrt's Condition 2
    caveat, this is ALSO necessarily a value_mismatch, since any single-
    bit flip on a fixed-width field changes the decoded value).

CONDITION 3 (target-computation null test on the value_mismatch_only
residual): clean=valid positive (x, y, correct z=x*y, all five shortcut
features hold). corrupt=SAME x, y, SAME field width n_z (so length_
sufficient is UNAFFECTED by construction), operator/equals counts
unchanged -- ONE bit of u_z flipped, producing z' != z. Swept LOW/MID/HIGH
tertile by flipped-bit index within u_z -- LSB-first: LOW tertile = low-
order/small-magnitude bit changes, HIGH tertile = high-order/large-
magnitude changes. INTERPRETIVE DISCIPLINE flagged explicitly per the
user's binary-addition/compute-sqrt precedent: gaps that scale with
tertile (high > low) indicate magnitude-sensitivity, NOT necessarily more
"comprehensive" verification; position-INDEPENDENT gaps indicate genuine
bit-exact verification (would be the first such finding in this pilot);
floor-effect (near-zero gap AND near-zero genuinely-hard-subset accuracy)
matches the missing-duplicate-string/stack-manipulation confounded-cluster
pattern.

PYTHONPATH=src:analysis python analysis/phase3_binmult_counterfactual_design.py
"""

import json
import math
from pathlib import Path

import numpy as np

RESULTS = Path("analysis_outputs/final_results")
OPERATOR, EQUALS = "×", "="
TERTILES = ["low", "mid", "high"]
MAX_TRIES = 2000


def categorize_tertile(frac):
    if frac < 1 / 3:
        return "low"
    elif frac < 2 / 3:
        return "mid"
    else:
        return "high"


# ---------------------------------------------------------------------------
# shared construction helpers
# ---------------------------------------------------------------------------

def encode_lsb_first(value, width):
    bits = []
    v = value
    for _ in range(width):
        bits.append("1" if (v & 1) else "0")
        v >>= 1
    return bits


def decode_binary(bits):
    x = 0
    for i, b in enumerate(bits):
        if b == "1":
            x |= (1 << i)
    return x


def bitlength(v):
    return v.bit_length() if v > 0 else 1


def build_valid_positive(rng, n_x_range=(3, 10), n_y_range=(3, 10), slack_range=(0, 4)):
    n_x = int(rng.integers(n_x_range[0], n_x_range[1] + 1))
    n_y = int(rng.integers(n_y_range[0], n_y_range[1] + 1))
    x = int(rng.integers(0, 2 ** n_x))
    y = int(rng.integers(0, 2 ** n_y))
    z = x * y
    need = bitlength(z)
    slack = int(rng.integers(slack_range[0], slack_range[1] + 1))
    n_z = need + slack
    u_x, u_y = encode_lsb_first(x, n_x), encode_lsb_first(y, n_y)
    u_z = encode_lsb_first(z, n_z)
    tokens = u_x + [OPERATOR] + u_y + [EQUALS] + u_z
    return {
        "tokens": tokens, "x": x, "y": y, "z": z, "n_x": n_x, "n_y": n_y, "n_z": n_z,
        "op_idx": n_x, "eq_idx": n_x + 1 + n_y,
    }


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


def is_positive_local(seq):
    parts = _parse(seq)
    if parts is None:
        return False
    u_x, u_y, u_z = parts
    x, y, z = decode_binary(u_x), decode_binary(u_y), decode_binary(u_z)
    return x * y == z


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
    return len(u_z) >= bitlength(x * y)


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


# ---------------------------------------------------------------------------
# condition 1: structural features
# ---------------------------------------------------------------------------

def operator_count_pairs(n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(rng)
        clean = b["tokens"]
        corrupt = clean[:]
        corrupt[0] = OPERATOR
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "clean_op_count": clean.count(OPERATOR), "corrupt_op_count": corrupt.count(OPERATOR),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} operator_count pairs")
    return pairs


def audit_operator_count_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["clean_all_count_1"] = all(p["clean_op_count"] == 1 for p in pairs)
    a["corrupt_all_count_2"] = all(p["corrupt_op_count"] == 2 for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and
        a["clean_all_count_1"] and a["corrupt_all_count_2"]
    )
    return a


def equals_count_pairs(n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(rng)
        clean = b["tokens"]
        corrupt = clean[:]
        corrupt[b["op_idx"] + 1] = EQUALS  # first bit of u_y -> extra '='
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "clean_eq_count": clean.count(EQUALS), "corrupt_eq_count": corrupt.count(EQUALS),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} equals_count pairs")
    return pairs


def audit_equals_count_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["clean_all_count_1"] = all(p["clean_eq_count"] == 1 for p in pairs)
    a["corrupt_all_count_2"] = all(p["corrupt_eq_count"] == 2 for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and
        a["clean_all_count_1"] and a["corrupt_all_count_2"]
    )
    return a


def fields_well_formed_pairs(n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(rng, n_y_range=(1, 10))
        clean = b["tokens"]
        u_x = clean[:b["op_idx"]]
        u_y = clean[b["op_idx"] + 1:b["eq_idx"]]
        u_z = clean[b["eq_idx"] + 1:]
        corrupt = u_x + [OPERATOR, EQUALS] + u_y + u_z  # u_y moved past equals -- now empty
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "clean_op_count": clean.count(OPERATOR), "corrupt_op_count": corrupt.count(OPERATOR),
            "clean_eq_count": clean.count(EQUALS), "corrupt_eq_count": corrupt.count(EQUALS),
            "clean_well_formed": fields_well_formed(clean), "corrupt_well_formed": fields_well_formed(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} fields_well_formed pairs")
    return pairs


def audit_fields_well_formed_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    a["same_op_count"] = all(p["clean_op_count"] == p["corrupt_op_count"] == 1 for p in pairs)
    a["same_eq_count"] = all(p["clean_eq_count"] == p["corrupt_eq_count"] == 1 for p in pairs)
    a["clean_all_well_formed"] = all(p["clean_well_formed"] for p in pairs)
    a["corrupt_all_malformed"] = all(not p["corrupt_well_formed"] for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["same_op_count"] and a["same_eq_count"] and
        a["clean_all_well_formed"] and a["corrupt_all_malformed"]
    )
    return a


def null_control_pairs(n_pairs, rng, n_range=(3, 10), max_tries=MAX_TRIES):
    """TRUE minimal pair via COMMUTATIVITY (x*y == y*x): same field width
    for both operands, x != y -- clean=x×y=z, corrupt=y×x=z (SAME u_z,
    operands swapped). Both genuinely valid, same true label."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * max_tries:
        attempts += 1
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        x = int(rng.integers(0, 2 ** n))
        y = int(rng.integers(0, 2 ** n))
        if x == y:
            continue
        z = x * y
        need = bitlength(z)
        slack = int(rng.integers(0, 4))
        n_z = need + slack
        u_x, u_y, u_z = encode_lsb_first(x, n), encode_lsb_first(y, n), encode_lsb_first(z, n_z)
        clean = u_x + [OPERATOR] + u_y + [EQUALS] + u_z
        corrupt = u_y + [OPERATOR] + u_x + [EQUALS] + u_z
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean), "x": x, "y": y,
            "clean_is_positive": is_positive_local(clean), "corrupt_is_positive": is_positive_local(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} null_control pairs")
    return pairs


def audit_null_control_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    a["operands_actually_differ"] = all(p["x"] != p["y"] for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["n_positions_differing"] = {"min": min(diffs), "max": max(diffs)}
    a["both_genuinely_valid_same_label"] = all(p["clean_is_positive"] and p["corrupt_is_positive"] for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["operands_actually_differ"] and a["both_genuinely_valid_same_label"]
    )
    a["note"] = (
        "TRUE minimal pair via commutativity (x*y == y*x): u_z is IDENTICAL between clean and "
        "corrupt, only the operand ORDER differs. Both genuinely valid with the SAME true label -- "
        "any observed gap here would indicate spurious operand-position dependence, not a genuine "
        "computation difference."
    )
    return a


# ---------------------------------------------------------------------------
# condition 2: task-specific structural features (the two cheapest shortcuts)
# ---------------------------------------------------------------------------

def length_insufficient_pairs(n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(rng, slack_range=(0, 0))  # n_z == bitlength(z) exactly
        if b["z"] == 0:
            continue  # z=0's minimal-width representation is a single '0' bit, not '1' -- skip
        clean = b["tokens"]
        assert clean[-1] == "1", "MSB of a minimal-width z representation must be 1"
        corrupt = clean[:-1]
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt,
            "clean_length": len(clean), "corrupt_length": len(corrupt),
            "clean_length_sufficient": length_sufficient(clean),
            "corrupt_length_sufficient": length_sufficient(corrupt),
            "clean_op_count": clean.count(OPERATOR), "corrupt_op_count": corrupt.count(OPERATOR),
            "clean_eq_count": clean.count(EQUALS), "corrupt_eq_count": corrupt.count(EQUALS),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} length_insufficient pairs")
    return pairs


def audit_length_insufficient_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["corrupt_is_one_token_shorter"] = all(p["corrupt_length"] == p["clean_length"] - 1 for p in pairs)
    a["same_op_eq_counts"] = all(
        p["clean_op_count"] == p["corrupt_op_count"] == 1 and p["clean_eq_count"] == p["corrupt_eq_count"] == 1
        for p in pairs)
    a["clean_all_length_sufficient"] = all(p["clean_length_sufficient"] for p in pairs)
    a["corrupt_all_length_insufficient"] = all(not p["corrupt_length_sufficient"] for p in pairs)
    a["prefix_unchanged"] = all(p["clean"][:p["corrupt_length"]] == p["corrupt"] for p in pairs)
    a["target_property_isolated"] = (
        a["corrupt_is_one_token_shorter"] and a["same_op_eq_counts"] and a["clean_all_length_sufficient"] and
        a["corrupt_all_length_insufficient"] and a["prefix_unchanged"]
    )
    return a


def lsb_flip_pairs(n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(rng, slack_range=(0, 4))
        clean = b["tokens"]
        eq_i = b["eq_idx"]
        corrupt = clean[:]
        corrupt[eq_i + 1] = "0" if corrupt[eq_i + 1] == "1" else "1"  # flip u_z[0]
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "clean_lsb_consistent": lsb_consistent(clean), "corrupt_lsb_consistent": lsb_consistent(corrupt),
            "clean_op_count": clean.count(OPERATOR), "corrupt_op_count": corrupt.count(OPERATOR),
            "clean_eq_count": clean.count(EQUALS), "corrupt_eq_count": corrupt.count(EQUALS),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} lsb_flip pairs")
    return pairs


def audit_lsb_flip_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["same_op_eq_counts"] = all(
        p["clean_op_count"] == p["corrupt_op_count"] == 1 and p["clean_eq_count"] == p["corrupt_eq_count"] == 1
        for p in pairs)
    a["clean_all_lsb_consistent"] = all(p["clean_lsb_consistent"] for p in pairs)
    a["corrupt_all_lsb_inconsistent"] = all(not p["corrupt_lsb_consistent"] for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and a["same_op_eq_counts"] and
        a["clean_all_lsb_consistent"] and a["corrupt_all_lsb_inconsistent"]
    )
    a["coupling_note"] = (
        "Flipping u_z[0] necessarily ALSO makes the pair a value_mismatch (any single-bit flip on a "
        "fixed-width field changes the decoded value) -- documented, not hidden, matching missing-"
        "duplicate-string/compute-sqrt's Condition 2 coupling caveat."
    )
    return a


# ---------------------------------------------------------------------------
# condition 3: target-computation null test, swept low/mid/high bit position
# ---------------------------------------------------------------------------

def target_computation_pairs(n_pairs, rng, n_z_range=(9, 15), fixed_tertile=None, max_tries=MAX_TRIES):
    pairs, seen, tries_total = [], set(), 0
    while len(pairs) < n_pairs and tries_total < n_pairs * max_tries:
        target_tertile = fixed_tertile if fixed_tertile is not None else TERTILES[int(rng.integers(0, 3))]
        found = False
        for _ in range(max_tries):
            tries_total += 1
            n_z = int(rng.integers(n_z_range[0], n_z_range[1] + 1))
            b = build_valid_positive(rng, n_x_range=(3, 10), n_y_range=(3, 10), slack_range=(0, 0))
            if b["n_z"] > n_z:
                continue
            x, y, z = b["x"], b["y"], b["z"]
            u_x = b["tokens"][:b["op_idx"]]
            u_y = b["tokens"][b["op_idx"] + 1:b["eq_idx"]]
            u_z = encode_lsb_first(z, n_z)
            clean = u_x + [OPERATOR] + u_y + [EQUALS] + u_z
            op_idx, eq_idx = b["op_idx"], len(u_x) + 1 + len(u_y)
            # bit_idx==0 (the LSB) is excluded: flipping it always breaks
            # lsb_consistent (one of the five shortcut features that must
            # stay IDENTICAL between clean/corrupt here) -- verified
            # empirically to trip the "all five features hold" assertion.
            candidates = list(range(1, n_z))
            denom = max(1, n_z - 1)
            def frac_of(j):
                return j / denom
            bucket = [j for j in candidates if categorize_tertile(frac_of(j)) == target_tertile]
            if not bucket:
                continue
            bit_idx = int(rng.choice(bucket))
            corrupt = clean[:]
            flip_pos = eq_idx + 1 + bit_idx
            corrupt[flip_pos] = "0" if corrupt[flip_pos] == "1" else "1"
            z_corrupt = decode_binary(corrupt[eq_idx + 1:])
            key = (tuple(clean), tuple(corrupt))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "clean": clean, "corrupt": corrupt, "length": len(clean),
                "x": x, "y": y, "z_clean": z, "z_corrupt": z_corrupt, "n_z": n_z,
                "flipped_bit_idx": bit_idx, "flip_position_relative": frac_of(bit_idx),
                "flip_position_tertile": target_tertile,
                "clean_all_five_features": all(fn(clean) for fn in SHORTCUT_FEATURES.values()),
                "corrupt_all_five_features": all(fn(corrupt) for fn in SHORTCUT_FEATURES.values()),
                "clean_is_positive": is_positive_local(clean), "corrupt_is_positive": is_positive_local(corrupt),
            })
            found = True
            break
        if not found:
            continue
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} target_computation pairs")
    return pairs


def audit_target_computation_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["clean_all_five_shortcut_features_hold"] = all(p["clean_all_five_features"] for p in pairs)
    a["corrupt_all_five_shortcut_features_ALSO_hold"] = all(p["corrupt_all_five_features"] for p in pairs)
    a["z_corrupt_always_differs_from_z_clean"] = all(p["z_corrupt"] != p["z_clean"] for p in pairs)
    a["clean_all_positive_corrupt_all_negative"] = all(
        p["clean_is_positive"] and not p["corrupt_is_positive"] for p in pairs)
    tertile_hist = {t: sum(1 for p in pairs if p["flip_position_tertile"] == t) for t in TERTILES}
    a["flip_position_tertile_histogram"] = tertile_hist
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and
        a["clean_all_five_shortcut_features_hold"] and a["corrupt_all_five_shortcut_features_ALSO_hold"] and
        a["z_corrupt_always_differs_from_z_clean"] and a["clean_all_positive_corrupt_all_negative"]
    )
    a["note"] = (
        "All five shortcut features (operator/equals counts, well-formedness, length-sufficiency, "
        "lsb-consistency) are IDENTICAL between clean and corrupt for every pair -- structurally "
        "guaranteed since only one u_z bit is touched, x/y and field widths are untouched. LSB-"
        "first encoding: 'low' tertile = low-order/small-magnitude bit flips, 'high' tertile = "
        "high-order/large-magnitude flips."
    )
    return a


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260718)

    ocp = operator_count_pairs(30, rng)
    ecp = equals_count_pairs(30, rng)
    fwp = fields_well_formed_pairs(30, rng)
    ncp = null_control_pairs(30, rng)
    lip = length_insufficient_pairs(30, rng)
    lfp = lsb_flip_pairs(30, rng)
    tcp = target_computation_pairs(30, rng)

    audits = {
        "operator_count_pairs": audit_operator_count_pairs(ocp),
        "equals_count_pairs": audit_equals_count_pairs(ecp),
        "fields_well_formed_pairs": audit_fields_well_formed_pairs(fwp),
        "null_control_pairs": audit_null_control_pairs(ncp),
        "length_insufficient_pairs": audit_length_insufficient_pairs(lip),
        "lsb_flip_pairs": audit_lsb_flip_pairs(lfp),
        "target_computation_pairs": audit_target_computation_pairs(tcp),
    }
    for name, a in audits.items():
        print(f"=== {name} ===", flush=True)
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    out = {
        "task": "binary-multiplication",
        "description": (
            "Phase 3 counterfactual pair construction + audit for binary-multiplication causal "
            "patching, all three approved conditions. NO PATCHING RUN. null_control_pairs is a TRUE "
            "minimal pair built from the start via COMMUTATIVITY (x*y == y*x, multiplication-"
            "specific math, per the compute-sqrt precedent). Condition 3 targets the 323-example "
            "value_mismatch_only residual, swept by LSB-first bit-position tertile."
        ),
        "condition_1_structural": {
            "operator_count_pairs": {"audit": audits["operator_count_pairs"], "sample_pairs": ocp[:3]},
            "equals_count_pairs": {"audit": audits["equals_count_pairs"], "sample_pairs": ecp[:3]},
            "fields_well_formed_pairs": {"audit": audits["fields_well_formed_pairs"], "sample_pairs": fwp[:3]},
            "null_control_pairs_TRUE_MINIMAL_PAIR_COMMUTATIVITY": {"audit": audits["null_control_pairs"], "sample_pairs": ncp[:3]},
        },
        "condition_2_task_specific_structural_cheapest_shortcuts": {
            "length_insufficient_pairs": {"audit": audits["length_insufficient_pairs"], "sample_pairs": lip[:3]},
            "lsb_flip_pairs": {"audit": audits["lsb_flip_pairs"], "sample_pairs": lfp[:3]},
        },
        "condition_3_target_computation_null_test_value_mismatch_residual": {
            "target_computation_pairs": {"audit": audits["target_computation_pairs"], "sample_pairs": tcp[:3]},
        },
    }
    out_path = RESULTS / "phase3_binmult_counterfactual_design.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
