"""Compute-sqrt Phase 3 (bounded) pilot: counterfactual pair construction +
audit for the three approved conditions. Mirrors the missing-duplicate-
string/stack-manipulation construction discipline: each pair type isolates
ONE candidate feature, audited explicitly, no patching here.

Uses phase3_computesqrt_task_audit.py's confirmed grammar (LSB-first binary
for both u_x and u_z, EQUALS marker, floor-sqrt semantics, NO mod-2
shortcut) and its three shortcut features (marker_count_is_one,
fields_well_formed, answer_length_sufficient), plus the 411-example
value_mismatch_only shortcut-blind residual.

CONDITION 1 (structural: marker count, marker position, well-formedness,
TRUE null control):
  marker_count_pairs: clean=valid positive. corrupt=SAME LENGTH, an EARLY
    token (the first bit of u_x) replaced by a second '=' -- marker_count
    1->2, EARLY/trivially-detectable placement (contrasts with Condition
    2's LATE placement probe).
  no_marker_pairs: clean=valid positive. corrupt=SAME LENGTH, the single
    '=' replaced by a bit -- marker_count 1->0 (no_marker malformation),
    the "well-formedness" (marker must be present) test.
  marker_position_pairs: clean=valid positive with marker in the LOW
    length-tertile. corrupt=INDEPENDENTLY RANDOM binary content of the
    SAME LENGTH with a single '=' forced into the HIGH tertile -- mirrors
    bucket-sort/missing-duplicate-string's marker_position_pairs recipe.
  null_control_pairs (TRUE minimal pair from the start): exploits compute-
    sqrt's many-to-one x->z structure directly -- for any z, EVERY x in
    [z^2, (z+1)^2-1] shares the same correct answer. Finds two such x
    values differing in EXACTLY ONE BIT (both still in-range for the SAME
    z), builds both full sequences with IDENTICAL u_z -- both genuinely
    valid, same true label, differing at exactly ONE token position (a
    low-order x bit that happens not to change which sqrt-bucket x falls
    in). Cleaner than the prior two tasks' 2-position mirrored-flip null
    controls -- a true 1-position minimal pair, directly justified by the
    task's own mathematical redundancy rather than an engineered mirror.

CONDITION 2 (task-specific structural -- answer_length_sufficient, LATE
multiple_markers probe):
  answer_length_insufficient_pairs: clean=valid positive with u_z built at
    EXACTLY the minimum sufficient width (n_z == bitlength(isqrt(x)), so
    the last/highest-order token of u_z is necessarily '1', the MSB).
    corrupt=that last token REMOVED (field one token shorter) -- breaks
    answer_length_sufficient (now insufficient), x and all remaining u_z
    bits unchanged.
  late_second_marker_pairs: clean=valid positive. corrupt=SAME LENGTH, the
    LAST token (highest-order bit of u_z) replaced by a second '=' --
    marker_count 1->2, but placed LATE (mirrors the REAL hard-negative
    distribution's own pattern: 46.5% of the nominally-"hard" population
    is multiple_markers where the 2nd '=' happens to occur late). Directly
    contrasts with Condition 1's EARLY marker_count_pairs.

CONDITION 3 (target-computation null test on the value_mismatch_only
residual): clean=valid positive (x, correct z=isqrt(x), all three
shortcut features hold). corrupt=SAME x, SAME field width n_z (so
answer_length_sufficient is UNAFFECTED by construction -- verified, not
assumed), marker count/position unchanged -- ONE bit of u_z flipped,
producing z' != z (a single-bit flip on a fixed-width binary field always
changes the decoded integer, so z' is guaranteed wrong). Swept
LOW/MID/HIGH tertile by flipped-bit index within u_z -- since encoding is
LSB-first (confirmed in the task audit), LOW-tertile flips are LOW-ORDER
(small-magnitude) bit changes and HIGH-tertile flips are HIGH-ORDER
(large-magnitude) bit changes. INTERPRETIVE DISCIPLINE flagged explicitly
per the user's binary-addition precedent: gaps that scale with tertile
(high > low) indicate magnitude-sensitivity, NOT necessarily more
"comprehensive" verification; position-INDEPENDENT gaps indicate genuine
bit-exact verification; large-early/small-late is a distinct pattern
requiring its own investigation, not folded into either reading.

PYTHONPATH=src:analysis python analysis/phase3_computesqrt_counterfactual_design.py
"""

import json
import math
from pathlib import Path

import numpy as np

RESULTS = Path("analysis_outputs/final_results")
EQUALS = "="
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


def build_valid_positive(rng, n_x_range=(4, 20), slack_range=(0, 4)):
    """Direct construction: random n_x in range, random x in [0, 2**n_x - 1],
    z = isqrt(x), n_z = bitlength(z) + random slack (extra legal zero
    padding). Returns dict with tokens + metadata."""
    n_x = int(rng.integers(n_x_range[0], n_x_range[1] + 1))
    x = int(rng.integers(0, 2 ** n_x))
    z = math.isqrt(x)
    need = bitlength(z)
    slack = int(rng.integers(slack_range[0], slack_range[1] + 1))
    n_z = need + slack
    u_x = encode_lsb_first(x, n_x)
    u_z = encode_lsb_first(z, n_z)
    tokens = u_x + [EQUALS] + u_z
    return {
        "tokens": tokens, "x": x, "z": z, "n_x": n_x, "n_z": n_z, "need_bits": need,
        "marker_idx": n_x,
    }


def is_positive_local(seq):
    eq_idx = [i for i, t in enumerate(seq) if t == EQUALS]
    if len(eq_idx) != 1:
        return False
    i = eq_idx[0]
    u_x, u_z = seq[:i], seq[i + 1:]
    if not u_x or not u_z:
        return False
    x = decode_binary(u_x)
    z = decode_binary(u_z)
    return math.isqrt(x) == z


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


# ---------------------------------------------------------------------------
# condition 1: structural features
# ---------------------------------------------------------------------------

def marker_count_pairs(n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(rng)
        clean = b["tokens"]
        corrupt = clean[:]
        corrupt[0] = EQUALS  # EARLY second marker
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "clean_marker_count": clean.count(EQUALS), "corrupt_marker_count": corrupt.count(EQUALS),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} marker_count pairs")
    return pairs


def audit_marker_count_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["clean_all_count_1"] = all(p["clean_marker_count"] == 1 for p in pairs)
    a["corrupt_all_count_2"] = all(p["corrupt_marker_count"] == 2 for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and
        a["clean_all_count_1"] and a["corrupt_all_count_2"]
    )
    return a


def no_marker_pairs(n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(rng)
        clean = b["tokens"]
        corrupt = clean[:]
        corrupt[b["marker_idx"]] = str(int(rng.integers(0, 2)))
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "clean_marker_count": clean.count(EQUALS), "corrupt_marker_count": corrupt.count(EQUALS),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} no_marker pairs")
    return pairs


def audit_no_marker_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["clean_all_count_1"] = all(p["clean_marker_count"] == 1 for p in pairs)
    a["corrupt_all_count_0"] = all(p["corrupt_marker_count"] == 0 for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and
        a["clean_all_count_1"] and a["corrupt_all_count_0"]
    )
    return a


def marker_position_pairs(n_pairs, rng, n_x_range=(4, 10), slack_range=(0, 2)):
    """NOTE: since u_x is structurally almost always longer than u_z (x
    needs roughly 2x the bits z does), the marker naturally falls in the
    MID/HIGH tertile, essentially never LOW -- confirmed empirically (0/30
    'low' hits attempted with a forced-low target). So clean uses its
    NATURAL (unconstrained) marker position, and corrupt is forced into
    whichever tertile clean's natural position is NOT in (preferring
    'low', which is the most structurally distinctive contrast)."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(rng, n_x_range=n_x_range, slack_range=slack_range)
        clean = b["tokens"]
        L = len(clean)
        clean_tertile = categorize_tertile(b["marker_idx"] / (L - 1))
        target_tertile = "low" if clean_tertile != "low" else "high"
        corrupt = [str(int(rng.integers(0, 2))) for _ in range(L)]
        target_positions = [i for i in range(L) if categorize_tertile(i / (L - 1)) == target_tertile]
        far_idx = int(rng.choice(target_positions))
        corrupt[far_idx] = EQUALS
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": L,
            "clean_marker_idx": b["marker_idx"], "corrupt_marker_idx": far_idx,
            "clean_marker_tertile": clean_tertile, "corrupt_marker_tertile": target_tertile,
            "clean_marker_count": clean.count(EQUALS), "corrupt_marker_count": corrupt.count(EQUALS),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} marker_position pairs")
    return pairs


def audit_marker_position_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    a["same_marker_count"] = all(p["clean_marker_count"] == p["corrupt_marker_count"] == 1 for p in pairs)
    a["tertiles_always_differ"] = all(p["clean_marker_tertile"] != p["corrupt_marker_tertile"] for p in pairs)
    a["clean_tertile_distribution"] = {t: sum(1 for p in pairs if p["clean_marker_tertile"] == t) for t in TERTILES}
    a["corrupt_tertile_distribution"] = {t: sum(1 for p in pairs if p["corrupt_marker_tertile"] == t) for t in TERTILES}
    a["target_property_isolated"] = (
        a["same_length"] and a["same_marker_count"] and a["tertiles_always_differ"]
    )
    return a


def null_control_pairs(n_pairs, rng, z_range=(2, 30), max_tries=MAX_TRIES):
    """Exploits the many-to-one x->z mapping: for a fixed z, EVERY x in
    [z**2, (z+1)**2 - 1] is equally correct. Finds x1, x2 in that range
    differing in EXACTLY ONE BIT, builds both full sequences with IDENTICAL
    u_z (n_x/n_z fixed to comfortably cover the range) -- a true 1-position
    minimal pair, both genuinely valid, same true label."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * max_tries:
        attempts += 1
        z = int(rng.integers(z_range[0], z_range[1] + 1))
        lo, hi = z * z, (z + 1) * (z + 1) - 1
        n_x = bitlength(hi) + 2  # comfortable fixed width covering the whole bucket
        n_z = bitlength(z) + 1
        x1 = int(rng.integers(lo, hi + 1))
        bit = int(rng.integers(0, n_x))
        x2 = x1 ^ (1 << bit)
        if not (lo <= x2 <= hi) or x2 == x1:
            continue
        assert math.isqrt(x1) == z and math.isqrt(x2) == z
        u_z = encode_lsb_first(z, n_z)
        clean = encode_lsb_first(x1, n_x) + [EQUALS] + u_z
        corrupt = encode_lsb_first(x2, n_x) + [EQUALS] + u_z
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "z": z, "x1": x1, "x2": x2, "flipped_bit": bit,
            "clean_is_positive": is_positive_local(clean), "corrupt_is_positive": is_positive_local(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} null_control pairs")
    return pairs


def audit_null_control_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["x_values_differ"] = all(p["x1"] != p["x2"] for p in pairs)
    a["both_genuinely_valid_same_label"] = all(p["clean_is_positive"] and p["corrupt_is_positive"] for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and a["x_values_differ"] and
        a["both_genuinely_valid_same_label"]
    )
    a["note"] = (
        "TRUE 1-position minimal pair, both genuinely valid with the SAME true label -- exploits "
        "compute-sqrt's own many-to-one x->z structure (a whole range of x maps to the same correct "
        "z) rather than an engineered mirrored flip. Any observed gap here cannot be attributed to "
        "unrelated content differences."
    )
    return a


# ---------------------------------------------------------------------------
# condition 2: task-specific structural features
# ---------------------------------------------------------------------------

def answer_length_insufficient_pairs(n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(rng, slack_range=(0, 0))  # n_z == need_bits exactly, no slack
        clean = b["tokens"]
        if b["n_z"] < 1:
            continue
        assert clean[-1] == "1", "MSB of a minimal-width z representation must be 1"
        corrupt = clean[:-1]  # drop the last (highest-order, necessarily '1') token
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt,
            "clean_length": len(clean), "corrupt_length": len(corrupt),
            "clean_length_sufficient": answer_length_sufficient(clean),
            "corrupt_length_sufficient": answer_length_sufficient(corrupt),
            "clean_marker_count": clean.count(EQUALS), "corrupt_marker_count": corrupt.count(EQUALS),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} answer_length_insufficient pairs")
    return pairs


def audit_answer_length_insufficient_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["corrupt_is_one_token_shorter"] = all(p["corrupt_length"] == p["clean_length"] - 1 for p in pairs)
    a["same_marker_count"] = all(p["clean_marker_count"] == p["corrupt_marker_count"] == 1 for p in pairs)
    a["clean_all_length_sufficient"] = all(p["clean_length_sufficient"] for p in pairs)
    a["corrupt_all_length_insufficient"] = all(not p["corrupt_length_sufficient"] for p in pairs)
    a["prefix_unchanged"] = all(p["clean"][:p["corrupt_length"]] == p["corrupt"] for p in pairs)
    a["target_property_isolated"] = (
        a["corrupt_is_one_token_shorter"] and a["same_marker_count"] and a["clean_all_length_sufficient"] and
        a["corrupt_all_length_insufficient"] and a["prefix_unchanged"]
    )
    return a


def late_second_marker_pairs(n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(rng, slack_range=(1, 4))  # ensure u_z has >=1 token to overwrite
        clean = b["tokens"]
        corrupt = clean[:]
        corrupt[-1] = EQUALS  # LATE second marker (mirrors the real hard-negative distribution)
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "clean_marker_count": clean.count(EQUALS), "corrupt_marker_count": corrupt.count(EQUALS),
            "marker_position_relative_corrupt_second_marker": (len(clean) - 1) / (len(clean) - 1),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} late_second_marker pairs")
    return pairs


def audit_late_second_marker_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["clean_all_count_1"] = all(p["clean_marker_count"] == 1 for p in pairs)
    a["corrupt_all_count_2"] = all(p["corrupt_marker_count"] == 2 for p in pairs)
    a["second_marker_always_at_last_position (LATE, matches real hard-negative pattern)"] = True
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and
        a["clean_all_count_1"] and a["corrupt_all_count_2"]
    )
    return a


# ---------------------------------------------------------------------------
# condition 3: target-computation null test, swept low/mid/high bit position
# (LSB-first: low tertile = low-order/small-magnitude, high tertile =
# high-order/large-magnitude)
# ---------------------------------------------------------------------------

def target_computation_pairs(n_pairs, rng, n_z_range=(9, 15), fixed_tertile=None, max_tries=MAX_TRIES):
    pairs, seen, tries_total = [], set(), 0
    while len(pairs) < n_pairs and tries_total < n_pairs * max_tries:
        target_tertile = fixed_tertile if fixed_tertile is not None else TERTILES[int(rng.integers(0, 3))]
        found = False
        for _ in range(max_tries):
            tries_total += 1
            n_z = int(rng.integers(n_z_range[0], n_z_range[1] + 1))
            b = build_valid_positive(rng, n_x_range=(4, 20), slack_range=(0, 0))
            if b["n_z"] > n_z:
                continue  # need slack room to place a bit at the target tertile
            # rebuild with the target n_z (adds legal slack padding)
            x, z = b["x"], b["z"]
            u_x, marker_idx = b["tokens"][:b["marker_idx"]], b["marker_idx"]
            u_z = encode_lsb_first(z, n_z)
            clean = u_x + [EQUALS] + u_z
            candidates = list(range(n_z))
            denom = max(1, n_z - 1)
            def frac_of(j):
                return j / denom
            bucket = [j for j in candidates if categorize_tertile(frac_of(j)) == target_tertile]
            if not bucket:
                continue
            bit_idx = int(rng.choice(bucket))
            corrupt = clean[:]
            flip_pos = marker_idx + 1 + bit_idx
            corrupt[flip_pos] = "0" if corrupt[flip_pos] == "1" else "1"
            z_corrupt = decode_binary(corrupt[marker_idx + 1:])
            key = (tuple(clean), tuple(corrupt))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "clean": clean, "corrupt": corrupt, "length": len(clean),
                "x": x, "z_clean": z, "z_corrupt": z_corrupt, "n_z": n_z,
                "flipped_bit_idx": bit_idx, "flip_position_relative": frac_of(bit_idx),
                "flip_position_tertile": target_tertile,
                "bit_magnitude": 2 ** bit_idx,
                "clean_all_three_features": all(fn(clean) for fn in SHORTCUT_FEATURES.values()),
                "corrupt_all_three_features": all(fn(corrupt) for fn in SHORTCUT_FEATURES.values()),
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
    a["clean_all_three_shortcut_features_hold"] = all(p["clean_all_three_features"] for p in pairs)
    a["corrupt_all_three_shortcut_features_ALSO_hold"] = all(p["corrupt_all_three_features"] for p in pairs)
    a["z_corrupt_always_differs_from_z_clean"] = all(p["z_corrupt"] != p["z_clean"] for p in pairs)
    a["clean_all_positive_corrupt_all_negative"] = all(
        p["clean_is_positive"] and not p["corrupt_is_positive"] for p in pairs)
    tertile_hist = {t: sum(1 for p in pairs if p["flip_position_tertile"] == t) for t in TERTILES}
    a["flip_position_tertile_histogram"] = tertile_hist
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and
        a["clean_all_three_shortcut_features_hold"] and a["corrupt_all_three_shortcut_features_ALSO_hold"] and
        a["z_corrupt_always_differs_from_z_clean"] and a["clean_all_positive_corrupt_all_negative"]
    )
    a["note"] = (
        "All three shortcut features (marker count, well-formedness, answer-length-sufficiency) are "
        "IDENTICAL between clean and corrupt for every pair -- structurally guaranteed since only one "
        "u_z bit is touched, x and field widths are untouched. LSB-first encoding: 'low' tertile = "
        "low-order/small-magnitude bit flips, 'high' tertile = high-order/large-magnitude flips."
    )
    return a


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260718)

    mc = marker_count_pairs(30, rng)
    nmk = no_marker_pairs(30, rng)
    mp = marker_position_pairs(30, rng)
    nc = null_control_pairs(30, rng)
    alip = answer_length_insufficient_pairs(30, rng)
    lsmp = late_second_marker_pairs(30, rng)
    tcp = target_computation_pairs(30, rng)

    audits = {
        "marker_count_pairs": audit_marker_count_pairs(mc),
        "no_marker_pairs": audit_no_marker_pairs(nmk),
        "marker_position_pairs": audit_marker_position_pairs(mp),
        "null_control_pairs": audit_null_control_pairs(nc),
        "answer_length_insufficient_pairs": audit_answer_length_insufficient_pairs(alip),
        "late_second_marker_pairs": audit_late_second_marker_pairs(lsmp),
        "target_computation_pairs": audit_target_computation_pairs(tcp),
    }
    for name, a in audits.items():
        print(f"=== {name} ===", flush=True)
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    out = {
        "task": "compute-sqrt",
        "description": (
            "Phase 3 counterfactual pair construction + audit for compute-sqrt causal patching, all "
            "three approved conditions. NO PATCHING RUN. null_control_pairs is a TRUE 1-position "
            "minimal pair built correctly from the start (per the missing-duplicate-string/stack-"
            "manipulation lesson), exploiting the task's own many-to-one x->z mathematical "
            "redundancy. Condition 3 targets the 411-example value_mismatch_only residual, swept by "
            "LSB-first bit-position tertile (low=small-magnitude, high=large-magnitude), with the "
            "magnitude-sensitivity interpretive discipline flagged explicitly."
        ),
        "condition_1_structural": {
            "marker_count_pairs_EARLY": {"audit": audits["marker_count_pairs"], "sample_pairs": mc[:3]},
            "no_marker_pairs": {"audit": audits["no_marker_pairs"], "sample_pairs": nmk[:3]},
            "marker_position_pairs": {"audit": audits["marker_position_pairs"], "sample_pairs": mp[:3]},
            "null_control_pairs_TRUE_MINIMAL_PAIR": {"audit": audits["null_control_pairs"], "sample_pairs": nc[:3]},
        },
        "condition_2_task_specific_structural": {
            "answer_length_insufficient_pairs": {"audit": audits["answer_length_insufficient_pairs"], "sample_pairs": alip[:3]},
            "late_second_marker_pairs_LATE_probe": {"audit": audits["late_second_marker_pairs"], "sample_pairs": lsmp[:3]},
        },
        "condition_3_target_computation_null_test_value_mismatch_residual": {
            "target_computation_pairs": {"audit": audits["target_computation_pairs"], "sample_pairs": tcp[:3]},
        },
    }
    out_path = RESULTS / "phase3_computesqrt_counterfactual_design.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
