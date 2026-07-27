"""Stack-manipulation Phase 3 (bounded) pilot: counterfactual pair
construction + audit for the three approved conditions. Mirrors bucket-
sort/Dyck-2-3/missing-duplicate-string's construction discipline: each pair
type isolates ONE candidate feature, audited explicitly, no patching here.

Uses phase3_stackmanip_task_audit.py's confirmed grammar and six shortcut
features (marker_count_is_one, operations_well_formed, stack_size_
arithmetic_consistent, first_token_never_pop, token_before_marker_never_
push, answer_bit_counts_subset_of_available), and its key finding: the
62-example value_mismatch_only residual (2.4% of negatives) is the ONLY
population requiring genuine stack simulation -- everything else is caught
by the six shortcut checks (97.6% joint coverage).

CONDITION 1 (structural: marker features, well-formedness, TRUE null
control):
  marker_count_pairs: clean=valid positive. corrupt=SAME LENGTH, the final
    answer token replaced by a second '#' -- marker_count 1->2.
  malformed_operations_pairs: clean=valid positive with >=1 PUSH. corrupt=
    SAME LENGTH, the bit immediately following a PUSH replaced by "POP" --
    breaks operations_well_formed (malformed_push_no_bit) at that exact
    token, marker count/position untouched.
  null_control_pairs (TRUE minimal pair, matching the missing-duplicate-
    string null-control-rebuild discipline from the start): clean=valid
    positive whose INITIAL STACK BOTTOM (index 0) survives to the end
    (never popped) -- flips initial_stack[0] and its unique mirror
    (answer[-1], the bottom-of-stack's position in the reversed final
    answer) SIMULTANEOUSLY to the other binary value. Since this value
    never influences control flow (which/how many push/pop operations
    occur), both members remain genuinely valid positives with the SAME
    true label -- a real 2-position minimal pair.

CONDITION 2 (task-specific structural features -- length_mismatch pathway /
stack_size_arithmetic_consistent, answer_bit_counts_subset_of_available):
  length_mismatch_pairs: clean=valid positive. corrupt=SAME operations/
    marker, ONE token appended to the answer (duplicating the last answer
    token, or a fresh token if the answer is empty) -- breaks stack_size_
    arithmetic_consistent (answer length no longer matches push-count -
    pop-count + initial-stack-size) while marker_count/operations_well_
    formed are untouched.
  bit_count_availability_pairs: clean=valid positive built from an
    ALL-ZERO program (initial stack and every pushed value are '0', so the
    only bit value EVER available is '0') -- final answer is therefore
    all-'0'. corrupt=ONE answer token flipped '0'->'1', which requires a
    '1' that was NEVER made available anywhere in the program -- breaks
    answer_bit_counts_subset_of_available (and necessarily also creates a
    value_mismatch, an expected/documented coupling, analogous to missing-
    duplicate-string's Condition 2 coupling caveat).

CONDITION 3 (target-computation null test on the value_mismatch_only
residual): clean=valid positive (marker_count=1, well-formed operations,
arithmetically-consistent answer length, bit-counts available -- ALL SIX
shortcut features satisfied). corrupt=SAME LENGTH, SAME marker position,
SAME operations, SAME answer length, SAME per-value bit counts in the
answer (all six shortcut features STILL satisfied) -- built via swapping
TWO ADJACENT answer positions with DIFFERING values (a swap can't change a
multiset, so bit-count-availability is untouched) -- breaks the true
LIFO order without tripping any of the six shortcut checks. Swap position
swept EARLY/MID/LATE within the answer via the established fixed-target-
tertile-then-retry discipline (matching bucket-sort's sort-order / Dyck-
2-3's LIFO-violation / missing-duplicate-string's position-mismatch
constructions).

PYTHONPATH=src:analysis python analysis/phase3_stackmanip_counterfactual_design.py
"""

import json
from pathlib import Path

import numpy as np

RESULTS = Path("analysis_outputs/final_results")
PUSH, POP, MARKER = "PUSH", "POP", "#"
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
# shared construction helper: direct simulation (not the biased FLaRe
# sampler), matching _is_positive's exact semantics
# ---------------------------------------------------------------------------

def build_valid_positive(n_stack, n_ops, rng, all_zero=False):
    """initial_stack: n_stack random bits (or all '0' if all_zero). n_ops
    random PUSH/POP operations (POP only when stack non-empty; pushed
    values random unless all_zero). Returns dict with tokens + metadata
    (n_push, n_pop, final_stack, whether the ORIGINAL bottom (index 0 of
    initial_stack) survived to the end -- i.e. the stack never emptied)."""
    def bit(rng):
        return "0" if all_zero else str(int(rng.integers(0, 2)))

    initial_stack = [bit(rng) for _ in range(n_stack)]
    stack = initial_stack[:]
    ops_tokens = []
    n_push = n_pop = 0
    emptied = False
    for _ in range(n_ops):
        can_pop = len(stack) > 0
        if can_pop and rng.random() < 0.5:
            stack.pop()
            ops_tokens.append(POP)
            n_pop += 1
            if not stack:
                emptied = True
        else:
            val = bit(rng)
            ops_tokens.append(PUSH)
            ops_tokens.append(val)
            stack.append(val)
            n_push += 1
    answer = list(reversed(stack))
    tokens = initial_stack + ops_tokens + [MARKER] + answer
    bottom_survived = n_stack > 0 and not emptied
    return {
        "tokens": tokens, "initial_stack": initial_stack, "ops_tokens": ops_tokens,
        "n_stack": n_stack, "n_push": n_push, "n_pop": n_pop, "final_stack": stack,
        "answer": answer, "bottom_survived": bottom_survived,
        "marker_idx": len(initial_stack) + len(ops_tokens),
    }


def is_positive_local(seq):
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
                return False
        else:
            if not stack:
                return False
            stack.pop()
            i += 1
    if i < n and seq[i] == MARKER:
        i += 1
    else:
        return False
    return seq[i:] == list(reversed(stack))


def marker_count_is_one(seq):
    return seq.count(MARKER) == 1


def operations_well_formed(seq):
    n = len(seq)
    i = 0
    stack = []
    while i < n and seq[i] in ("0", "1"):
        stack.append(seq[i])
        i += 1
    while i < n and seq[i] in (PUSH, POP):
        if seq[i] == PUSH:
            i += 1
            if not (i < n and seq[i] in ("0", "1")):
                return False
            stack.append(seq[i])
            i += 1
        else:
            if not stack:
                return False
            stack.pop()
            i += 1
    return True


def stack_size_arithmetic_consistent(seq):
    if seq.count(MARKER) != 1 or not operations_well_formed(seq):
        return False
    i = seq.index(MARKER)
    n = len(seq)
    j = 0
    stack = []
    while j < n and seq[j] in ("0", "1"):
        stack.append(seq[j])
        j += 1
    while j < i:
        if seq[j] == PUSH:
            j += 2
            stack.append(None)
        else:
            stack.pop()
            j += 1
    return len(seq) - (i + 1) == len(stack)


def first_token_never_pop(seq):
    return bool(seq) and seq[0] != POP


def token_before_marker_never_push(seq):
    if seq.count(MARKER) != 1:
        return False
    i = seq.index(MARKER)
    return i == 0 or seq[i - 1] != PUSH


def answer_bit_counts_subset_of_available(seq):
    if seq.count(MARKER) != 1:
        return False
    i = seq.index(MARKER)
    prefix, suffix = seq[:i], seq[i + 1:]
    from collections import Counter
    avail = Counter(t for t in prefix if t in ("0", "1"))
    ans = Counter(t for t in suffix if t in ("0", "1"))
    if len(suffix) != sum(ans.values()):
        return False
    return ans["0"] <= avail["0"] and ans["1"] <= avail["1"]


SHORTCUT_FEATURES = {
    "marker_count_is_one": marker_count_is_one,
    "operations_well_formed": operations_well_formed,
    "stack_size_arithmetic_consistent": stack_size_arithmetic_consistent,
    "first_token_never_pop": first_token_never_pop,
    "token_before_marker_never_push": token_before_marker_never_push,
    "answer_bit_counts_subset_of_available": answer_bit_counts_subset_of_available,
}


# ---------------------------------------------------------------------------
# condition 1: structural features
# ---------------------------------------------------------------------------

def marker_count_pairs(n_pairs, rng, n_stack=3, n_ops=8):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(n_stack, n_ops, rng)
        if not b["answer"]:
            continue
        clean = b["tokens"]
        corrupt = clean[:]
        corrupt[-1] = MARKER
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "clean_marker_count": clean.count(MARKER), "corrupt_marker_count": corrupt.count(MARKER),
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


def malformed_operations_pairs(n_pairs, rng, n_stack=2, n_ops=8):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(n_stack, n_ops, rng)
        clean = b["tokens"]
        push_positions = [i for i, t in enumerate(clean[:b["marker_idx"]]) if t == PUSH]
        if not push_positions:
            continue
        p_idx = int(rng.choice(push_positions))
        corrupt = clean[:]
        corrupt[p_idx + 1] = POP  # PUSH now followed by POP, not a bit
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean), "flip_idx": p_idx + 1,
            "clean_marker_count": clean.count(MARKER), "corrupt_marker_count": corrupt.count(MARKER),
            "clean_well_formed": operations_well_formed(clean), "corrupt_well_formed": operations_well_formed(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} malformed_operations pairs")
    return pairs


def audit_malformed_operations_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["same_marker_count"] = all(p["clean_marker_count"] == p["corrupt_marker_count"] == 1 for p in pairs)
    a["clean_all_well_formed"] = all(p["clean_well_formed"] for p in pairs)
    a["corrupt_all_malformed"] = all(not p["corrupt_well_formed"] for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and a["same_marker_count"] and
        a["clean_all_well_formed"] and a["corrupt_all_malformed"]
    )
    return a


def null_control_pairs(n_pairs, rng, n_stack=3, n_ops=8, max_tries=MAX_TRIES):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * max_tries:
        attempts += 1
        b = build_valid_positive(n_stack, n_ops, rng)
        if not b["bottom_survived"]:
            continue
        clean = b["tokens"]
        v0 = clean[0]
        newv = "0" if v0 == "1" else "1"
        corrupt = clean[:]
        corrupt[0] = newv
        assert corrupt[-1] == v0, "answer[-1] should equal the surviving stack bottom"
        corrupt[-1] = newv
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "clean_first_symbol": clean[0], "corrupt_first_symbol": corrupt[0],
            "clean_is_positive": is_positive_local(clean), "corrupt_is_positive": is_positive_local(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} null_control pairs")
    return pairs


def audit_null_control_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_two_positions"] = all(d == 2 for d in diffs)
    a["first_symbols_differ"] = all(p["clean_first_symbol"] != p["corrupt_first_symbol"] for p in pairs)
    a["both_genuinely_valid_same_label"] = all(p["clean_is_positive"] and p["corrupt_is_positive"] for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_two_positions"] and a["first_symbols_differ"] and
        a["both_genuinely_valid_same_label"]
    )
    a["note"] = (
        "TRUE minimal pair (differs at exactly the 2 mirrored positions -- initial_stack[0] and its "
        "unique surviving mirror answer[-1] -- everything else, including all operation tokens and "
        "marker position, identical): a value that never influences control flow, only data. Any "
        "observed gap here cannot be attributed to unrelated content differences."
    )
    return a


# ---------------------------------------------------------------------------
# condition 2: task-specific structural features (critical, 97.6% coverage)
# ---------------------------------------------------------------------------

def length_mismatch_pairs(n_pairs, rng, n_stack=3, n_ops=8):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(n_stack, n_ops, rng)
        clean = b["tokens"]
        extra = clean[-1] if b["answer"] else "0"
        corrupt = clean + [extra]
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt,
            "clean_length": len(clean), "corrupt_length": len(corrupt),
            "clean_arith_consistent": stack_size_arithmetic_consistent(clean),
            "corrupt_arith_consistent": stack_size_arithmetic_consistent(corrupt),
            "clean_marker_count": clean.count(MARKER), "corrupt_marker_count": corrupt.count(MARKER),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} length_mismatch pairs")
    return pairs


def audit_length_mismatch_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["corrupt_is_one_token_longer"] = all(p["corrupt_length"] == p["clean_length"] + 1 for p in pairs)
    a["same_marker_count"] = all(p["clean_marker_count"] == p["corrupt_marker_count"] == 1 for p in pairs)
    a["clean_all_arith_consistent"] = all(p["clean_arith_consistent"] for p in pairs)
    a["corrupt_all_inconsistent"] = all(not p["corrupt_arith_consistent"] for p in pairs)
    a["prefix_unchanged"] = all(p["corrupt"][:p["clean_length"]] == p["clean"] for p in pairs)
    a["target_property_isolated"] = (
        a["corrupt_is_one_token_longer"] and a["same_marker_count"] and a["clean_all_arith_consistent"] and
        a["corrupt_all_inconsistent"] and a["prefix_unchanged"]
    )
    return a


def bit_count_availability_pairs(n_pairs, rng, n_stack=3, n_ops=8):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        b = build_valid_positive(n_stack, n_ops, rng, all_zero=True)
        if not b["answer"]:
            continue
        clean = b["tokens"]
        corrupt = clean[:]
        corrupt[-1] = "1"  # requires a '1' that was NEVER made available (all-zero program)
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "clean_bitcount_ok": answer_bit_counts_subset_of_available(clean),
            "corrupt_bitcount_ok": answer_bit_counts_subset_of_available(corrupt),
            "clean_marker_count": clean.count(MARKER), "corrupt_marker_count": corrupt.count(MARKER),
            "clean_arith_consistent": stack_size_arithmetic_consistent(clean),
            "corrupt_arith_consistent": stack_size_arithmetic_consistent(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} bit_count_availability pairs")
    return pairs


def audit_bit_count_availability_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["same_marker_count"] = all(p["clean_marker_count"] == p["corrupt_marker_count"] == 1 for p in pairs)
    a["same_length_arith (unaffected by this feature)"] = all(
        p["clean_arith_consistent"] and p["corrupt_arith_consistent"] for p in pairs)
    a["clean_all_bitcount_ok"] = all(p["clean_bitcount_ok"] for p in pairs)
    a["corrupt_all_bitcount_violated"] = all(not p["corrupt_bitcount_ok"] for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and a["same_marker_count"] and
        a["clean_all_bitcount_ok"] and a["corrupt_all_bitcount_violated"]
    )
    a["coupling_note"] = (
        "A single-token flip necessarily ALSO creates a value_mismatch (the flipped answer token no "
        "longer equals the true reversed stack) -- documented, not hidden, analogous to missing-"
        "duplicate-string's Condition 2 coupling caveat: length/well-formedness/marker features are "
        "held fixed, but bit-count-availability and true-value-correctness are coupled for a single "
        "flip in a binary alphabet."
    )
    return a


# ---------------------------------------------------------------------------
# condition 3: target-computation null test (value_mismatch_only residual),
# swept early/mid/late
# ---------------------------------------------------------------------------

def target_computation_pairs(n_pairs, rng, n_stack=3, n_ops=20, max_tries=MAX_TRIES, fixed_tertile=None):
    """Swap two ADJACENT, DIFFERING answer positions -- preserves length,
    marker count/position, well-formedness (only the answer is touched),
    and per-value bit counts (a swap can't change a multiset) -- breaks
    ONLY true LIFO order. Swept by tertile (fixed-target-tertile-then-retry,
    BSF1/bucket-sort/Dyck-2-3/missing-duplicate-string discipline)."""
    pairs, seen, tries_total = [], set(), 0
    while len(pairs) < n_pairs and tries_total < n_pairs * max_tries:
        target_tertile = fixed_tertile if fixed_tertile is not None else TERTILES[int(rng.integers(0, 3))]
        found = False
        for _ in range(max_tries):
            tries_total += 1
            b = build_valid_positive(n_stack, n_ops, rng)
            answer = b["answer"]
            k = len(answer)
            if k < 3:
                continue
            candidates = [j for j in range(k - 1) if answer[j] != answer[j + 1]]
            if not candidates:
                continue
            denom = max(1, k - 2)
            def frac_of(j):
                return j / denom
            bucket = [j for j in candidates if categorize_tertile(min(1.0, frac_of(j))) == target_tertile]
            if not bucket:
                continue
            j = int(rng.choice(bucket))
            clean = b["tokens"]
            marker_idx = b["marker_idx"]
            i1, i2 = marker_idx + 1 + j, marker_idx + 1 + j + 1
            corrupt = clean[:]
            corrupt[i1], corrupt[i2] = corrupt[i2], corrupt[i1]
            key = (tuple(clean), tuple(corrupt))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "clean": clean, "corrupt": corrupt, "length": len(clean),
                "swap_indices": [i1, i2], "swap_position_relative": frac_of(j),
                "swap_position_tertile": target_tertile,
                "clean_all_six_features": all(fn(clean) for fn in SHORTCUT_FEATURES.values()),
                "corrupt_all_six_features": all(fn(corrupt) for fn in SHORTCUT_FEATURES.values()),
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
    a["differs_at_exactly_two_positions"] = all(d == 2 for d in diffs)
    a["clean_all_six_shortcut_features_hold"] = all(p["clean_all_six_features"] for p in pairs)
    a["corrupt_all_six_shortcut_features_ALSO_hold"] = all(p["corrupt_all_six_features"] for p in pairs)
    a["clean_all_positive_corrupt_all_negative"] = all(
        p["clean_is_positive"] and not p["corrupt_is_positive"] for p in pairs)
    tertile_hist = {t: sum(1 for p in pairs if p["swap_position_tertile"] == t) for t in TERTILES}
    a["swap_position_tertile_histogram"] = tertile_hist
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_two_positions"] and
        a["clean_all_six_shortcut_features_hold"] and a["corrupt_all_six_shortcut_features_ALSO_hold"] and
        a["clean_all_positive_corrupt_all_negative"]
    )
    a["note"] = (
        "All six shortcut features are IDENTICAL between clean and corrupt for every pair -- a model "
        "relying purely on those checks cannot distinguish them, yet the true label flips (a genuine "
        "value_mismatch_only case, matching the 62-example residual isolated in the task audit)."
    )
    return a


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260718)

    mc = marker_count_pairs(30, rng)
    mop = malformed_operations_pairs(30, rng)
    nc = null_control_pairs(30, rng)
    lmp = length_mismatch_pairs(30, rng)
    bcp = bit_count_availability_pairs(30, rng)
    tcp = target_computation_pairs(30, rng, n_ops=20)

    audits = {
        "marker_count_pairs": audit_marker_count_pairs(mc),
        "malformed_operations_pairs": audit_malformed_operations_pairs(mop),
        "null_control_pairs": audit_null_control_pairs(nc),
        "length_mismatch_pairs": audit_length_mismatch_pairs(lmp),
        "bit_count_availability_pairs": audit_bit_count_availability_pairs(bcp),
        "target_computation_pairs": audit_target_computation_pairs(tcp),
    }
    for name, a in audits.items():
        print(f"=== {name} ===", flush=True)
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    out = {
        "task": "stack-manipulation",
        "description": (
            "Phase 3 counterfactual pair construction + audit for stack-manipulation causal "
            "patching, all three approved conditions. NO PATCHING RUN. Condition 1 includes a TRUE "
            "minimal-pair null control from the start (matching the discipline established after "
            "missing-duplicate-string's null-control catch). Condition 3 targets the exact 62-example "
            "value_mismatch_only residual identified in phase3_stackmanip_task_audit.json."
        ),
        "condition_1_structural": {
            "marker_count_pairs": {"audit": audits["marker_count_pairs"], "sample_pairs": mc[:3]},
            "malformed_operations_pairs": {"audit": audits["malformed_operations_pairs"], "sample_pairs": mop[:3]},
            "null_control_pairs_TRUE_MINIMAL_PAIR": {"audit": audits["null_control_pairs"], "sample_pairs": nc[:3]},
        },
        "condition_2_task_specific_structural_critical_97pct_coverage": {
            "length_mismatch_pairs": {"audit": audits["length_mismatch_pairs"], "sample_pairs": lmp[:3]},
            "bit_count_availability_pairs": {"audit": audits["bit_count_availability_pairs"], "sample_pairs": bcp[:3]},
        },
        "condition_3_target_computation_null_test_value_mismatch_residual": {
            "target_computation_pairs": {"audit": audits["target_computation_pairs"], "sample_pairs": tcp[:3]},
        },
    }
    out_path = RESULTS / "phase3_stackmanip_counterfactual_design.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
