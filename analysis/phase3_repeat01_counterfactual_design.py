"""Repeat-01 Phase 3 (bounded) pilot: counterfactual pair construction +
audit for the three approved conditions. Mirrors the established
construction discipline: each pair type isolates ONE candidate feature,
audited explicitly, no patching here.

NULL CONTROL CORRECTION (per the user's own "verify this" instruction):
the suggested position-i/mirror-(n-1-i) swap does NOT preserve validity --
verified empirically (and provable directly from the DFA): expected[i] and
expected[n-1-i] are ALWAYS opposite values (since n-1 is odd for even n,
parity of n-1-i is opposite to parity of i), so swapping them makes BOTH
positions wrong simultaneously. This is a direct consequence of the task
audit's finding that repeat-01 has ZERO within-length content redundancy
-- for any fixed even length, (01)^k is the UNIQUE valid string (the DFA
is fully deterministic with no unspecified transitions besides dead-
reject). Unlike every other task in this pilot (compute-sqrt's sqrt-
bucket, missing-duplicate-string's non-surviving-stack-bit, binary-
multiplication's commutativity), there is no content-level "irrelevant
bit" to exploit for a same-length minimal pair. The CLOSEST achievable
TRUE pair (both genuinely valid, same label) is therefore a LENGTH-
EXTENSION pair: clean=(01)^k, corrupt=(01)^(k+1) -- clean is an exact
PREFIX of corrupt, differing only by one appended '01' block (2 tokens),
the smallest possible content difference between two genuinely distinct
valid strings in this language.

CONDITION 1 (structural: length_parity_even, first_symbol_is_0, last_
symbol_is_1, count_balance, TRUE null control):
  length_parity_pairs: clean=valid positive (even length). corrupt=clean
    with one token appended (odd length).
  first_symbol_pairs: clean=valid positive. corrupt=position 0 swapped
    with an INTERIOR odd position j (1 <= j <= n-3, j != n-1) -- breaks
    first_symbol_is_0 (position 0 becomes '1') without touching last_
    symbol_is_1 (position n-1 untouched), preserves count_balance/length
    (a swap can't change either).
  last_symbol_pairs: clean=valid positive. corrupt=position n-1 swapped
    with an INTERIOR even position j (2 <= j <= n-3, j != 0) -- breaks
    last_symbol_is_1 without touching first_symbol_is_0, same
    preservation logic.
  count_balance_pairs: clean=valid positive. corrupt=one STRICTLY
    INTERIOR position (not 0, not n-1) flipped -- breaks count_balance by
    +/-2 without touching first/last symbol (boundary positions
    untouched).
  null_control_pairs (TRUE minimal pair, length-extension): clean=(01)^k,
    corrupt=(01)^(k+1) -- corrupt is clean plus one appended '01' block.
    Both genuinely valid, same true label.

CONDITION 2 (task-specific -- COLLAPSES INTO CONDITION 1's feature set per
the task's simplicity, as anticipated): no additional TRIVIAL aggregate
feature beyond the four already identified was found in the task audit
(the one candidate considered -- "every adjacent pair differs," local
alternation regardless of absolute phase -- turns out to be EQUIVALENT to
full correctness when combined with first_symbol_is_0 and length parity,
i.e. it IS genuine verification, not a shortcut, so it does not belong
here). Condition 2 instead tests the COMBINED effect of all four
shortcuts violated AT ONCE (matching how real "scrambled" negatives
typically fail multiple checks simultaneously, unlike Condition 1's
surgical single-feature isolations):
  combined_shortcut_violation_pairs: clean=valid positive. corrupt=an
    INDEPENDENTLY RANDOM binary string of the SAME LENGTH (typically
    violates length-parity-irrelevant-but-count/first/last simultaneously
    -- audited explicitly, not assumed).

CONDITION 3 (target-computation null test on the 115-example genuinely-
shortcut-blind population, NOT viol_repeat_01's hard set -- the audit's
key correction): clean=valid positive. corrupt=SAME LENGTH, an ADJACENT
swap at two DIFFERING positions (always available since adjacent positions
in (01)^k always differ) -- preserves length_parity/first_symbol/last_
symbol/count_balance (all four; a swap changes neither the boundary
values when placed away from the boundary, nor the multiset), breaks
alternation at exactly the two swapped positions. Swept EARLY/MID/LATE by
swap position, using the audit's own empirical distribution discipline:
the true shortcut-blind population is EARLY-CONCENTRATED (58/115 in the
first quintile), so the early tertile is expected to be well-powered here
-- UNLIKE the arithmetic tasks (binary-addition/compute-sqrt/binary-
multiplication), where "early=small effect, late=large effect" reflects
LSB-first MAGNITUDE scaling. For repeat-01, position has NO magnitude
interpretation (every position is an equally-weighted boolean check) --
an early>late gap here would simply match the NATURAL violation
distribution's own early concentration, not a magnitude-sensitivity
confound, and is reported with that discipline explicitly.

PYTHONPATH=src:analysis python analysis/phase3_repeat01_counterfactual_design.py
"""

import json
from pathlib import Path

import numpy as np

RESULTS = Path("analysis_outputs/final_results")
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

def build_valid_positive(k):
    """Direct construction: (01)^k, length 2k."""
    tokens = []
    for _ in range(k):
        tokens += ["0", "1"]
    return tokens


def expected(i):
    return "0" if i % 2 == 0 else "1"


def is_positive_local(seq):
    if len(seq) % 2 != 0:
        return False
    return all(seq[i] == expected(i) for i in range(len(seq)))


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


# ---------------------------------------------------------------------------
# condition 1: structural features
# ---------------------------------------------------------------------------

def length_parity_pairs(n_pairs, rng, k_range=(3, 20)):
    pairs = []
    for _ in range(n_pairs):
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        clean = build_valid_positive(k)
        corrupt = clean + ["0"]  # append a fixed content token
        pairs.append({
            "clean": clean, "corrupt": corrupt,
            "clean_length": len(clean), "corrupt_length": len(corrupt),
            "clean_length_parity": length_parity_even(clean), "corrupt_length_parity": length_parity_even(corrupt),
        })
    return pairs


def audit_length_parity_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["corrupt_is_one_token_longer"] = all(p["corrupt_length"] == p["clean_length"] + 1 for p in pairs)
    a["clean_all_even_corrupt_all_odd"] = all(p["clean_length_parity"] and not p["corrupt_length_parity"] for p in pairs)
    a["prefix_unchanged"] = all(p["corrupt"][:p["clean_length"]] == p["clean"] for p in pairs)
    a["target_property_isolated"] = (
        a["corrupt_is_one_token_longer"] and a["clean_all_even_corrupt_all_odd"] and a["prefix_unchanged"]
    )
    return a


def first_symbol_pairs(n_pairs, rng, k_range=(4, 20), max_tries=MAX_TRIES):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * max_tries:
        attempts += 1
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        clean = build_valid_positive(k)
        n = len(clean)
        odd_interior = [j for j in range(1, n - 1) if j % 2 == 1]  # exclude 0 and n-1
        if not odd_interior:
            continue
        j = int(rng.choice(odd_interior))
        corrupt = clean[:]
        corrupt[0], corrupt[j] = corrupt[j], corrupt[0]
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": n, "swap_j": j,
            "clean_first_symbol_is_0": first_symbol_is_0(clean), "corrupt_first_symbol_is_0": first_symbol_is_0(corrupt),
            "clean_last_symbol_is_1": last_symbol_is_1(clean), "corrupt_last_symbol_is_1": last_symbol_is_1(corrupt),
            "clean_count_balance": count_balance(clean), "corrupt_count_balance": count_balance(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} first_symbol pairs")
    return pairs


def audit_first_symbol_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    a["clean_all_first_symbol_ok"] = all(p["clean_first_symbol_is_0"] for p in pairs)
    a["corrupt_all_first_symbol_broken"] = all(not p["corrupt_first_symbol_is_0"] for p in pairs)
    a["last_symbol_untouched"] = all(p["clean_last_symbol_is_1"] == p["corrupt_last_symbol_is_1"] == True for p in pairs)
    a["count_balance_preserved"] = all(p["clean_count_balance"] and p["corrupt_count_balance"] for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["clean_all_first_symbol_ok"] and a["corrupt_all_first_symbol_broken"] and
        a["last_symbol_untouched"] and a["count_balance_preserved"]
    )
    return a


def last_symbol_pairs(n_pairs, rng, k_range=(4, 20), max_tries=MAX_TRIES):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * max_tries:
        attempts += 1
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        clean = build_valid_positive(k)
        n = len(clean)
        even_interior = [j for j in range(1, n - 1) if j % 2 == 0 and j != 0]
        if not even_interior:
            continue
        j = int(rng.choice(even_interior))
        corrupt = clean[:]
        corrupt[n - 1], corrupt[j] = corrupt[j], corrupt[n - 1]
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": n, "swap_j": j,
            "clean_first_symbol_is_0": first_symbol_is_0(clean), "corrupt_first_symbol_is_0": first_symbol_is_0(corrupt),
            "clean_last_symbol_is_1": last_symbol_is_1(clean), "corrupt_last_symbol_is_1": last_symbol_is_1(corrupt),
            "clean_count_balance": count_balance(clean), "corrupt_count_balance": count_balance(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} last_symbol pairs")
    return pairs


def audit_last_symbol_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    a["clean_all_last_symbol_ok"] = all(p["clean_last_symbol_is_1"] for p in pairs)
    a["corrupt_all_last_symbol_broken"] = all(not p["corrupt_last_symbol_is_1"] for p in pairs)
    a["first_symbol_untouched"] = all(p["clean_first_symbol_is_0"] == p["corrupt_first_symbol_is_0"] == True for p in pairs)
    a["count_balance_preserved"] = all(p["clean_count_balance"] and p["corrupt_count_balance"] for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["clean_all_last_symbol_ok"] and a["corrupt_all_last_symbol_broken"] and
        a["first_symbol_untouched"] and a["count_balance_preserved"]
    )
    return a


def count_balance_pairs(n_pairs, rng, k_range=(4, 20), max_tries=MAX_TRIES):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * max_tries:
        attempts += 1
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        clean = build_valid_positive(k)
        n = len(clean)
        interior = list(range(1, n - 1))
        if not interior:
            continue
        j = int(rng.choice(interior))
        corrupt = clean[:]
        corrupt[j] = "0" if corrupt[j] == "1" else "1"
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": n, "flip_j": j,
            "clean_first_symbol_is_0": first_symbol_is_0(clean), "corrupt_first_symbol_is_0": first_symbol_is_0(corrupt),
            "clean_last_symbol_is_1": last_symbol_is_1(clean), "corrupt_last_symbol_is_1": last_symbol_is_1(corrupt),
            "clean_count_balance": count_balance(clean), "corrupt_count_balance": count_balance(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} count_balance pairs")
    return pairs


def audit_count_balance_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["clean_all_balanced"] = all(p["clean_count_balance"] for p in pairs)
    a["corrupt_all_unbalanced"] = all(not p["corrupt_count_balance"] for p in pairs)
    a["first_last_symbol_untouched"] = all(
        p["clean_first_symbol_is_0"] == p["corrupt_first_symbol_is_0"] == True and
        p["clean_last_symbol_is_1"] == p["corrupt_last_symbol_is_1"] == True for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_one_position"] and a["clean_all_balanced"] and
        a["corrupt_all_unbalanced"] and a["first_last_symbol_untouched"]
    )
    return a


def null_control_pairs(n_pairs, rng, k_range=(3, 20)):
    """TRUE minimal pair via length-extension: clean=(01)^k, corrupt=
    (01)^(k+1) -- clean is an exact prefix of corrupt. Both genuinely
    valid, same true label -- the closest achievable true pair given this
    task's zero within-length content redundancy (see module docstring)."""
    pairs = []
    for _ in range(n_pairs):
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        clean = build_valid_positive(k)
        corrupt = build_valid_positive(k + 1)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "clean_length": len(clean), "corrupt_length": len(corrupt),
            "clean_is_positive": is_positive_local(clean), "corrupt_is_positive": is_positive_local(corrupt),
        })
    return pairs


def audit_null_control_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["corrupt_is_exactly_two_tokens_longer"] = all(p["corrupt_length"] == p["clean_length"] + 2 for p in pairs)
    a["clean_is_prefix_of_corrupt"] = all(p["corrupt"][:p["clean_length"]] == p["clean"] for p in pairs)
    a["both_genuinely_valid_same_label"] = all(p["clean_is_positive"] and p["corrupt_is_positive"] for p in pairs)
    a["target_property_isolated"] = (
        a["corrupt_is_exactly_two_tokens_longer"] and a["clean_is_prefix_of_corrupt"] and a["both_genuinely_valid_same_label"]
    )
    a["note"] = (
        "TRUE pair (both genuinely valid, same true label) -- NOT a same-length minimal pair (the "
        "task has zero within-length content redundancy, verified in the task audit and re-"
        "confirmed here: the user's suggested position-i/mirror-swap does NOT preserve validity). "
        "This is the smallest achievable true pair: clean is an exact prefix of corrupt, differing "
        "by exactly one appended '01' block."
    )
    return a


# ---------------------------------------------------------------------------
# condition 2: collapses into condition 1's features, tested COMBINED
# ---------------------------------------------------------------------------

def combined_shortcut_violation_pairs(n_pairs, rng, k_range=(4, 20)):
    pairs = []
    for _ in range(n_pairs):
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        clean = build_valid_positive(k)
        n = len(clean)
        corrupt = [str(int(rng.integers(0, 2))) for _ in range(n)]
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": n,
            "clean_features": {name: fn(clean) for name, fn in SHORTCUT_FEATURES.items()},
            "corrupt_features": {name: fn(corrupt) for name, fn in SHORTCUT_FEATURES.items()},
        })
    return pairs


def audit_combined_shortcut_violation_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    a["clean_all_features_hold"] = all(all(p["clean_features"].values()) for p in pairs)
    n_features_broken_per_pair = [sum(1 for v in p["corrupt_features"].values() if not v) for p in pairs]
    a["mean_n_features_broken_in_corrupt"] = float(np.mean(n_features_broken_per_pair))
    a["n_features_broken_histogram"] = {str(k): n_features_broken_per_pair.count(k) for k in range(5)}
    a["target_property_isolated"] = a["same_length"] and a["clean_all_features_hold"]
    a["note"] = (
        "Condition 2 collapses into Condition 1's feature set per the task's simplicity (no "
        "additional trivial aggregate feature identified beyond the four already tested) -- this "
        "pair type tests the COMBINED effect of an independently-random corruption, which "
        "typically breaks MULTIPLE shortcut features simultaneously (reported directly, not "
        "assumed), matching how real scrambled negatives naturally look."
    )
    return a


# ---------------------------------------------------------------------------
# condition 3: target-computation null test, swept low/mid/high position
# (using the audit's own empirical early-concentrated distribution)
# ---------------------------------------------------------------------------

def target_computation_pairs(n_pairs, rng, k_range=(6, 20), fixed_tertile=None, max_tries=MAX_TRIES):
    pairs, seen, tries_total = [], set(), 0
    while len(pairs) < n_pairs and tries_total < n_pairs * max_tries:
        target_tertile = fixed_tertile if fixed_tertile is not None else TERTILES[int(rng.integers(0, 3))]
        found = False
        for _ in range(max_tries):
            tries_total += 1
            k = int(rng.integers(k_range[0], k_range[1] + 1))
            clean = build_valid_positive(k)
            n = len(clean)
            candidates = list(range(1, n - 2))  # avoid the boundary pair (0,1) and (n-2,n-1)
            denom = max(1, n - 3)
            def frac_of(j):
                return (j - 1) / denom
            bucket = [j for j in candidates if categorize_tertile(min(1.0, max(0.0, frac_of(j)))) == target_tertile]
            if not bucket:
                continue
            i = int(rng.choice(bucket))
            corrupt = clean[:]
            corrupt[i], corrupt[i + 1] = corrupt[i + 1], corrupt[i]  # adjacent positions always differ
            key = (tuple(clean), tuple(corrupt))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "clean": clean, "corrupt": corrupt, "length": n,
                "swap_indices": [i, i + 1], "swap_position_relative": frac_of(i),
                "swap_position_tertile": target_tertile,
                "clean_all_four_features": all(fn(clean) for fn in SHORTCUT_FEATURES.values()),
                "corrupt_all_four_features": all(fn(corrupt) for fn in SHORTCUT_FEATURES.values()),
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
    a["clean_all_four_shortcut_features_hold"] = all(p["clean_all_four_features"] for p in pairs)
    a["corrupt_all_four_shortcut_features_ALSO_hold"] = all(p["corrupt_all_four_features"] for p in pairs)
    a["clean_all_positive_corrupt_all_negative"] = all(
        p["clean_is_positive"] and not p["corrupt_is_positive"] for p in pairs)
    tertile_hist = {t: sum(1 for p in pairs if p["swap_position_tertile"] == t) for t in TERTILES}
    a["swap_position_tertile_histogram"] = tertile_hist
    a["target_property_isolated"] = (
        a["same_length"] and a["differs_at_exactly_two_positions"] and
        a["clean_all_four_shortcut_features_hold"] and a["corrupt_all_four_shortcut_features_ALSO_hold"] and
        a["clean_all_positive_corrupt_all_negative"]
    )
    a["note"] = (
        "All four shortcut features are IDENTICAL between clean and corrupt for every pair "
        "(structurally guaranteed: an adjacent swap can't change length, boundary values when kept "
        "away from position 0/n-1, or the content multiset). Position has NO magnitude "
        "interpretation for this task (unlike the LSB-first arithmetic tasks) -- an early>late gap "
        "would match the audit's own early-concentrated natural violation distribution, not "
        "magnitude sensitivity."
    )
    return a


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260718)

    lpp = length_parity_pairs(30, rng)
    fsp = first_symbol_pairs(30, rng)
    lsp = last_symbol_pairs(30, rng)
    cbp = count_balance_pairs(30, rng)
    ncp = null_control_pairs(30, rng)
    csv_pairs = combined_shortcut_violation_pairs(30, rng)
    tcp = target_computation_pairs(30, rng)

    audits = {
        "length_parity_pairs": audit_length_parity_pairs(lpp),
        "first_symbol_pairs": audit_first_symbol_pairs(fsp),
        "last_symbol_pairs": audit_last_symbol_pairs(lsp),
        "count_balance_pairs": audit_count_balance_pairs(cbp),
        "null_control_pairs": audit_null_control_pairs(ncp),
        "combined_shortcut_violation_pairs": audit_combined_shortcut_violation_pairs(csv_pairs),
        "target_computation_pairs": audit_target_computation_pairs(tcp),
    }
    for name, a in audits.items():
        print(f"=== {name} ===", flush=True)
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    out = {
        "task": "repeat-01",
        "description": (
            "Phase 3 counterfactual pair construction + audit for repeat-01 causal patching, all "
            "three approved conditions. NO PATCHING RUN. null_control_pairs uses a length-extension "
            "TRUE pair (NOT the user's originally-suggested position-mirror swap, which was verified "
            "empirically to break validity -- this task has zero within-length content redundancy, "
            "the strictest of any task in this pilot). Condition 2 collapses into Condition 1's "
            "feature set (combined test), per the task's simplicity. Condition 3 targets the "
            "115-example genuinely-shortcut-blind population (not viol_repeat_01's hard set), swept "
            "by tertile with the discipline that early>late reflects the natural violation "
            "distribution, not magnitude sensitivity."
        ),
        "condition_1_structural": {
            "length_parity_pairs": {"audit": audits["length_parity_pairs"], "sample_pairs": lpp[:3]},
            "first_symbol_pairs": {"audit": audits["first_symbol_pairs"], "sample_pairs": fsp[:3]},
            "last_symbol_pairs": {"audit": audits["last_symbol_pairs"], "sample_pairs": lsp[:3]},
            "count_balance_pairs": {"audit": audits["count_balance_pairs"], "sample_pairs": cbp[:3]},
            "null_control_pairs_TRUE_PAIR_LENGTH_EXTENSION": {"audit": audits["null_control_pairs"], "sample_pairs": ncp[:3]},
        },
        "condition_2_collapses_into_condition_1_combined_test": {
            "combined_shortcut_violation_pairs": {"audit": audits["combined_shortcut_violation_pairs"], "sample_pairs": csv_pairs[:3]},
        },
        "condition_3_target_computation_null_test_shortcut_blind_115": {
            "target_computation_pairs": {"audit": audits["target_computation_pairs"], "sample_pairs": tcp[:3]},
        },
    }
    out_path = RESULTS / "phase3_repeat01_counterfactual_design.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
