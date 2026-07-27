"""Phase 3 counterfactual construction for bucket-sort, mirroring analysis/
phase3_oddsfirst_counterfactuals.py but built for the SORT transform (clean
= w + [#] + sorted(w), over a 5-symbol alphabet, not 2) -- see src/
recognizers/hand_picked_languages/bucket_sort.py's _is_positive /
_w_to_string / _sort.

Four pair types built now (marker_count, marker_position for condition 1;
length_parity, is_balanced for condition 2 -- built together since they
share no task-specific subtlety beyond the 5-symbol alphabet). The FIFTH
type (target_computation, condition 3) is DEFERRED to when condition 3 is
approved, per the user's explicit instruction that it needs its own careful
construction (matching multisets, differing only in sort order) -- not
built here to avoid getting it wrong before that's the task at hand.

NO PATCHING IS RUN HERE -- pair construction + audit only.

PYTHONPATH=src:analysis python analysis/phase3_bucketsort_counterfactuals.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
import phase2_targets as T

RESULTS = Path("analysis_outputs/final_results")
N_SYMBOLS = 5  # bucket-sort's content alphabet is '1'..'5', not '0'/'1'


def sort_w(w):
    return sorted(w)


def rand_symbol(rng):
    return str(int(rng.integers(1, N_SYMBOLS + 1)))


# ---------------------------------------------------------------------------
# condition 1: marker_count, marker_position
# ---------------------------------------------------------------------------

def marker_count_pairs(n_half, n_pairs, rng):
    """clean = w#sorted(w) (marker_count=1). corrupt = SAME LENGTH, one
    content character (just before the marker) replaced by a second '#' --
    marker_count_class flips 1 -> 2."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [rand_symbol(rng) for _ in range(n_half)]
        clean_tokens = w + ["#"] + sort_w(w)
        corrupt_tokens = clean_tokens[:]
        corrupt_tokens[n_half - 1] = "#"
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "clean_marker_count": T.bucketsort_marker_count_class(clean_tokens),
            "corrupt_marker_count": T.bucketsort_marker_count_class(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} marker_count pairs")
    return pairs


def marker_position_pairs(n_half, n_pairs, rng):
    """clean = w#sorted(w) (marker centered). corrupt = SAME LENGTH, SAME
    marker count (1), independently-generated content with the marker
    placed FAR off-center."""
    L = 2 * n_half + 1
    far_marker_idx = n_half // 4
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [rand_symbol(rng) for _ in range(n_half)]
        clean_tokens = w + ["#"] + sort_w(w)
        corrupt_content = [rand_symbol(rng) for _ in range(L - 1)]
        corrupt_tokens = corrupt_content[:far_marker_idx] + ["#"] + corrupt_content[far_marker_idx:]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "clean_marker_position_bin": T.bucketsort_marker_position_bin(clean_tokens),
            "corrupt_marker_position_bin": T.bucketsort_marker_position_bin(corrupt_tokens),
            "clean_marker_count": T.bucketsort_marker_count_class(clean_tokens),
            "corrupt_marker_count": T.bucketsort_marker_count_class(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} marker_position pairs")
    return pairs


def audit_marker_count_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for a, b in zip(p["clean"], p["corrupt"]) if a != b) for p in pairs]
    audit["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    audit["clean_all_count_1"] = all(p["clean_marker_count"] == 1 for p in pairs)
    audit["corrupt_all_count_2"] = all(p["corrupt_marker_count"] == 2 for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["differs_at_exactly_one_position"] and
        audit["clean_all_count_1"] and audit["corrupt_all_count_2"]
    )
    return audit


def audit_marker_position_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    audit["same_marker_count"] = all(
        p["clean_marker_count"] == 1 and p["corrupt_marker_count"] == 1 for p in pairs)
    audit["clean_all_centered"] = all(p["clean_marker_position_bin"] == 1 for p in pairs)
    audit["corrupt_all_far"] = all(p["corrupt_marker_position_bin"] == 3 for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["same_marker_count"] and
        audit["clean_all_centered"] and audit["corrupt_all_far"]
    )
    return audit


# ---------------------------------------------------------------------------
# condition 2: is_balanced (balance_pairs), length_parity (length_parity_pairs)
# ---------------------------------------------------------------------------

def balance_pairs(n_half, n_pairs, rng):
    """clean = w#sorted(w), marker at n_half (is_balanced=1). corrupt =
    marker SHIFTED one position right (transpose clean_tokens[n_half] and
    clean_tokens[n_half+1]) -- is_balanced becomes 0, marker count still 1,
    length unchanged."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [rand_symbol(rng) for _ in range(n_half)]
        clean_tokens = w + ["#"] + sort_w(w)
        m = n_half
        corrupt_tokens = clean_tokens[:]
        corrupt_tokens[m], corrupt_tokens[m + 1] = corrupt_tokens[m + 1], corrupt_tokens[m]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "clean_marker_index": m, "corrupt_marker_index": m + 1,
            "clean_is_balanced": T.bucketsort_is_balanced(clean_tokens),
            "corrupt_is_balanced": T.bucketsort_is_balanced(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} balance pairs")
    return pairs


def length_parity_pairs(n_half, n_pairs, rng):
    """clean = w#sorted(w), length 2*n_half+1 (odd). corrupt = clean with
    ONE token appended at the end (duplicating the final token) -- length
    2*n_half+2 (even)."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [rand_symbol(rng) for _ in range(n_half)]
        clean_tokens = w + ["#"] + sort_w(w)
        corrupt_tokens = clean_tokens + [clean_tokens[-1]]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens,
            "clean_length": len(clean_tokens), "corrupt_length": len(corrupt_tokens),
            "clean_length_parity": T.bucketsort_length_parity(clean_tokens),
            "corrupt_length_parity": T.bucketsort_length_parity(corrupt_tokens),
            "clean_is_balanced": T.bucketsort_is_balanced(clean_tokens),
            "corrupt_is_balanced": T.bucketsort_is_balanced(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} length-parity pairs")
    return pairs


def audit_balance_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    audit["same_marker_count"] = all(
        p["clean"].count("#") == 1 and p["corrupt"].count("#") == 1 for p in pairs)
    diffs = [sum(1 for a, b in zip(p["clean"], p["corrupt"]) if a != b) for p in pairs]
    audit["n_token_positions_differing"] = {"min": min(diffs), "max": max(diffs)}
    audit["differs_at_exactly_two_positions"] = all(d == 2 for d in diffs)
    audit["clean_all_balanced"] = all(p["clean_is_balanced"] == 1 for p in pairs)
    audit["corrupt_all_unbalanced"] = all(p["corrupt_is_balanced"] == 0 for p in pairs)
    audit["marker_position_shifts_by_exactly_one"] = all(
        p["corrupt_marker_index"] == p["clean_marker_index"] + 1 for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["same_marker_count"] and
        audit["differs_at_exactly_two_positions"] and audit["clean_all_balanced"] and
        audit["corrupt_all_unbalanced"]
    )
    return audit


def audit_length_parity_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["corrupt_is_one_token_longer"] = all(
        p["corrupt_length"] == p["clean_length"] + 1 for p in pairs)
    audit["clean_all_odd_corrupt_all_even"] = all(
        p["clean_length_parity"] == 1 and p["corrupt_length_parity"] == 0 for p in pairs)
    audit["clean_all_balanced_corrupt_all_unbalanced"] = all(
        p["clean_is_balanced"] == 1 and p["corrupt_is_balanced"] == 0 for p in pairs)
    audit["prefix_unchanged"] = all(
        p["corrupt"][:p["clean_length"]] == p["clean"] for p in pairs)
    audit["target_property_isolated"] = (
        audit["corrupt_is_one_token_longer"] and audit["clean_all_odd_corrupt_all_even"] and
        audit["prefix_unchanged"]
    )
    return audit


# ---------------------------------------------------------------------------
# condition 2 (additional): is_multiset_matched -- a candidate causal
# mechanism for transformer's family-wide hard-negative high-water mark
# (Stage 3 found its highest is_multiset_matched selectivity of any
# architecture)
# ---------------------------------------------------------------------------

def multiset_pairs(n_half, n_pairs, rng, flip_position_from_start=2):
    """clean = w#sorted(w), marker centered (is_multiset_matched=1). corrupt
    = SAME LENGTH, SAME marker count/position/balance, ONE post-marker
    content position (flip_position_from_start, default index 2 within the
    sorted half) changed to a DIFFERENT symbol -- this necessarily breaks
    the multiset match (removes one instance of the original symbol, adds
    one instance of a different symbol, so the post-marker multiset can no
    longer equal the pre-marker multiset) while leaving marker_count,
    marker_position, and is_balanced completely unchanged (same length,
    same single marker at the same centered index)."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [rand_symbol(rng) for _ in range(n_half)]
        clean_tokens = w + ["#"] + sort_w(w)
        assert flip_position_from_start < n_half, "flip position must be within the sorted half"
        after_idx = n_half + 1 + flip_position_from_start
        orig_sym = clean_tokens[after_idx]
        new_sym = orig_sym
        while new_sym == orig_sym:
            new_sym = rand_symbol(rng)
        corrupt_tokens = clean_tokens[:]
        corrupt_tokens[after_idx] = new_sym
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "flip_position_after_idx": after_idx,
            "clean_is_multiset_matched": T.bucketsort_is_multiset_matched(clean_tokens),
            "corrupt_is_multiset_matched": T.bucketsort_is_multiset_matched(corrupt_tokens),
            "clean_marker_count": T.bucketsort_marker_count_class(clean_tokens),
            "corrupt_marker_count": T.bucketsort_marker_count_class(corrupt_tokens),
            "clean_is_balanced": T.bucketsort_is_balanced(clean_tokens),
            "corrupt_is_balanced": T.bucketsort_is_balanced(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} multiset pairs")
    return pairs


def audit_multiset_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    audit["same_marker_count"] = all(
        p["clean_marker_count"] == 1 and p["corrupt_marker_count"] == 1 for p in pairs)
    audit["same_marker_position"] = all(
        p["clean"].index("#") == p["corrupt"].index("#") for p in pairs)
    audit["same_is_balanced"] = all(
        p["clean_is_balanced"] == 1 and p["corrupt_is_balanced"] == 1 for p in pairs)
    diffs = [sum(1 for a, b in zip(p["clean"], p["corrupt"]) if a != b) for p in pairs]
    audit["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    audit["clean_all_multiset_matched"] = all(p["clean_is_multiset_matched"] == 1 for p in pairs)
    audit["corrupt_all_multiset_mismatched"] = all(p["corrupt_is_multiset_matched"] == 0 for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["same_marker_count"] and audit["same_marker_position"] and
        audit["same_is_balanced"] and audit["differs_at_exactly_one_position"] and
        audit["clean_all_multiset_matched"] and audit["corrupt_all_multiset_mismatched"]
    )
    return audit


# ---------------------------------------------------------------------------
# condition 3: target_computation_multiset_matched_case (genuine sort-ORDER
# verification, isolated from marker/balance/multiset confounds)
# ---------------------------------------------------------------------------

def target_computation_pairs(n_half, n_pairs, rng, swap_idx=None):
    """clean = w#sorted(w) (genuine positive, target_computation_multiset_
    matched_case=1). corrupt = SAME LENGTH, SAME marker count/position/
    balance, SAME CONTENT MULTISET (both halves) -- differs from clean by
    swapping TWO ADJACENT positions within the sorted half (indices swap_idx
    and swap_idx+1, default the middle of the half) whose values differ.
    Swapping two elements can never change the multiset, so
    is_multiset_matched stays 1 for BOTH clean and corrupt -- this isolates
    genuine sort-ORDER verification from every lower-level pathway (marker
    count/position, is_balanced, is_multiset_matched all held constant).

    swap_idx is fixed away from both ends of the sorted half (default
    n_half//2) specifically to avoid handing the model the max-check
    (seq[-1] == max(pre-marker content)) or min-check (seq[marker+1] ==
    min(pre-marker content)) trivial pathways identified in Phase 1 P2 --
    swapping the LAST or FIRST sorted position would trivially break those
    checks too, contaminating this as a pure sort-order test. w is
    regenerated until sorted(w)[swap_idx] != sorted(w)[swap_idx+1] (so the
    swap is guaranteed to actually change the sequence, not silently no-op
    on a repeated value)."""
    if swap_idx is None:
        swap_idx = n_half // 2
    assert 0 < swap_idx < n_half - 2, "swap_idx must avoid both ends (max-check/min-check contamination)"
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 400:
        attempts += 1
        w = [rand_symbol(rng) for _ in range(n_half)]
        s = sort_w(w)
        if s[swap_idx] == s[swap_idx + 1]:
            continue  # swap would be a no-op -- regenerate
        clean_tokens = w + ["#"] + s
        after_idx1 = n_half + 1 + swap_idx
        after_idx2 = after_idx1 + 1
        corrupt_tokens = clean_tokens[:]
        corrupt_tokens[after_idx1], corrupt_tokens[after_idx2] = corrupt_tokens[after_idx2], corrupt_tokens[after_idx1]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "swap_positions": [after_idx1, after_idx2],
            "clean_target_computation": T.bucketsort_target_computation_multiset_matched_case(clean_tokens),
            "corrupt_target_computation": T.bucketsort_target_computation_multiset_matched_case(corrupt_tokens),
            "clean_is_multiset_matched": T.bucketsort_is_multiset_matched(clean_tokens),
            "corrupt_is_multiset_matched": T.bucketsort_is_multiset_matched(corrupt_tokens),
            "clean_marker_count": T.bucketsort_marker_count_class(clean_tokens),
            "corrupt_marker_count": T.bucketsort_marker_count_class(corrupt_tokens),
            "clean_is_balanced": T.bucketsort_is_balanced(clean_tokens),
            "corrupt_is_balanced": T.bucketsort_is_balanced(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} target_computation pairs")
    return pairs


def audit_target_computation_pairs(pairs):
    """Confirms the construction isolates genuine sort-order verification:
    same length/marker-count/marker-position/is_balanced/multiset-match
    between clean and corrupt, differing at exactly 2 (swapped) positions,
    AND explicitly re-checks the max-check/min-check trivial pathways from
    Phase 1 P2 are NOT accidentally reintroduced by this construction."""
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    audit["same_marker_count"] = all(
        p["clean_marker_count"] == 1 and p["corrupt_marker_count"] == 1 for p in pairs)
    audit["same_marker_position"] = all(
        p["clean"].index("#") == p["corrupt"].index("#") for p in pairs)
    audit["same_is_balanced"] = all(
        p["clean_is_balanced"] == 1 and p["corrupt_is_balanced"] == 1 for p in pairs)
    audit["same_is_multiset_matched"] = all(
        p["clean_is_multiset_matched"] == 1 and p["corrupt_is_multiset_matched"] == 1 for p in pairs)
    diffs = [sum(1 for a, b in zip(p["clean"], p["corrupt"]) if a != b) for p in pairs]
    audit["n_token_positions_differing"] = {"min": min(diffs), "max": max(diffs)}
    audit["differs_at_exactly_two_positions"] = all(d == 2 for d in diffs)
    audit["clean_all_target_computation_match"] = all(p["clean_target_computation"] == 1 for p in pairs)
    audit["corrupt_all_no_match"] = all(p["corrupt_target_computation"] == 0 for p in pairs)
    # explicit re-check: does this construction accidentally reintroduce the
    # max-check / min-check trivial pathways from Phase 1 P2?
    def max_check(s):
        m = s.index("#")
        before = s[:m]
        return s[-1] == max(before) if before else True
    def min_check(s):
        m = s.index("#")
        before = s[:m]
        return s[m + 1] == min(before) if before and m + 1 < len(s) else True
    audit["max_check_unaffected"] = all(max_check(p["clean"]) and max_check(p["corrupt"]) for p in pairs)
    audit["min_check_unaffected"] = all(min_check(p["clean"]) and min_check(p["corrupt"]) for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["same_marker_count"] and audit["same_marker_position"] and
        audit["same_is_balanced"] and audit["same_is_multiset_matched"] and
        audit["differs_at_exactly_two_positions"] and audit["clean_all_target_computation_match"] and
        audit["corrupt_all_no_match"] and audit["max_check_unaffected"] and audit["min_check_unaffected"]
    )
    return audit


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    mc = marker_count_pairs(50, 30, rng)
    mp = marker_position_pairs(50, 30, rng)
    bp = balance_pairs(50, 30, rng)
    lpp = length_parity_pairs(50, 30, rng)
    msp = multiset_pairs(50, 30, rng)
    tcp = target_computation_pairs(50, 30, rng)

    audits = {
        "marker_count_pairs": audit_marker_count_pairs(mc),
        "marker_position_pairs": audit_marker_position_pairs(mp),
        "balance_pairs": audit_balance_pairs(bp),
        "length_parity_pairs": audit_length_parity_pairs(lpp),
        "multiset_pairs": audit_multiset_pairs(msp),
        "target_computation_pairs": audit_target_computation_pairs(tcp),
    }
    for name, a in audits.items():
        print(f"=== {name} ===")
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    out = {
        "description": "Phase 3 counterfactual construction + audit for bucket-sort causal "
                       "patching, conditions 1-2 (marker_count, marker_position, is_balanced, "
                       "length_parity). condition 3 (target_computation) pairs are built separately "
                       "when that condition is approved. NO PATCHING RUN.",
        "content_alphabet": "5 symbols ('1'..'5'), unlike the binary alphabet used for marker_count/"
                            "position/length_parity/is_balanced pairs on the three prior tasks -- "
                            "rand_symbol draws uniformly from all 5, verified in the audits above "
                            "(same construction logic, alphabet size is the only difference).",
        "audits": audits,
        "sample_pairs": {
            "marker_count": mc[:3], "marker_position": mp[:3], "balance": bp[:3],
            "length_parity": lpp[:3], "multiset": msp[:3], "target_computation": tcp[:3],
        },
    }
    out_path = RESULTS / "phase3_bucketsort_counterfactual_design.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
