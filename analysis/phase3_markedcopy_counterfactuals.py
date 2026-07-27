"""Phase 3 counterfactual construction for marked-copy, mirroring analysis/
phase3_counterfactual_design.py's marked-reversal section but built for the
IDENTITY transform (clean = w + [#] + w, not w + [#] + reversed(w)) -- see
src/recognizers/hand_picked_languages/marked_copy.py's _is_positive /
_w_to_string for the ground-truth grammar this must match.

Five pair types, one per Stage 2/3 probe target:
  - marker_count_pairs      (condition 1)
  - marker_position_pairs   (condition 1)
  - balance_pairs           (condition 2, is_balanced)
  - length_parity_pairs     (condition 2, length_parity)
  - copy_match_pairs        (condition 3, copy_match_balanced_case; the
                              marked-copy analog of markedrev_mirror_pairs)

NO PATCHING IS RUN HERE -- pair construction + audit only.

PYTHONPATH=src:analysis python analysis/phase3_markedcopy_counterfactuals.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
import phase2_targets as T

RESULTS = Path("analysis_outputs/final_results")


# ---------------------------------------------------------------------------
# condition 1: marker_count, marker_position
# ---------------------------------------------------------------------------

def marker_count_pairs(n_half, n_pairs, rng):
    """clean = w#w (marker_count=1). corrupt = SAME LENGTH, one content
    character (just before the marker) replaced by a second '#' --
    marker_count_class flips 1 -> 2. Differs at exactly 1 token position."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + w
        corrupt_tokens = clean_tokens[:]
        corrupt_tokens[n_half - 1] = "#"
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "clean_marker_count": T.markedcopy_marker_count_class(clean_tokens),
            "corrupt_marker_count": T.markedcopy_marker_count_class(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} marker_count pairs")
    return pairs


def marker_position_pairs(n_half, n_pairs, rng):
    """clean = w#w (marker centered). corrupt = SAME LENGTH, SAME marker
    count (1), independently-generated content with the marker placed FAR
    off-center (relative deviation 0.38, past the 0.05 near-center
    threshold) -- marker_position_bin flips 1 (exact-center) -> 3 (far)."""
    L = 2 * n_half + 1
    far_marker_idx = n_half // 4
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + w
        corrupt_content = [str(int(rng.integers(0, 2))) for _ in range(L - 1)]
        corrupt_tokens = corrupt_content[:far_marker_idx] + ["#"] + corrupt_content[far_marker_idx:]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "clean_marker_position_bin": T.markedcopy_marker_position_bin(clean_tokens),
            "corrupt_marker_position_bin": T.markedcopy_marker_position_bin(corrupt_tokens),
            "clean_marker_count": T.markedcopy_marker_count_class(clean_tokens),
            "corrupt_marker_count": T.markedcopy_marker_count_class(corrupt_tokens),
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
    """clean = w#w, marker at n_half (is_balanced=1). corrupt = marker
    SHIFTED one position right (transpose clean_tokens[n_half] and
    clean_tokens[n_half+1]) -- is_balanced becomes 0 (n_before=n_half+1,
    n_after=n_half-1), marker count still 1, length unchanged. Exactly 2
    token positions differ (the transposed pair)."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + w
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
            "clean_is_balanced": T.markedcopy_is_balanced(clean_tokens),
            "corrupt_is_balanced": T.markedcopy_is_balanced(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} balance pairs")
    return pairs


def length_parity_pairs(n_half, n_pairs, rng):
    """clean = w#w, length 2*n_half+1 (odd). corrupt = clean with ONE token
    appended at the end (duplicating the final token) -- length 2*n_half+2
    (even). Differs from clean by exactly one ADDED token; every original
    token unchanged."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + w
        corrupt_tokens = clean_tokens + [clean_tokens[-1]]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens,
            "clean_length": len(clean_tokens), "corrupt_length": len(corrupt_tokens),
            "clean_length_parity": T.markedcopy_length_parity(clean_tokens),
            "corrupt_length_parity": T.markedcopy_length_parity(corrupt_tokens),
            "clean_is_balanced": T.markedcopy_is_balanced(clean_tokens),
            "corrupt_is_balanced": T.markedcopy_is_balanced(corrupt_tokens),
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
# condition 3: copy_match_balanced_case (null test; marked-copy analog of
# markedrev_mirror_pairs)
# ---------------------------------------------------------------------------

def copy_match_pairs(n_half, n_pairs, rng, flip_position_from_start=2):
    """clean = genuine positive w#w, marker at n_half (copy_match_balanced_
    case=1). corrupt = SAME LENGTH, SAME marker count/position, differs from
    clean at EXACTLY ONE content position in the FIRST HALF
    (flip_position_from_start, default index 2 -- far from both marker and
    readout). Flipping one first-half bit breaks exactly the ONE copy pair
    (i=flip_position_from_start: seq[i] vs seq[m+1+i], per markedcopy_copy_
    match_balanced_case's independent per-i check), leaving every other
    token and every other copy pair unchanged. clean: copy_match_balanced_
    case=1 (accept). corrupt: copy_match_balanced_case=0 (hard negative:
    marker count/position/balance all correct, content wrong)."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + w
        assert flip_position_from_start < n_half, "flip position must be in the first half"
        orig_bit = clean_tokens[flip_position_from_start]
        new_bit = "0" if orig_bit == "1" else "1"
        corrupt_tokens = clean_tokens[:]
        corrupt_tokens[flip_position_from_start] = new_bit
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "marker_index": n_half, "flip_position": flip_position_from_start,
            "clean_copy_match": T.markedcopy_copy_match_balanced_case(clean_tokens),
            "corrupt_copy_match": T.markedcopy_copy_match_balanced_case(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} copy_match pairs")
    return pairs


def audit_copy_match_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    audit["same_marker_count"] = all(
        p["clean"].count("#") == 1 and p["corrupt"].count("#") == 1 for p in pairs)
    audit["same_marker_position"] = all(
        p["clean"].index("#") == p["corrupt"].index("#") for p in pairs)
    diffs = [sum(1 for a, b in zip(p["clean"], p["corrupt"]) if a != b) for p in pairs]
    audit["n_token_positions_differing"] = {"min": min(diffs), "max": max(diffs)}
    audit["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    audit["flip_position_is_the_only_difference"] = all(
        p["clean"][p["flip_position"]] != p["corrupt"][p["flip_position"]] and
        all(p["clean"][i] == p["corrupt"][i] for i in range(len(p["clean"])) if i != p["flip_position"])
        for p in pairs
    )
    audit["clean_all_copy_match"] = all(p["clean_copy_match"] == 1 for p in pairs)
    audit["corrupt_all_no_match"] = all(p["corrupt_copy_match"] == 0 for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["same_marker_count"] and audit["same_marker_position"] and
        audit["differs_at_exactly_one_position"] and audit["clean_all_copy_match"] and
        audit["corrupt_all_no_match"]
    )
    return audit


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    mc = marker_count_pairs(50, 30, rng)
    mp = marker_position_pairs(50, 30, rng)
    bp = balance_pairs(50, 30, rng)
    lpp = length_parity_pairs(50, 30, rng)
    cmp_ = copy_match_pairs(50, 30, rng)

    audits = {
        "marker_count_pairs": audit_marker_count_pairs(mc),
        "marker_position_pairs": audit_marker_position_pairs(mp),
        "balance_pairs": audit_balance_pairs(bp),
        "length_parity_pairs": audit_length_parity_pairs(lpp),
        "copy_match_pairs": audit_copy_match_pairs(cmp_),
    }
    for name, a in audits.items():
        print(f"=== {name} ===")
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    out = {
        "description": "Phase 3 counterfactual construction + audit for marked-copy causal "
                       "patching. NO PATCHING RUN.",
        "audits": audits,
        "sample_pairs": {
            "marker_count": mc[:3], "marker_position": mp[:3], "balance": bp[:3],
            "length_parity": lpp[:3], "copy_match": cmp_[:3],
        },
    }
    out_path = RESULTS / "phase3_markedcopy_counterfactual_design.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
