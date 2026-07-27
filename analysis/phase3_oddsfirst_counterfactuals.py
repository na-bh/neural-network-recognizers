"""Phase 3 counterfactual construction for odds-first, mirroring analysis/
phase3_markedcopy_counterfactuals.py but built for the DEINTERLEAVE
transform (clean = w + [#] + odds_first(w), not w+[#]+w or w+[#]+reversed(w))
-- see src/recognizers/hand_picked_languages/odds_first.py's _is_positive /
_w_to_string / _odds_first for the ground-truth grammar this must match, and
analysis/phase2_targets.py's oddsfirst_indices/oddsfirst_target_computation_
balanced_case for the validated index map.

Five pair types, one per Stage 2/3 probe target:
  - marker_count_pairs      (condition 1)
  - marker_position_pairs   (condition 1)
  - balance_pairs           (condition 2, is_balanced)
  - length_parity_pairs     (condition 2, length_parity)
  - target_computation_pairs (condition 3, target_computation_balanced_case;
                               the odds-first analog of markedcopy_copy_
                               match_pairs / markedrev_mirror_pairs)

NO PATCHING IS RUN HERE -- pair construction + audit only.

PYTHONPATH=src:analysis python analysis/phase3_oddsfirst_counterfactuals.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
import phase2_targets as T

RESULTS = Path("analysis_outputs/final_results")


def odds_first(w):
    idx = T.oddsfirst_indices(len(w))
    return [w[i] for i in idx]


# ---------------------------------------------------------------------------
# condition 1: marker_count, marker_position
# ---------------------------------------------------------------------------

def marker_count_pairs(n_half, n_pairs, rng):
    """clean = w#odds_first(w) (marker_count=1). corrupt = SAME LENGTH, one
    content character (just before the marker) replaced by a second '#' --
    marker_count_class flips 1 -> 2."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + odds_first(w)
        corrupt_tokens = clean_tokens[:]
        corrupt_tokens[n_half - 1] = "#"
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "clean_marker_count": T.oddsfirst_marker_count_class(clean_tokens),
            "corrupt_marker_count": T.oddsfirst_marker_count_class(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} marker_count pairs")
    return pairs


def marker_position_pairs(n_half, n_pairs, rng):
    """clean = w#odds_first(w) (marker centered). corrupt = SAME LENGTH,
    SAME marker count (1), independently-generated content with the marker
    placed FAR off-center -- marker_position_bin flips 1 (exact-center) -> 3
    (far)."""
    L = 2 * n_half + 1
    far_marker_idx = n_half // 4
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + odds_first(w)
        corrupt_content = [str(int(rng.integers(0, 2))) for _ in range(L - 1)]
        corrupt_tokens = corrupt_content[:far_marker_idx] + ["#"] + corrupt_content[far_marker_idx:]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "clean_marker_position_bin": T.oddsfirst_marker_position_bin(clean_tokens),
            "corrupt_marker_position_bin": T.oddsfirst_marker_position_bin(corrupt_tokens),
            "clean_marker_count": T.oddsfirst_marker_count_class(clean_tokens),
            "corrupt_marker_count": T.oddsfirst_marker_count_class(corrupt_tokens),
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
    """clean = w#odds_first(w), marker at n_half (is_balanced=1). corrupt =
    marker SHIFTED one position right (transpose clean_tokens[n_half] and
    clean_tokens[n_half+1]) -- is_balanced becomes 0, marker count still 1,
    length unchanged."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + odds_first(w)
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
            "clean_is_balanced": T.oddsfirst_is_balanced(clean_tokens),
            "corrupt_is_balanced": T.oddsfirst_is_balanced(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} balance pairs")
    return pairs


def length_parity_pairs(n_half, n_pairs, rng):
    """clean = w#odds_first(w), length 2*n_half+1 (odd). corrupt = clean with
    ONE token appended at the end (duplicating the final token) -- length
    2*n_half+2 (even)."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + odds_first(w)
        corrupt_tokens = clean_tokens + [clean_tokens[-1]]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens,
            "clean_length": len(clean_tokens), "corrupt_length": len(corrupt_tokens),
            "clean_length_parity": T.oddsfirst_length_parity(clean_tokens),
            "corrupt_length_parity": T.oddsfirst_length_parity(corrupt_tokens),
            "clean_is_balanced": T.oddsfirst_is_balanced(clean_tokens),
            "corrupt_is_balanced": T.oddsfirst_is_balanced(corrupt_tokens),
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
# condition 3: target_computation_balanced_case (null test)
# ---------------------------------------------------------------------------

def target_computation_pairs(n_half, n_pairs, rng, flip_position_from_start=2):
    """clean = genuine positive w#odds_first(w), marker at n_half
    (target_computation_balanced_case=1). corrupt = SAME LENGTH, SAME marker
    count/position, differs from clean at EXACTLY ONE content position in
    the FIRST HALF (flip_position_from_start, default index 2). Flipping
    one first-half bit breaks exactly the ONE transform-pair that reads
    from that index (oddsfirst_indices(n) is a PERMUTATION of range(n), so
    each pre-marker index feeds exactly one post-marker position), leaving
    every other token and every other transform-pair unchanged."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + odds_first(w)
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
            "clean_target_computation": T.oddsfirst_target_computation_balanced_case(clean_tokens),
            "corrupt_target_computation": T.oddsfirst_target_computation_balanced_case(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} target_computation pairs")
    return pairs


def audit_target_computation_pairs(pairs):
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
    audit["clean_all_target_computation_match"] = all(p["clean_target_computation"] == 1 for p in pairs)
    audit["corrupt_all_no_match"] = all(p["corrupt_target_computation"] == 0 for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["same_marker_count"] and audit["same_marker_position"] and
        audit["differs_at_exactly_one_position"] and audit["clean_all_target_computation_match"] and
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
    tcp = target_computation_pairs(50, 30, rng)

    audits = {
        "marker_count_pairs": audit_marker_count_pairs(mc),
        "marker_position_pairs": audit_marker_position_pairs(mp),
        "balance_pairs": audit_balance_pairs(bp),
        "length_parity_pairs": audit_length_parity_pairs(lpp),
        "target_computation_pairs": audit_target_computation_pairs(tcp),
    }
    for name, a in audits.items():
        print(f"=== {name} ===")
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    out = {
        "description": "Phase 3 counterfactual construction + audit for odds-first causal "
                       "patching. NO PATCHING RUN.",
        "audits": audits,
        "sample_pairs": {
            "marker_count": mc[:3], "marker_position": mp[:3], "balance": bp[:3],
            "length_parity": lpp[:3], "target_computation": tcp[:3],
        },
    }
    out_path = RESULTS / "phase3_oddsfirst_counterfactual_design.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
