"""Missing-duplicate-string Phase 3 (bounded) pilot: counterfactual pair
construction + audit for the three approved conditions. Mirrors bucket-
sort's phase3_bucketsort_counterfactuals.py / Dyck-2-3's phase3_dyck_
task_audit.py construction discipline: each pair type isolates ONE
candidate feature (verified identical/flipped between clean and corrupt as
appropriate), audited explicitly, no patching run here.

Uses phase3_missdup_task_audit.py's confirmed grammar: alphabet is binary
(0/1) + blank '_'; genuine positives are w+w (length 2*n_half) with exactly
one '1'-position blanked (never '0' -- fixed constant fill); the true
target computation is an order-sensitive per-position halves-match check,
NOT an aggregate/majority-count task.

CONDITION 1 (structural features -- length parity, blank position, first/
last symbols): shallow shortcuts a model could exploit WITHOUT any content-
preserving carry.
  length_parity_pairs: clean=valid positive (even length). corrupt=clean
    with one token duplicated-and-appended (odd length) -- blank count/
    position/content otherwise identical, matches bucket-sort's length_
    parity_pairs recipe exactly.
  blank_position_pairs: clean=valid positive with its blank constrained to
    the EARLY length-tertile (fixed-target-tertile-then-retry, BSF1
    discipline). corrupt=INDEPENDENTLY RANDOM binary content of the SAME
    LENGTH with a single blank forced into the LATE tertile -- mirrors
    bucket-sort's marker_position_pairs (independently-generated content,
    only length + blank-count held fixed) since blank position alone,
    divorced from content validity, is the feature under test.
  first_last_symbol_pairs (NULL CONTROL): two INDEPENDENT valid positives
    of the same length with blanks in the same (mid) tertile, whose first
    tokens differ (one starts '0', one starts '1'). Both are equally VALID
    (same label) -- a real target-computation-driven model should show NO
    behavioral gap here; a model spuriously keying off the first symbol
    would.

CONDITION 2 (aggregate-route -- per-half symbol counts, the CRITICAL test):
  count_mismatch_pairs: clean=valid positive (per-half count-of-'1's equal
    by construction). corrupt=SAME length, SAME blank position/count, ONE
    non-blank bit within the half NOT containing the blank flipped (0<->1)
    -- this necessarily makes that half's count-of-'1's differ from the
    other half's (aggregate mismatch) AND breaks true positional equality
    at the flipped index (for a binary alphabet the two effects are
    inherently coupled for a single-bit change -- documented explicitly,
    not hidden -- unlike bucket-sort's 5-symbol alphabet, a binary flip
    cannot change one property without the other; only a same-value-class
    SWAP, used in Condition 3, decouples them).

CONDITION 3 (revised per phase3_missdup_task_audit.json's finding that
"which symbol is duplicated" is not constructible -- the blank fill is
always the fixed constant '1'; redefined as a POSITION-OF-MISMATCH null
test): position_mismatch_pairs: clean=valid positive. corrupt=SAME length,
SAME blank position/count, per-half counts of '1's MATCHING BETWEEN CLEAN
AND CORRUPT (i.e. the aggregate route would call corrupt "positive" too) --
built via a same-half SWAP of two differing positions (preserves that
half's count exactly, breaks true positional match at the two swapped
indices only), swapping position swept EARLY/MID/LATE via the SAME fixed-
target-tertile-then-retry discipline as bucket-sort's sort-order
counterfactuals and Dyck-2-3's LIFO-violation construction.

PYTHONPATH=src:analysis python analysis/phase3_missdup_counterfactual_design.py
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
# shared construction helpers (mirror the real generator exactly)
# ---------------------------------------------------------------------------

def sample_w(n_half, rng):
    w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
    i = int(rng.integers(0, n_half))
    w[i] = "1"
    return w


def build_valid_positive(n_half, rng):
    """Returns (tokens_with_blank, blank_idx). Matches _w_to_string exactly:
    s = w+w, choose blank_idx uniformly among ALL positions in s holding
    '1' (not just within one half)."""
    w = sample_w(n_half, rng)
    s = w + w
    idx1 = [i for i, a in enumerate(s) if a == "1"]
    blank_idx = int(rng.choice(idx1))
    s[blank_idx] = "_"
    return s, blank_idx


def is_positive_local(seq):
    if len(seq) % 2 != 0 or seq.count("_") != 1:
        return False
    i = seq.index("_")
    half = len(seq) // 2
    ss = list(seq)
    ss[i] = "1"
    return ss[:half] == ss[half:]


def per_half_counts(seq, blank_fill="1"):
    half = len(seq) // 2
    ss = list(seq)
    if "_" in ss:
        ss[ss.index("_")] = blank_fill
    return ss[:half].count("1"), ss[half:].count("1")


# ---------------------------------------------------------------------------
# condition 1: structural features
# ---------------------------------------------------------------------------

def length_parity_pairs(n_half, n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        clean, blank_idx = build_valid_positive(n_half, rng)
        corrupt = clean + ["0"]  # append a fixed content token, never the blank itself
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt,
            "clean_length": len(clean), "corrupt_length": len(corrupt),
            "clean_length_parity_even": len(clean) % 2 == 0,
            "corrupt_length_parity_even": len(corrupt) % 2 == 0,
            "clean_blank_count": clean.count("_"), "corrupt_blank_count": corrupt.count("_"),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} length_parity pairs")
    return pairs


def audit_length_parity_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["corrupt_is_one_token_longer"] = all(p["corrupt_length"] == p["clean_length"] + 1 for p in pairs)
    a["clean_all_even_corrupt_all_odd"] = all(
        p["clean_length_parity_even"] and not p["corrupt_length_parity_even"] for p in pairs)
    a["blank_count_unchanged"] = all(p["clean_blank_count"] == p["corrupt_blank_count"] == 1 for p in pairs)
    a["prefix_unchanged"] = all(p["corrupt"][:p["clean_length"]] == p["clean"] for p in pairs)
    a["target_property_isolated"] = (
        a["corrupt_is_one_token_longer"] and a["clean_all_even_corrupt_all_odd"] and
        a["blank_count_unchanged"] and a["prefix_unchanged"]
    )
    return a


def blank_position_pairs(n_half, n_pairs, rng):
    L = 2 * n_half
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        clean, blank_idx = build_valid_positive(n_half, rng)
        if categorize_tertile(blank_idx / (L - 1)) != "low":
            continue
        # independently random content, blank forced into the "high" tertile
        corrupt = [str(int(rng.integers(0, 2))) for _ in range(L)]
        high_positions = [i for i in range(L) if categorize_tertile(i / (L - 1)) == "high"]
        far_idx = int(rng.choice(high_positions))
        corrupt[far_idx] = "_"
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": L,
            "clean_blank_idx": blank_idx, "corrupt_blank_idx": far_idx,
            "clean_blank_tertile": categorize_tertile(blank_idx / (L - 1)),
            "corrupt_blank_tertile": categorize_tertile(far_idx / (L - 1)),
            "clean_blank_count": clean.count("_"), "corrupt_blank_count": corrupt.count("_"),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} blank_position pairs")
    return pairs


def audit_blank_position_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(p["clean"].__len__() == p["corrupt"].__len__() for p in pairs)
    a["same_blank_count"] = all(p["clean_blank_count"] == p["corrupt_blank_count"] == 1 for p in pairs)
    a["clean_all_low_tertile"] = all(p["clean_blank_tertile"] == "low" for p in pairs)
    a["corrupt_all_high_tertile"] = all(p["corrupt_blank_tertile"] == "high" for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["same_blank_count"] and a["clean_all_low_tertile"] and a["corrupt_all_high_tertile"]
    )
    return a


def first_last_symbol_pairs(n_half, n_pairs, rng):
    """NULL CONTROL: two independently-sampled genuine positives, same
    length, blanks both in the mid tertile, first tokens deliberately
    different (one '0' one '1'). BOTH are valid (label identical) -- tests
    whether patching first-symbol identity alone spuriously flips a
    model's decision despite no true label difference."""
    L = 2 * n_half
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        clean, blank_idx_c = build_valid_positive(n_half, rng)
        if categorize_tertile(blank_idx_c / (L - 1)) != "mid" or clean[0] != "0":
            continue
        corrupt, blank_idx_x = build_valid_positive(n_half, rng)
        if categorize_tertile(blank_idx_x / (L - 1)) != "mid" or corrupt[0] != "1":
            continue
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": L,
            "clean_first_symbol": clean[0], "corrupt_first_symbol": corrupt[0],
            "clean_last_symbol": clean[-1], "corrupt_last_symbol": corrupt[-1],
            "clean_is_positive": is_positive_local(clean), "corrupt_is_positive": is_positive_local(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} first_last_symbol pairs")
    return pairs


def audit_first_last_symbol_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    a["first_symbols_differ"] = all(p["clean_first_symbol"] != p["corrupt_first_symbol"] for p in pairs)
    a["both_genuinely_valid_same_label"] = all(p["clean_is_positive"] and p["corrupt_is_positive"] for p in pairs)
    a["target_property_isolated"] = a["same_length"] and a["first_symbols_differ"] and a["both_genuinely_valid_same_label"]
    a["note"] = "null-control pair type: expect ~0 behavioral gap since both members share the true label"
    return a


def null_control_rebuild_pairs(n_half, n_pairs, rng):
    """TRUE minimal-pair rebuild of the first/last-symbol null control (the
    original version independently resampled clean/corrupt, confounding the
    comparison with unrelated content differences -- see phase3_missdup_
    condition1.json's flagged caveat). Flips position 0 AND its mirror
    position (half+0) SIMULTANEOUSLY, holding every other position
    (including the blank) identical. Since clean[0] must equal clean[half]
    for clean to be a genuine positive (neither is the blank), flipping both
    to the other binary value keeps corrupt ALSO a genuine positive with the
    SAME true label -- a real 2-position minimal pair, not two independently
    drawn strings."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        clean, blank_idx = build_valid_positive(n_half, rng)
        half = len(clean) // 2
        if blank_idx == 0 or blank_idx == half:
            continue  # keep the flipped positions clear of the blank
        assert clean[0] == clean[half]
        v0 = clean[0]
        newv = "0" if v0 == "1" else "1"
        corrupt = clean[:]
        corrupt[0] = newv
        corrupt[half] = newv
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": len(clean),
            "clean_blank_idx": blank_idx, "corrupt_blank_idx": blank_idx,
            "clean_first_symbol": clean[0], "corrupt_first_symbol": corrupt[0],
            "clean_is_positive": is_positive_local(clean), "corrupt_is_positive": is_positive_local(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} null_control_rebuild pairs")
    return pairs


def audit_null_control_rebuild_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    a["same_blank_position"] = all(p["clean_blank_idx"] == p["corrupt_blank_idx"] for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_two_positions"] = all(d == 2 for d in diffs)
    a["first_symbols_differ"] = all(p["clean_first_symbol"] != p["corrupt_first_symbol"] for p in pairs)
    a["both_genuinely_valid_same_label"] = all(p["clean_is_positive"] and p["corrupt_is_positive"] for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["same_blank_position"] and a["differs_at_exactly_two_positions"] and
        a["first_symbols_differ"] and a["both_genuinely_valid_same_label"]
    )
    a["note"] = (
        "TRUE minimal pair (differs at exactly the 2 mirrored positions being tested, all else "
        "including the blank identical) -- unlike the original null control's independently-"
        "resampled construction, any observed gap here cannot be attributed to unrelated content "
        "differences."
    )
    return a


# ---------------------------------------------------------------------------
# condition 2: aggregate-route (per-half symbol counts) -- critical test
# ---------------------------------------------------------------------------

def count_mismatch_pairs(n_half, n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * MAX_TRIES:
        attempts += 1
        clean, blank_idx = build_valid_positive(n_half, rng)
        L = len(clean)
        half = L // 2
        blank_half = 0 if blank_idx < half else 1
        flip_half = 1 - blank_half  # flip in the half NOT containing the blank
        flip_local_idx = int(rng.integers(0, half))
        flip_idx = flip_half * half + flip_local_idx
        if clean[flip_idx] == "_":
            continue
        corrupt = clean[:]
        corrupt[flip_idx] = "0" if corrupt[flip_idx] == "1" else "1"
        c1_first, c1_second = per_half_counts(clean)
        c2_first, c2_second = per_half_counts(corrupt)
        key = (tuple(clean), tuple(corrupt))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean, "corrupt": corrupt, "length": L, "flip_idx": flip_idx,
            "clean_blank_idx": blank_idx, "corrupt_blank_idx": blank_idx,
            "clean_first_half_count1": c1_first, "clean_second_half_count1": c1_second,
            "corrupt_first_half_count1": c2_first, "corrupt_second_half_count1": c2_second,
            "clean_is_positive": is_positive_local(clean), "corrupt_is_positive": is_positive_local(corrupt),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} count_mismatch pairs")
    return pairs


def audit_count_mismatch_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    a["same_blank_position"] = all(p["clean_blank_idx"] == p["corrupt_blank_idx"] for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    a["clean_counts_match_between_halves"] = all(
        p["clean_first_half_count1"] == p["clean_second_half_count1"] for p in pairs)
    a["corrupt_counts_differ_between_halves"] = all(
        p["corrupt_first_half_count1"] != p["corrupt_second_half_count1"] for p in pairs)
    a["clean_all_positive_corrupt_all_negative"] = all(
        p["clean_is_positive"] and not p["corrupt_is_positive"] for p in pairs)
    a["target_property_isolated"] = (
        a["same_length"] and a["same_blank_position"] and a["differs_at_exactly_one_position"] and
        a["clean_counts_match_between_halves"] and a["corrupt_counts_differ_between_halves"] and
        a["clean_all_positive_corrupt_all_negative"]
    )
    a["coupling_caveat"] = (
        "For this task's BINARY alphabet, a single-bit flip necessarily changes BOTH the aggregate "
        "per-half count AND the local positional match at that index simultaneously -- unlike bucket-"
        "sort's 5-symbol alphabet (where a SWAP can change order without changing counts), a binary "
        "flip cannot decouple the two. This is documented, not hidden: Condition 2 tests whether "
        "patching activations that causally carry the COUNT signal flips the decision; Condition 3's "
        "SWAP construction is what isolates pure positional sensitivity from aggregate counting."
    )
    return a


# ---------------------------------------------------------------------------
# condition 3: position-of-mismatch null test (revised), swept early/mid/late
# ---------------------------------------------------------------------------

def position_mismatch_pairs(n_half, n_pairs, rng, max_tries=MAX_TRIES, fixed_tertile=None):
    """Fixed-target-tertile-then-retry (BSF1 discipline). Swap two DIFFERING
    positions within the half NOT containing the blank, preserving that
    half's count-of-'1's exactly (a swap can't change a multiset) while
    breaking true positional equality at the two swapped indices.

    fixed_tertile: if given (one of TERTILES), every pair targets that same
    tertile (used to generate an EXACT N per tertile, rather than a random
    per-pair draw that only roughly balances across tertiles)."""
    pairs = []
    seen = set()
    tries_total = 0
    while len(pairs) < n_pairs and tries_total < n_pairs * max_tries:
        target_tertile = fixed_tertile if fixed_tertile is not None else TERTILES[int(rng.integers(0, 3))]
        found = False
        for _ in range(max_tries):
            tries_total += 1
            clean, blank_idx = build_valid_positive(n_half, rng)
            L = len(clean)
            half = L // 2
            blank_half = 0 if blank_idx < half else 1
            swap_half = 1 - blank_half
            base = swap_half * half
            local_vals = clean[base:base + half]
            candidates = [j for j in range(half - 1) if local_vals[j] != local_vals[j + 1]]
            if not candidates:
                continue
            denom = max(1, half - 2)
            def frac_of(j):
                return j / denom
            bucket = [j for j in candidates if categorize_tertile(min(1.0, frac_of(j))) == target_tertile]
            if not bucket:
                continue
            j = int(rng.choice(bucket))
            i1, i2 = base + j, base + j + 1
            corrupt = clean[:]
            corrupt[i1], corrupt[i2] = corrupt[i2], corrupt[i1]
            key = (tuple(clean), tuple(corrupt))
            if key in seen:
                continue
            seen.add(key)
            c1_first, c1_second = per_half_counts(clean)
            c2_first, c2_second = per_half_counts(corrupt)
            pairs.append({
                "clean": clean, "corrupt": corrupt, "length": L,
                "clean_blank_idx": blank_idx, "corrupt_blank_idx": blank_idx,
                "swap_indices": [i1, i2],
                "swap_position_relative": frac_of(j), "swap_position_tertile": target_tertile,
                "clean_first_half_count1": c1_first, "clean_second_half_count1": c1_second,
                "corrupt_first_half_count1": c2_first, "corrupt_second_half_count1": c2_second,
                "clean_is_positive": is_positive_local(clean), "corrupt_is_positive": is_positive_local(corrupt),
            })
            found = True
            break
        if not found:
            continue
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} position_mismatch pairs")
    return pairs


def audit_position_mismatch_pairs(pairs):
    a = {"n_pairs": len(pairs)}
    a["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    a["same_blank_position"] = all(p["clean_blank_idx"] == p["corrupt_blank_idx"] for p in pairs)
    diffs = [sum(1 for x, y in zip(p["clean"], p["corrupt"]) if x != y) for p in pairs]
    a["differs_at_exactly_two_positions"] = all(d == 2 for d in diffs)
    a["clean_counts_match_between_halves"] = all(
        p["clean_first_half_count1"] == p["clean_second_half_count1"] for p in pairs)
    a["corrupt_counts_ALSO_match_between_halves"] = all(
        p["corrupt_first_half_count1"] == p["corrupt_second_half_count1"] for p in pairs)
    a["corrupt_counts_equal_clean_counts"] = all(
        p["corrupt_first_half_count1"] == p["clean_first_half_count1"] and
        p["corrupt_second_half_count1"] == p["clean_second_half_count1"] for p in pairs)
    a["clean_all_positive_corrupt_all_negative"] = all(
        p["clean_is_positive"] and not p["corrupt_is_positive"] for p in pairs)
    tertile_hist = {t: sum(1 for p in pairs if p["swap_position_tertile"] == t) for t in TERTILES}
    hmin, hmax = min(tertile_hist.values()), max(tertile_hist.values())
    a["swap_position_tertile_histogram"] = tertile_hist
    a["tertiles_within_20_percent_of_each_other"] = bool(hmin >= 0.8 * hmax)
    a["target_property_isolated"] = (
        a["same_length"] and a["same_blank_position"] and a["differs_at_exactly_two_positions"] and
        a["clean_counts_match_between_halves"] and a["corrupt_counts_ALSO_match_between_halves"] and
        a["corrupt_counts_equal_clean_counts"] and a["clean_all_positive_corrupt_all_negative"]
    )
    a["note"] = (
        "aggregate route (per-half count-of-1s) is IDENTICAL between clean and corrupt for every "
        "pair -- a model relying purely on aggregate counts cannot distinguish them at all, yet the "
        "true label flips. Swept across early/mid/late swap position to test whether the position "
        "of the mismatch (not just its existence) causally matters."
    )
    return a


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260718)
    n_half = 40  # long enough to give real early/mid/late separation

    lpp = length_parity_pairs(n_half, 30, rng)
    bpp = blank_position_pairs(n_half, 30, rng)
    flsp = first_last_symbol_pairs(n_half, 30, rng)
    cmp_ = count_mismatch_pairs(n_half, 30, rng)
    pmp = position_mismatch_pairs(n_half, 30, rng)

    audits = {
        "length_parity_pairs": audit_length_parity_pairs(lpp),
        "blank_position_pairs": audit_blank_position_pairs(bpp),
        "first_last_symbol_pairs": audit_first_last_symbol_pairs(flsp),
        "count_mismatch_pairs": audit_count_mismatch_pairs(cmp_),
        "position_mismatch_pairs": audit_position_mismatch_pairs(pmp),
    }
    for name, a in audits.items():
        print(f"=== {name} ===", flush=True)
        print(json.dumps(a, indent=2, default=str))
        assert a["target_property_isolated"], f"{name} failed to isolate its target property"

    out = {
        "task": "missing-duplicate-string",
        "description": (
            "Phase 3 counterfactual pair construction + audit for missing-duplicate-string causal "
            "patching, all three approved conditions. NO PATCHING RUN -- construction + isolation "
            "audit only. Condition 3 redefined per phase3_missdup_task_audit.json's finding (the "
            "blank's fill value is a fixed constant '1', so 'which symbol is duplicated' is not "
            "constructible; redefined as a position-of-mismatch null test, matching bucket-sort's "
            "sort-order / Dyck-2-3's LIFO-violation same-aggregate-different-order construction "
            "discipline)."
        ),
        "n_half_used_for_samples": n_half,
        "condition_1_structural": {
            "length_parity_pairs": {"audit": audits["length_parity_pairs"], "sample_pairs": lpp[:3]},
            "blank_position_pairs": {"audit": audits["blank_position_pairs"], "sample_pairs": bpp[:3]},
            "first_last_symbol_pairs_NULL_CONTROL": {"audit": audits["first_last_symbol_pairs"], "sample_pairs": flsp[:3]},
        },
        "condition_2_aggregate_route_critical_test": {
            "count_mismatch_pairs": {"audit": audits["count_mismatch_pairs"], "sample_pairs": cmp_[:3]},
        },
        "condition_3_position_of_mismatch_null_test_revised": {
            "position_mismatch_pairs": {"audit": audits["position_mismatch_pairs"], "sample_pairs": pmp[:3]},
        },
    }
    out_path = RESULTS / "phase3_missdup_counterfactual_design.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
