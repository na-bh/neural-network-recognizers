"""Phase 3 counterfactual construction for Dyck-2-3 (bounded, patching-only
pilot). Task-specific throughout -- no marker/arithmetic-family assumptions
inherited (this is a stack-carrier task: LIFO type-matching + a bounded
depth-3 counter, per phase3_dyck_task_audit.json).

All constructions below are built to hold as many of the audit's four
listed shortcut features (bracket_count_parity, per_type_bracket_counts,
sequence_length_parity, first_last_check) fixed as mathematically possible,
isolating exactly the ONE property each condition targets -- verified by
audit functions, not assumed.

Condition 1 (bracket_count_parity): flip ONE bracket's ROLE (opener<->
closer) at an INTERIOR position -- preserves length (hence length_parity)
and first/last tokens exactly; necessarily breaks total open/close balance
AND that bracket's own per-type balance (mathematically forced: if every
per-type count were still balanced, total would be too -- so breaking
total parity requires breaking at least one per-type count. Documented,
not hidden).

Condition 2 (per_type_bracket_counts): RELABEL a single closer's type
(e.g. ')0' -> ')1') at an interior position -- preserves length, TOTAL
open/close counts exactly (hence bracket_count_parity is UNCHANGED, a
clean isolation from condition 1), and first/last tokens; breaks per-type
balance for the two affected types only.

Condition 3a (target_computation, type-mismatch): swap the TYPE LABELS of
two closers whose corresponding opens are both still on the stack at that
point (mirrors the audit's own counterexample, generalized to a chosen
nesting depth level) -- preserves EVERY listed shortcut feature exactly
(total counts, per-type counts, length, first/last all identical), breaks
ONLY LIFO stack-order matching.

Condition 3b (target_computation, depth-exceeded): INSERT a matched
(open, close) pair of an arbitrary type at a point where nesting depth is
already 3 -- momentarily reaches depth 4 (invalid, even though the pair is
properly matched and the string ends balanced) -- preserves every listed
shortcut feature (adds 1 to that type's open AND close count together, so
totals/per-type/parity all unchanged; inserts in the interior, so first/
last unchanged; length changes by +2, an EVEN delta, so length_parity is
ALSO unchanged) -- architecturally distinct from 3a (a bounded-counter
violation, not a LIFO-order violation).

NO PATCHING IS RUN HERE -- pair construction + audit only.

PYTHONPATH=src:analysis python analysis/phase3_dyck_counterfactuals.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_dyck_task_audit import (
    is_valid_dyck, SHORTCUT_FEATURES, K, M,
)

RESULTS = Path("analysis_outputs/final_results")


# ---------------------------------------------------------------------------
# genuine-positive generator (stack-based random walk, depth capped at M)
# ---------------------------------------------------------------------------

def random_valid_dyck(rng, min_len=20, max_len=60, close_bias=0.5):
    """Random walk: at each step, if stack is non-empty and (stack is at
    depth M or a coin flip says close), close the top of the stack;
    otherwise open a random type (if depth < M). Continues until the
    target length is (approximately) reached AND the stack is empty."""
    target_len = rng.integers(min_len, max_len + 1)
    # target_len must be even (every valid string has even length) -- round up
    if target_len % 2 == 1:
        target_len += 1
    seq = []
    stack = []
    while True:
        remaining = target_len - len(seq)
        if remaining == 0 and not stack:
            break
        must_close = len(stack) == M or remaining <= len(stack)
        if stack and (must_close or rng.random() < close_bias):
            t = stack.pop()
            seq.append(f"){t}")
        else:
            t = int(rng.integers(0, K))
            stack.append(t)
            seq.append(f"({t}")
    return seq


def find_positions_at_depth(seq, depth_level):
    """Returns indices right after which the running stack depth equals
    depth_level (i.e., positions of OPEN tokens whose push makes depth ==
    depth_level), each paired with the index of its matching closer."""
    stack = []  # list of (type, open_index)
    matches = []
    for i, tok in enumerate(seq):
        if tok.startswith("("):
            stack.append((tok[1:], i))
            if len(stack) == depth_level:
                open_idx = i
        elif tok.startswith(")"):
            t, oi = stack.pop()
            if len(stack) + 1 == depth_level and oi == open_idx:
                matches.append((oi, i))
    return matches


def audit_features_match(clean, corrupted):
    return {name: (fn(clean) == fn(corrupted)) for name, fn in SHORTCUT_FEATURES.items()}


# ---------------------------------------------------------------------------
# condition 1: bracket_count_parity
# ---------------------------------------------------------------------------

def parity_pairs(n_pairs, rng, position_frac):
    """Flip one CLOSER at relative position `position_frac` (0,1) into an
    OPENER of the same type. Requires the chosen position to be an interior
    closer (not first/last token)."""
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 300:
        attempts += 1
        clean = random_valid_dyck(rng)
        closer_idx = [i for i, t in enumerate(clean) if t.startswith(")") and 0 < i < len(clean) - 1]
        if not closer_idx:
            continue
        target_i = min(closer_idx, key=lambda i: abs(i / (len(clean) - 1) - position_frac))
        corrupted = clean[:]
        t = clean[target_i][1:]
        corrupted[target_i] = f"({t}"
        pairs.append({"clean": clean, "corrupted": corrupted, "flip_index": target_i,
                      "flip_index_relative": target_i / (len(clean) - 1)})
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} parity pairs")
    return pairs


def audit_parity_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupted"]) for p in pairs)
    audit["clean_all_valid"] = all(is_valid_dyck(p["clean"], M) for p in pairs)
    audit["corrupted_all_invalid"] = all(not is_valid_dyck(p["corrupted"], M) for p in pairs)
    feat_matches = [audit_features_match(p["clean"], p["corrupted"]) for p in pairs]
    audit["length_parity_preserved"] = all(f["sequence_length_parity"] for f in feat_matches)
    audit["first_last_preserved"] = all(f["first_last_check"] for f in feat_matches)
    audit["bracket_count_parity_CHANGES_as_intended"] = all(
        SHORTCUT_FEATURES["bracket_count_parity"](p["clean"]) and not SHORTCUT_FEATURES["bracket_count_parity"](p["corrupted"])
        for p in pairs
    )
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["clean_all_valid"] and audit["corrupted_all_invalid"]
        and audit["length_parity_preserved"] and audit["first_last_preserved"]
        and audit["bracket_count_parity_CHANGES_as_intended"]
    )
    return audit


# ---------------------------------------------------------------------------
# condition 2: per_type_bracket_counts (bracket_count_parity held fixed)
# ---------------------------------------------------------------------------

def per_type_pairs(n_pairs, rng, position_frac):
    """Relabel one interior closer's TYPE (e.g. )0 -> )1), leaving every
    other token untouched. Requires at least 2 bracket types (K>=2,
    satisfied for dyck-2-3) and an interior closer."""
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 300:
        attempts += 1
        clean = random_valid_dyck(rng)
        closer_idx = [i for i, t in enumerate(clean) if t.startswith(")") and 0 < i < len(clean) - 1]
        if not closer_idx:
            continue
        target_i = min(closer_idx, key=lambda i: abs(i / (len(clean) - 1) - position_frac))
        old_t = int(clean[target_i][1:])
        new_t = (old_t + 1) % K
        corrupted = clean[:]
        corrupted[target_i] = f"){new_t}"
        pairs.append({"clean": clean, "corrupted": corrupted, "relabel_index": target_i,
                      "relabel_index_relative": target_i / (len(clean) - 1),
                      "old_type": old_t, "new_type": new_t})
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} per_type pairs")
    return pairs


def audit_per_type_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupted"]) for p in pairs)
    audit["clean_all_valid"] = all(is_valid_dyck(p["clean"], M) for p in pairs)
    audit["corrupted_all_invalid"] = all(not is_valid_dyck(p["corrupted"], M) for p in pairs)
    feat_matches = [audit_features_match(p["clean"], p["corrupted"]) for p in pairs]
    audit["length_parity_preserved"] = all(f["sequence_length_parity"] for f in feat_matches)
    audit["first_last_preserved"] = all(f["first_last_check"] for f in feat_matches)
    audit["bracket_count_parity_preserved (condition 1 isolation)"] = all(f["bracket_count_parity"] for f in feat_matches)
    audit["per_type_bracket_counts_CHANGES_as_intended"] = all(
        SHORTCUT_FEATURES["per_type_bracket_counts"](p["clean"]) and not SHORTCUT_FEATURES["per_type_bracket_counts"](p["corrupted"])
        for p in pairs
    )
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["clean_all_valid"] and audit["corrupted_all_invalid"]
        and audit["length_parity_preserved"] and audit["first_last_preserved"]
        and audit["bracket_count_parity_preserved (condition 1 isolation)"]
        and audit["per_type_bracket_counts_CHANGES_as_intended"]
    )
    return audit


# ---------------------------------------------------------------------------
# condition 3a: target_computation, type-mismatch (all 4 shortcuts preserved)
# ---------------------------------------------------------------------------

def target_computation_type_mismatch_pairs(n_pairs, rng, depth_level, max_tries_per=300):
    """At a chosen nesting depth level, find two closers whose opens are
    BOTH on the stack simultaneously at some point (i.e., nested, not
    sequential) and swap their TYPE LABELS -- preserves every listed
    shortcut feature (same multiset of tokens, just two closer positions
    exchanged), breaks LIFO order matching. Mirrors the audit's own
    counterexample, generalized to depth_level."""
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * max_tries_per:
        attempts += 1
        clean = random_valid_dyck(rng)
        matches = find_positions_at_depth(clean, depth_level)
        if len(matches) < 2:
            continue
        # need two DIFFERENT-type closers to make a genuine swap
        idx_pair = None
        for a in range(len(matches)):
            for b in range(a + 1, len(matches)):
                oa, ca = matches[a]
                ob, cb = matches[b]
                if clean[ca][1:] != clean[cb][1:]:
                    idx_pair = (ca, cb)
                    break
            if idx_pair:
                break
        if idx_pair is None:
            continue
        ca, cb = idx_pair
        corrupted = clean[:]
        corrupted[ca], corrupted[cb] = clean[cb], clean[ca]
        pairs.append({"clean": clean, "corrupted": corrupted, "swap_indices": [ca, cb],
                      "depth_level": depth_level})
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} type-mismatch pairs at depth {depth_level}")
    return pairs


def audit_type_mismatch_pairs(pairs, depth_level):
    audit = {"n_pairs": len(pairs), "depth_level": depth_level}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupted"]) for p in pairs)
    diffs = [sum(1 for a, b in zip(p["clean"], p["corrupted"]) if a != b) for p in pairs]
    audit["differs_at_exactly_two_positions"] = all(d == 2 for d in diffs)
    audit["clean_all_valid"] = all(is_valid_dyck(p["clean"], M) for p in pairs)
    audit["corrupted_all_invalid"] = all(not is_valid_dyck(p["corrupted"], M) for p in pairs)
    feat_matches = [audit_features_match(p["clean"], p["corrupted"]) for p in pairs]
    audit["all_four_shortcut_features_preserved"] = all(all(f.values()) for f in feat_matches)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["differs_at_exactly_two_positions"]
        and audit["clean_all_valid"] and audit["corrupted_all_invalid"]
        and audit["all_four_shortcut_features_preserved"]
    )
    return audit


# ---------------------------------------------------------------------------
# condition 3b: target_computation, depth-exceeded (all 4 shortcuts preserved)
# ---------------------------------------------------------------------------

def target_computation_depth_exceeded_pairs(n_pairs, rng, position_frac, max_tries=300):
    """Insert a matched (open,close) pair of a random type at a point where
    the running stack depth is already M -- momentarily reaches M+1, invalid
    even though the inserted pair is itself properly matched and the string
    ends balanced. Preserves total/per-type counts (the inserted pair adds
    1 to both that type's open and close count), length_parity (+2, even),
    and first/last (insertion is interior)."""
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * max_tries:
        attempts += 1
        clean = random_valid_dyck(rng)
        depth_m_positions = []
        stack_depth = 0
        for i, tok in enumerate(clean):
            if tok.startswith("("):
                stack_depth += 1
                if stack_depth == M:
                    depth_m_positions.append(i)
            elif tok.startswith(")"):
                stack_depth -= 1
        if not depth_m_positions:
            continue
        insert_after = min(depth_m_positions, key=lambda i: abs((i + 1) / len(clean) - position_frac))
        t = int(rng.integers(0, K))
        corrupted = clean[:insert_after + 1] + [f"({t}", f"){t}"] + clean[insert_after + 1:]
        pairs.append({"clean": clean, "corrupted": corrupted, "insert_after_index": insert_after,
                      "insert_position_relative": (insert_after + 1) / len(clean), "inserted_type": t})
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} depth-exceeded pairs")
    return pairs


def audit_depth_exceeded_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["length_delta_is_plus_2"] = all(len(p["corrupted"]) - len(p["clean"]) == 2 for p in pairs)
    audit["clean_all_valid"] = all(is_valid_dyck(p["clean"], M) for p in pairs)
    audit["corrupted_all_invalid"] = all(not is_valid_dyck(p["corrupted"], M) for p in pairs)
    audit["corrupted_all_valid_if_depth_unbounded"] = all(is_valid_dyck(p["corrupted"], None) for p in pairs)
    feat_matches = [audit_features_match(p["clean"], p["corrupted"]) for p in pairs]
    audit["all_four_shortcut_features_preserved"] = all(all(f.values()) for f in feat_matches)
    audit["target_property_isolated"] = (
        audit["length_delta_is_plus_2"] and audit["clean_all_valid"] and audit["corrupted_all_invalid"]
        and audit["corrupted_all_valid_if_depth_unbounded"] and audit["all_four_shortcut_features_preserved"]
    )
    return audit


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print("=== condition 1: parity_pairs (position sweep) ===")
    for label, frac in [("early", 0.2), ("mid", 0.5), ("late", 0.8)]:
        pairs = parity_pairs(10, rng, frac)
        audit = audit_parity_pairs(pairs)
        print(f"  [{label}] {json.dumps(audit)}")

    print("=== condition 2: per_type_pairs (position sweep) ===")
    for label, frac in [("early", 0.2), ("mid", 0.5), ("late", 0.8)]:
        pairs = per_type_pairs(10, rng, frac)
        audit = audit_per_type_pairs(pairs)
        print(f"  [{label}] {json.dumps(audit)}")

    print("=== condition 3a: type_mismatch_pairs (depth sweep) ===")
    for depth in [1, 2, 3]:
        pairs = target_computation_type_mismatch_pairs(10, rng, depth)
        audit = audit_type_mismatch_pairs(pairs, depth)
        print(f"  [depth={depth}] {json.dumps(audit)}")

    print("=== condition 3b: depth_exceeded_pairs (position sweep) ===")
    for label, frac in [("early", 0.2), ("mid", 0.5), ("late", 0.8)]:
        pairs = target_computation_depth_exceeded_pairs(10, rng, frac)
        audit = audit_depth_exceeded_pairs(pairs)
        print(f"  [{label}] {json.dumps(audit)}")
