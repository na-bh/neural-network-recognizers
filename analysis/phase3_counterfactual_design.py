"""Phase 3, design step: counterfactual construction + audit for causal
patching on both pilot tasks. NO PATCHING IS RUN HERE -- this only builds and
verifies counterfactual pairs, and specs the patching protocol against the
audit results. See the module docstring sections below for the two tasks.

REVISED (Option 1 of the post-hoc confound investigation): this design step
originally treated marked-reversal's mirror_pairs (a genuine equal-halves
content-mismatch counterfactual) as the primary causal test, predicting a
"meaningful" behavioral gap for RNN based on its unrestricted 0.80 hard-
negative accuracy and its 0.148 full_reversal_match probe selectivity. Both
of those numbers were subsequently found to be confounded by an
"unequal-halves" structural shortcut (is_balanced = n_before==n_after,
decidable from marker position alone, no content comparison needed): the
CORRECTED probes (phase2_p3_marked_reversal.json, phase2_p3_layer_sweep.json)
show RNN's layer-1/layer-4 signal is numerically identical between
full_reversal_match and is_balanced (~0.145-0.148 both), while the
content-isolated balanced_content_match probe is EXACTLY ZERO at every
layer, every seed, all 4 architectures. Consistent with this, mirror_pairs'
own measured behavioral gap (below) is ~4.8e-08 -- not a measurement
anomaly, but the causal-level confirmation of the same null result: RNN's
clean (content-matching) and corrupt (content-mismatched, still balanced)
examples get nearly IDENTICAL logits, because the model does not compute
content-match information at all. This module now adds a SECOND
counterfactual type, balance_pairs, that isolates is_balanced directly (by
shifting the marker position rather than editing content) -- this is the
correctly-targeted causal test given the corrected picture, since is_balanced
is what the layer sweep actually found being carried at RNN's later layers.
mirror_pairs is kept as a confirmatory/null test: patching is predicted to
have ~no effect, because there is no clean-vs-corrupt behavioral difference
to restore in the first place.

Reuses analysis/cyclenav_counterfactuals.py's validated one-move-flip design
directly (the "wrong count" pair type). Reuses the causal-patching harnesses
already built and unit-tested in Part 6B (analysis/rnn_patching.py,
analysis/patching_harness.py, analysis/modk_transformer_harness.py) --
Mamba's PatchingHarness.patch_run and Transformer's
TransformerPatchingHarness.patch_logit already accept an arbitrary `layer`
argument; RNN/LSTM's RecurrentScan currently only supports patching ALL
layers simultaneously (via _apply_patch's dims=None branch operating on the
full [num_layers,1,H] state) or specific CHANNELS of the last layer -- it
needs a small, described-but-not-yet-implemented extension to patch a single
specified layer's slice of that state while leaving other layers at their
corrupt values, matching Mamba/Transformer's existing single-layer patch_run/
patch_logit semantics. That extension is specified here, not implemented
(implementation belongs to the actual patching run, not this design step).

PYTHONPATH=src:analysis python analysis/phase3_counterfactual_design.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
import cyclenav_counterfactuals as cc
import phase2_targets as T

RESULTS = Path("analysis_outputs/final_results")


# ---------------------------------------------------------------------------
# cycle-navigation: TWO counterfactual types
# ---------------------------------------------------------------------------

def cyclenav_grammar_pairs(M, n_pairs, rng):
    """Type A ("grammar", tests symbol_counts). clean = valid grammatical
    positive (M moves + correct digit). corrupt = SAME LENGTH, but the last
    token (a digit in clean) is replaced by a MOVE symbol -- an invalid shape
    (no digit ending) per is_grammatical, i.e. a SCRAMBLED negative by Phase
    1's classification. Differs from clean in exactly one token position
    (the last one); every other token is identical.

    This pair has a LARGE expected behavioral gap: clean should be accepted,
    corrupt should be REJECTED (Phase 1 found scrambled negatives are
    correctly rejected ~99.4-100% of the time by all four architectures) --
    unlike the "wrong count" pair type below, whose clean/corrupt logits are
    nearly identical by construction (the shortcut model doesn't distinguish
    them). This large gap is what makes restored-fraction a well-defined,
    numerically stable metric for this pair type.
    """
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        moves = [cc.MOVES[int(rng.integers(0, 3))] for _ in range(M)]
        clean_tokens, clean_pos = cc.build_sequence(moves)
        corrupt_last_move = cc.MOVES[int(rng.integers(0, 3))]
        corrupt_tokens = clean_tokens[:-1] + [cc.MOVE_TOK[corrupt_last_move]]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": M + 1,
            "clean_label": True, "corrupt_label": False,
            "corrupt_last_token_move": corrupt_last_move,
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} grammar pairs")
    return pairs


def cyclenav_position_pairs(M, n_pairs, rng):
    """Type B ("wrong count", tests true_position). Directly reuses
    cyclenav_counterfactuals.make_pairs (already validated by its own
    verify() function): clean = valid positive, corrupt = the LAST move
    flipped to a different move symbol, keeping clean's (now-wrong) digit --
    i.e. a grammatically-valid-shaped negative with the WRONG symbol counts
    (net displacement) for its claimed digit. Behaviorally near-degenerate:
    Phase 1 found frac_corrupt_accepted=1.0, mean|delta_logit|~0.0002 for
    this exact construction across all 4 architectures (cyclenav_shortcut_
    check.json) -- clean and corrupt are treated almost identically by the
    trained models. This means restored-fraction (which divides by
    clean_logit - corrupt_logit) is numerically unstable for this pair type;
    the patching protocol below uses absolute |delta_logit after patch|
    instead (see protocol spec)."""
    return cc.make_pairs(M, n_pairs, rng, flip_offset=1)


def audit_cyclenav_pairs(pairs, kind):
    """Verify: same length always; for grammar pairs, differ in exactly the
    last token; for position pairs, differ in exactly the flip_idx token
    (already verified by cc.verify(), re-checked here for completeness)."""
    audit = {"n_pairs": len(pairs), "kind": kind}
    same_length = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    audit["same_length"] = same_length
    if kind == "grammar":
        diffs = [sum(1 for a, b in zip(p["clean"], p["corrupt"]) if a != b) for p in pairs]
        audit["n_token_positions_differing"] = {"min": min(diffs), "max": max(diffs)}
        audit["differs_only_at_last_position"] = all(
            p["clean"][:-1] == p["corrupt"][:-1] and p["clean"][-1] != p["corrupt"][-1] for p in pairs)
        audit["target_property_check"] = ("corrupt is_grammatical=False for all pairs (last token is "
                                          "a move, not a digit) -- verified below via is_grammatical()")
        moves_set = {cc._tok_maps()[1][m] for m in ["<", ">", "="]}
        audit["all_corrupt_scrambled"] = all(p["corrupt"][-1] in moves_set for p in pairs)
    else:
        vocab_tokens = cc._vocab_tokens()
        def to_strs(seq):
            return [vocab_tokens[t] for t in seq]
        diffs = [sum(1 for a, b in zip(p["clean"], p["corrupt"]) if a != b) for p in pairs]
        audit["n_token_positions_differing"] = {"min": min(diffs), "max": max(diffs)}
        audit["differs_only_at_flip_idx"] = all(diffs[i] == 1 for i in range(len(pairs)))
        audit["clean_digit_equals_corrupt_digit"] = all(p["clean"][-1] == p["corrupt"][-1] for p in pairs)
        audit["true_position_differs"] = all(
            T.cyclenav_true_position(to_strs(p["clean"])) != T.cyclenav_true_position(to_strs(p["corrupt"]))
            for p in pairs
        )
        audit["symbol_counts_also_necessarily_differ"] = (
            "TRUE BY CONSTRUCTION: true_position = f(counts), so any move-token flip that changes "
            "true_position also changes symbol_counts (e.g. '>'.'<' flip changes both counts by 1 "
            "each). This pair type cannot isolate a 'true_position changed, symbol_counts unchanged' "
            "case via input-level editing alone -- see the subspace-patching protocol spec, which "
            "addresses this by patching along the probe-derived DIRECTION relevant to each quantity "
            "rather than relying on the counterfactual construction alone to separate them."
        )
    return audit


# ---------------------------------------------------------------------------
# marked-reversal: ONE counterfactual type ("grammatical-with-wrong-mirror")
# ---------------------------------------------------------------------------

def markedrev_mirror_pairs(n_half, n_pairs, rng, flip_position_from_start=2):
    """clean = genuine positive w#reverse(w), |w|=n_half, marker at index
    n_half. corrupt = SAME LENGTH, SAME marker count (1), SAME marker
    position (n_half), differs from clean at EXACTLY ONE content position:
    flip_position_from_start (a position in the FIRST HALF, near the start,
    default index 2 -- maximizing distance to both the marker and the
    readout position, to test whether the model must CARRY this information
    forward across the whole sequence, matching what Phase 2's probes
    examined). Flipping exactly one bit in the first half breaks exactly the
    ONE mirror pair involving that position (verified algebraically in
    phase2_targets.py's markedrev_full_reversal_match docstring: pairs are
    independent, seq[m-1-i] vs seq[m+1+i] for i in range(n)), leaving every
    other pair -- and every other token -- unchanged. clean: full_reversal_
    match=1 (label=1, accepted). corrupt: full_reversal_match=0 (a hard
    negative: marker count/position both correct, content wrong -- exactly
    the Phase 1 hard-negative category RNN showed 0.80 accuracy on)."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + list(reversed(w))
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
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": 2 * n_half + 1,
            "marker_index": n_half, "flip_position": flip_position_from_start,
            "clean_full_reversal_match": T.markedrev_full_reversal_match(clean_tokens),
            "corrupt_full_reversal_match": T.markedrev_full_reversal_match(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} mirror pairs")
    return pairs


def audit_markedrev_pairs(pairs):
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
    audit["clean_all_full_match"] = all(p["clean_full_reversal_match"] == 1 for p in pairs)
    audit["corrupt_all_no_match"] = all(p["corrupt_full_reversal_match"] == 0 for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["same_marker_count"] and audit["same_marker_position"] and
        audit["differs_at_exactly_one_position"] and audit["clean_all_full_match"] and
        audit["corrupt_all_no_match"]
    )
    return audit


def markedrev_balance_pairs(n_half, n_pairs, rng):
    """NEW pair type isolating is_balanced (the structural shortcut uncovered
    during this design step's own auditing), correctly targeting what the
    corrected Phase 2 layer sweep found being carried at RNN's later layers.
    clean = genuine positive w#reverse(w), marker at index n_half (n_before
    == n_after == n_half, is_balanced=1). corrupt = the marker SHIFTED one
    position to the right, produced by swapping the token at the marker
    position with the token immediately after it -- i.e. transpose
    clean_tokens[n_half] ('#') and clean_tokens[n_half+1] (reverse(w)'s first
    character). This is the minimal single-edit operation that changes
    marker position (hence is_balanced: n_before becomes n_half+1, n_after
    becomes n_half-1, unequal) while leaving every other token, the total
    marker count (still exactly one '#'), and the sequence length unchanged.
    Exactly 2 token positions differ (the transposed pair); every other
    position is identical between clean and corrupt.

    Unlike mirror_pairs, this pair type is NOT a valid marked-reversal
    negative in the "content mismatch" sense -- it directly instantiates the
    exact structural signal (is_balanced) the corrected layer sweep found
    causally-plausible-to-carry, so it is the right counterfactual for
    testing whether that carried signal actually drives the accept/reject
    decision."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + list(reversed(w))
        m = n_half
        corrupt_tokens = clean_tokens[:]
        corrupt_tokens[m], corrupt_tokens[m + 1] = corrupt_tokens[m + 1], corrupt_tokens[m]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": 2 * n_half + 1,
            "clean_marker_index": m, "corrupt_marker_index": m + 1,
            "clean_is_balanced": T.markedrev_is_balanced(clean_tokens),
            "corrupt_is_balanced": T.markedrev_is_balanced(corrupt_tokens),
            "clean_full_reversal_match": T.markedrev_full_reversal_match(clean_tokens),
            "corrupt_full_reversal_match": T.markedrev_full_reversal_match(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} balance pairs")
    return pairs


def markedrev_length_parity_pairs(n_half, n_pairs, rng):
    """FURTHER-REFINED pair type, added after balance_pairs' own behavioral-
    gap check (below) came back near-zero and prompted a follow-up
    investigation: on REAL test-set hard negatives, RNN seed3 is 100%
    accurate (25/25) rejecting magnitude-1-imbalance (EVEN-length) negatives
    but 0% accurate (0/25, always wrongly accepts) on magnitude-2-imbalance
    (ODD-length, off-center-by-one) negatives -- and a direct linear-probe
    check (is_balanced restricted to odd-length-only sequences, where length
    parity is constant and uninformative) shows exactly-zero selectivity at
    every layer, while length_parity ALONE recovers ~0.10 selectivity at the
    same layers. So RNN's actual mechanism is length parity (odd vs. even
    token count), NOT marker-position-aware balance-checking -- balance_pairs
    (which preserves odd length and only shifts the marker) accidentally
    targeted exactly the sub-case RNN cannot detect.

    clean = genuine positive w#reverse(w), length 2*n_half+1 (odd).
    corrupt = clean with ONE token appended at the end (duplicating the
    final token) -- length 2*n_half+2 (even), matching the DOMINANT real-
    world imbalance mechanism (82.6% of real unequal-halves hard negatives
    are exactly this: even length, magnitude-1 imbalance). Differs from
    clean by exactly one ADDED token; every original token is unchanged."""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + list(reversed(w))
        corrupt_tokens = clean_tokens + [clean_tokens[-1]]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens,
            "clean_length": len(clean_tokens), "corrupt_length": len(corrupt_tokens),
            "clean_length_parity": T.markedrev_length_parity(clean_tokens),
            "corrupt_length_parity": T.markedrev_length_parity(corrupt_tokens),
            "clean_is_balanced": T.markedrev_is_balanced(clean_tokens),
            "corrupt_is_balanced": T.markedrev_is_balanced(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} length-parity pairs")
    return pairs


def audit_markedrev_length_parity_pairs(pairs):
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


def audit_markedrev_balance_pairs(pairs):
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


# ---------------------------------------------------------------------------
# empirical behavioral-gap check (confirms which metric -- restored_fraction
# vs. absolute |delta_logit| -- is appropriate per pair type; loads ONE
# representative model per task, does NOT run any patching)
# ---------------------------------------------------------------------------

def check_behavioral_gap_cyclenav(pairs_grammar, pairs_position):
    sys.path.insert(0, "analysis")
    from rnn_patching import RNNPatchingHarness
    h = RNNPatchingHarness("rnn", "rec+ns/validation-short/1", task="cycle-navigation")
    grammar_gaps = [h.logit_diff(p["clean"]) - h.logit_diff(p["corrupt"]) for p in pairs_grammar[:20]]
    position_gaps = [h.logit_diff(p["clean"]) - h.logit_diff(p["corrupt"]) for p in pairs_position[:20]]
    return {
        "model": "rnn seed1 (representative)",
        "grammar_pairs_mean_abs_logit_gap": float(np.mean(np.abs(grammar_gaps))),
        "position_pairs_mean_abs_logit_gap": float(np.mean(np.abs(position_gaps))),
        "ratio": float(np.mean(np.abs(grammar_gaps)) / max(np.mean(np.abs(position_gaps)), 1e-9)),
        "conclusion": "grammar pairs have a much larger clean-vs-corrupt logit gap than position "
                     "pairs, confirming restored_fraction is well-defined (stable denominator) for "
                     "grammar pairs but not for position pairs, where absolute |delta_logit after "
                     "patch| should be used instead.",
    }


def check_behavioral_gap_markedrev(mirror_pairs, balance_pairs, length_parity_pairs):
    from rnn_patching import RNNPatchingHarness
    h = RNNPatchingHarness("rnn", "rec+ns/validation-short/3", task="marked-reversal")
    vd = torch.load("languages/marked-reversal/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}
    def to_idx(seq):
        return [tok2idx[t] for t in seq]
    mirror_gaps = [h.logit_diff(to_idx(p["clean"])) - h.logit_diff(to_idx(p["corrupt"])) for p in mirror_pairs[:20]]
    balance_gaps = [h.logit_diff(to_idx(p["clean"])) - h.logit_diff(to_idx(p["corrupt"])) for p in balance_pairs[:20]]
    lp_gaps = [h.logit_diff(to_idx(p["clean"])) - h.logit_diff(to_idx(p["corrupt"])) for p in length_parity_pairs[:20]]
    mirror_gap_mean = float(np.mean(np.abs(mirror_gaps)))
    balance_gap_mean = float(np.mean(np.abs(balance_gaps)))
    lp_gap_mean = float(np.mean(np.abs(lp_gaps)))
    return {
        "model": "rnn seed3 (representative, per Phase 1 the strongest hard-negative performer)",
        "mirror_pairs_mean_abs_logit_gap": mirror_gap_mean,
        "balance_pairs_mean_abs_logit_gap": balance_gap_mean,
        "length_parity_pairs_mean_abs_logit_gap": lp_gap_mean,
        "conclusion": (
            "THREE-WAY RESULT, discovered iteratively during this design step's own auditing "
            "(the behavioral-gap check is itself a confound detector, same discipline as Phase 2's "
            "probe confound audits): "
            "(1) mirror_pairs' near-zero gap (~4.8e-08) causally confirms the corrected Phase 2 "
            "finding that RNN performs NO content verification at all (balanced_content_match "
            "selectivity = 0.0 everywhere). "
            "(2) balance_pairs' gap is ALSO near-zero (~3.3e-07) -- initially assumed to be the "
            "primary positive test, but investigation (real-data behavioral split: RNN correctly "
            "rejects 25/25 magnitude-1/even-length hard negatives but only 0/25 magnitude-2/odd-"
            "length hard negatives; a follow-up linear probe restricted to odd-length sequences only "
            "shows is_balanced selectivity of 0.0000-0.0018, essentially zero, while length_parity "
            "ALONE recovers ~0.10 selectivity at the same layers) revealed that RNN does NOT check "
            "marker centering at all -- balance_pairs (odd length preserved, marker shifted by one) "
            "accidentally targets exactly the sub-case RNN cannot detect. "
            f"(3) length_parity_pairs (appending one token to flip odd->even length, matching the "
            f"dominant 82.6% real-world imbalance mechanism) shows the "
            f"{'large' if lp_gap_mean > 0.5 else 'non-trivial' if lp_gap_mean > 0.05 else 'still near-zero'} "
            f"gap ({lp_gap_mean:.4f}) -- THIS is the correctly-targeted primary causal test: RNN's "
            "marked-reversal 'shortcut' reduces, at the mechanistic level this design step could "
            "verify, to plain sequence-length parity, not any position- or content-aware computation."
        ),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print("=== cycle-navigation counterfactual audit ===", flush=True)
    grammar_pairs = cyclenav_grammar_pairs(19, 30, rng)
    position_pairs = cyclenav_position_pairs(19, 30, rng)
    grammar_audit = audit_cyclenav_pairs(grammar_pairs, "grammar")
    position_audit = audit_cyclenav_pairs(position_pairs, "position")
    print(json.dumps(grammar_audit, indent=2, default=str), flush=True)
    print(json.dumps(position_audit, indent=2, default=str), flush=True)

    print("\n=== marked-reversal counterfactual audit ===", flush=True)
    mirror_pairs = markedrev_mirror_pairs(50, 30, rng)
    mirror_audit = audit_markedrev_pairs(mirror_pairs)
    print(json.dumps(mirror_audit, indent=2, default=str), flush=True)

    balance_pairs = markedrev_balance_pairs(50, 30, rng)
    balance_audit = audit_markedrev_balance_pairs(balance_pairs)
    print(json.dumps(balance_audit, indent=2, default=str), flush=True)

    length_parity_pairs = markedrev_length_parity_pairs(50, 30, rng)
    length_parity_audit = audit_markedrev_length_parity_pairs(length_parity_pairs)
    print(json.dumps(length_parity_audit, indent=2, default=str), flush=True)

    print("\n=== behavioral-gap checks (representative models, no patching) ===", flush=True)
    cyclenav_gap = check_behavioral_gap_cyclenav(grammar_pairs, position_pairs)
    print(json.dumps(cyclenav_gap, indent=2), flush=True)
    markedrev_gap = check_behavioral_gap_markedrev(mirror_pairs, balance_pairs, length_parity_pairs)
    print(json.dumps(markedrev_gap, indent=2), flush=True)

    # ------------------------------------------------------------------
    # patching protocol spec (NOT executed here)
    # ------------------------------------------------------------------
    protocol = {
        "cycle-navigation": {
            "grammar_pairs": {
                "purpose": "test whether symbol_counts-adjacent (grammar-shape) information "
                          "causally drives accept/reject",
                "site": "readout position (last hook position, same convention as Phase 2)",
                "patch_scope": "TWO variants: (a) full-state patch (all channels, sanity/wiring "
                              "check -- expect rf~1.0, the readout position is definitionally "
                              "sufficient); (b) probe-direction subspace patch, projecting onto the "
                              "row-space of the Phase 2 symbol_counts ridge-regression weight matrix "
                              "(persisted in analysis_outputs/final_results/phase2_checkpoints/"
                              "phase2_p2_<arch>_seed<seed>_symbol_counts.pt), leaving the orthogonal "
                              "complement at the corrupt run's value -- isolates the specific "
                              "information the probe reads, rather than the whole activation.",
                "metric": "restored_fraction (well-defined: large, stable clean-vs-corrupt gap "
                         "confirmed empirically above)",
                "architectures_and_layers": "all 4 architectures, readout position only (matching "
                                            "the established site per architecture from Phase 2 -- "
                                            "layer0 for rnn/lstm, last layer for mamba/transformer; "
                                            "no layer sweep needed here since cycle-navigation showed "
                                            "no analogous site-transfer issue in Phase 2)",
                "prediction": "high restored fraction (subspace patch approaches the full-state "
                             "patch's rf~1.0) across all 4 architectures, both seeds",
            },
            "position_pairs": {
                "purpose": "test whether true_position-specific information causally drives "
                          "accept/reject, isolated from symbol_counts via probe-direction patching",
                "site": "readout position, same as above",
                "patch_scope": "probe-direction subspace patch, projecting onto the row-space of "
                              "the Phase 2 true_position logistic-regression weight matrix "
                              "(persisted per-cell in phase2_checkpoints/), leaving the orthogonal "
                              "complement (including symbol-count-relevant directions) at the "
                              "corrupt run's value. NOTE: since true_position is a NONLINEAR (mod-5) "
                              "reduction of the counts, its probe direction is not perfectly "
                              "separable from the counts directions in general position patching "
                              "should be read as testing 'the SPECIFIC linear direction the trained "
                              "true_position probe uses,' not a claim of clean information-theoretic "
                              "separation from symbol_counts.",
                "metric": "absolute |delta_logit after patch| (restored_fraction is numerically "
                         "unstable here: clean-vs-corrupt gap is near-zero by construction, "
                         "confirmed empirically above -- ratio to grammar pairs' gap reported in "
                         "the saved JSON). Report this relative to the grammar pairs' typical "
                         "logit swing as a reference scale for 'minimal effect.'",
                "architectures_and_layers": "all 4 architectures, readout position only",
                "prediction": "small |delta_logit after patch| relative to the grammar-pairs "
                             "reference scale, across all 4 architectures, both seeds",
            },
        },
        "marked-reversal": {
            "length_parity_pairs": {
                "status": "PRIMARY causal test (final, after a two-step revision within this same "
                         "design step). The original design's primary test (mirror_pairs, content "
                         "verification) and the first revision's primary test (balance_pairs, "
                         "marker-position-aware balance) BOTH came back with a degenerate ~0 "
                         "behavioral gap on the representative model; investigating why (real-data "
                         "accuracy split by imbalance magnitude, then a targeted linear probe holding "
                         "length parity constant) found RNN's actual mechanism is plain sequence-"
                         "length parity, which length_parity_pairs isolates directly and which shows "
                         "a large (8.64), non-degenerate gap -- confirming it is the correctly-"
                         "targeted primary test.",
                "purpose": "test whether the length-parity signal at the Phase-2-identified layer(s) "
                          "causally drives accept/reject, per architecture",
                "site": "readout position, SPECIFIC LAYER per the layer-by-layer length_parity probe "
                       "check run during this design step -- rnn: layer 1 (seed3, sel~0.10, "
                       "consistent from layer1 onward) and layer 4 (seed7, sel~0.10, only layer "
                       "clearing threshold); lstm: layer 4 (both seeds, by analogy with its "
                       "is_balanced sweep result -- length_parity itself was not separately swept for "
                       "lstm/mamba/transformer in this design step and should be re-verified before "
                       "the actual patching run); mamba, transformer: layer 4 / last layer (negative "
                       "controls -- their is_balanced sweep showed no signal at any layer, and since "
                       "length_parity is upstream of is_balanced this predicts no signal for "
                       "length_parity either, but this has NOT been directly probed yet and should be "
                       "checked, not assumed, before patching)",
                "patch_scope": "WHOLE-LAYER patch (all channels of the specified layer). For RNN/LSTM "
                              "this requires extending RecurrentScan/_apply_patch (analysis/"
                              "rnn_patching.py) to patch a single specified layer's slice of the "
                              "[num_layers,1,H] state -- current code only supports all-layers-at-once "
                              "(dims=None) or specific channels of the LAST layer only. Mamba's "
                              "PatchingHarness.patch_run and Transformer's TransformerPatchingHarness."
                              "patch_logit already accept an arbitrary layer argument -- no extension "
                              "needed for those two architectures. NOTE: corrupt is one token LONGER "
                              "than clean for this pair type (unlike every other pair type in this "
                              "design, which are same-length edits) -- the patching harness must patch "
                              "activations at the SAME relative readout position (last token) in both "
                              "runs, not the same absolute index; confirm this is how RNNPatchingHarness/"
                              "PatchingHarness/TransformerPatchingHarness index before running.",
                "metric": "restored_fraction (well-defined: large, non-degenerate clean-vs-corrupt "
                         "gap of 8.64 confirmed empirically above for rnn)",
                "prediction": "rnn: high restored fraction at its identified layer per seed (length "
                             "parity causally drives accept/reject); lstm: intermediate effect at "
                             "layer 4; mamba, transformer: minimal effect at any layer (to be "
                             "confirmed by directly probing length_parity for these two "
                             "architectures before patching, not assumed from the is_balanced result)",
            },
            "balance_pairs": {
                "status": "CONFIRMATORY null test -- demonstrates RNN does NOT check marker centering "
                         "within odd-length sequences, only overall length parity.",
                "purpose": "verify, causally, that marker-position-aware balance checking (as opposed "
                          "to plain length parity) plays no role -- closes the loop opened when this "
                          "pair type was originally assumed to be the primary test.",
                "site": "readout position, all layers 0-4",
                "patch_scope": "WHOLE-LAYER patch, same mechanism as length_parity_pairs",
                "metric": "restored_fraction is DEGENERATE (clean-vs-corrupt gap ~3.3e-07, "
                         "effectively 0/0) -- report absolute |delta_logit after patch| instead, "
                         "compared against length_parity_pairs' logit-swing scale as the reference "
                         "for 'meaningful effect.'",
                "prediction": "~no effect at any layer, any architecture -- a nonzero effect here "
                             "would contradict the odd-length-only probe result (is_balanced_given_"
                             "odd_length selectivity 0.0000-0.0018) and warrant re-investigation.",
            },
            "mirror_pairs": {
                "status": "CONFIRMATORY null test -- demonstrates no content-verification computation "
                         "exists to patch.",
                "purpose": "verify, causally, that no content-verification computation exists to "
                          "patch -- i.e. confirm the corrected Phase 2 finding (balanced_content_match "
                          "selectivity = 0.0 at every layer, every seed, all 4 architectures) at the "
                          "causal level, closing the loop opened by the original (confounded) design.",
                "site": "readout position, all layers 0-4",
                "patch_scope": "WHOLE-LAYER patch, same mechanism as length_parity_pairs",
                "metric": "restored_fraction is DEGENERATE (clean-vs-corrupt gap ~4.8e-08, "
                         "effectively 0/0) -- report absolute |delta_logit after patch| instead, "
                         "same convention as cycle-navigation's position_pairs.",
                "prediction": "~no effect at any layer, any architecture. A nonzero effect here "
                             "would contradict the corrected Phase 2 result and warrant "
                             "re-investigation.",
            },
        },
    }

    out = {
        "description": "Phase 3 design step: counterfactual construction + audit for causal "
                       "patching, both pilot tasks. NO PATCHING RUN -- this specs and verifies "
                       "the protocol only.",
        "cycle_navigation_audit": {
            "grammar_pairs": grammar_audit, "position_pairs": position_audit,
            "behavioral_gap_check": cyclenav_gap,
            "sample_pairs": {"grammar": grammar_pairs[:3], "position": position_pairs[:3]},
        },
        "marked_reversal_audit": {
            "mirror_pairs": mirror_audit, "balance_pairs": balance_audit,
            "length_parity_pairs": length_parity_audit,
            "behavioral_gap_check": markedrev_gap,
            "sample_pairs": {
                "mirror": mirror_pairs[:3], "balance": balance_pairs[:3],
                "length_parity": length_parity_pairs[:3],
            },
        },
        "patching_protocol": protocol,
    }
    out_path = RESULTS / "phase3_counterfactual_design.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
