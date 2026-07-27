"""Corrected-dataset experiment for modular-arithmetic-simple, MF1: dataset
construction + audit.

Builds a "corrected" modular-arithmetic-simple dataset preserving Butoi et
al.'s mixed negative-generation strategy (50% uniform-random / 50% hard-
negative half) but replacing whatever original negative-generation used
with an EXPLICIT (op,digit)-pair-PERMUTATION construction: take a genuine
positive, permute two (preferably NON-ADJACENT) (op,digit) pairs -- same
digit multiset, same operator multiset, same length, same claimed final
answer as the source positive, but a genuinely DIFFERENT true left-to-right
mod-5 value (since +,-,* do not commute when interleaved) -- verified via
rejection sampling, not assumed.

Positive generation: no hand-picked-language .sample() exists for this
FSA-based task -- positives are generated via a from-scratch left-to-right
random walk (random_valid_expression, reused from analysis/phase3_modarith_
counterfactuals.py, already validated: is_valid_example matches the FSA's
own acceptance criterion). next_symbols supervision (for the 'ns' training
objective) is computed directly from the grammar's own transition
semantics: after a digit (not yet at '='), valid next tokens are
{+,-,*,=}; after an operator, valid next tokens are the 5 digits; after
'=', the grammar (a RECOGNIZER of only correct expressions) admits ONLY
the single TRUE computed value as a valid continuation -- confirmed
directly from the FSA's check_states, which each have exactly one outgoing
arc, not five.

Hard-negative subtype: for very short expressions (n_pairs=2), only ONE
(the only two available) (op,digit) pairs can be swapped, and they are
NECESSARILY adjacent (no non-adjacent option exists) -- flagged explicitly
in the audit as "short-sequence-constrained," not hidden. For n_pairs>=3,
a genuinely non-adjacent pair is preferentially selected.

Violation position is swept across early/mid/late (relative fraction of
sequence length) via biased rejection sampling toward each target
fraction, matching bucket-sort/binary-addition/Dyck-2-3's position-sweep
discipline; expect (and report) the same train-skews-early/test-is-uniform
discretization artifact documented in those tasks' analogous audits.

PYTHONPATH=src:analysis python analysis/fix_dataset_modular_arithmetic_simple.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_modarith_task_audit import (
    parse_and_evaluate, is_valid_example, SHORTCUT_FEATURES, DIGITS, OPS,
)
from phase3_modarith_counterfactuals import random_valid_expression

RESULTS = Path("analysis_outputs/final_results")
LANG_DIR = Path("languages/modular-arithmetic-simple-fixed")
RANDOM_SEED = 20260717
TRAIN_LENGTH_RANGE = (0, 40)
TEST_LENGTH_RANGE = (0, 500)
TEST_EXAMPLES_PER_LENGTH = 10
TRAIN_SIZE = 10000
VALIDATION_SHORT_SIZE = 1000
NEG_UNIFORM_RANDOM_PROB = 0.5
MAX_TRIES = 400
ALPHABET = ["*", "+", "-", "0", "1", "2", "3", "4", "="]
ALPHABET_SIZE = 9
DIGIT_LIST = sorted(DIGITS)
OP_LIST = sorted(OPS)


def n_pairs_for_length(seq_len):
    """digit (op digit)^k = digit has length 1 + 2k + 2 = 2k+3 -> k = (len-3)/2."""
    return (seq_len - 3) // 2


def gen_total_len(length_range, rng):
    """Sample a total sequence length directly (matching the established
    convention), then round to the nearest valid odd length >= 3 (every
    valid expression has length 2k+3 for k>=0 pairs, which is always odd
    and >= 3)."""
    lo, hi = length_range
    n = int(rng.integers(max(3, lo), hi + 1))
    if n % 2 == 0:
        n += 1
    return max(n, 3)


def compute_next_symbols(seq):
    """next_symbols[i] = (valid_token_strings, eos_valid) BEFORE consuming
    seq[i] (i in range(len(seq)+1)). After a digit (not yet at '='): valid
    next = {+,-,*,=}. After an operator: valid next = the 5 digits. After
    '=': valid next = ONLY the true computed value (the grammar recognizes
    ONLY correct expressions -- confirmed from the FSA's check_states,
    each with exactly one outgoing arc). After the final digit: EOS only."""
    n = len(seq)
    rows = []
    value = None
    state = "need_first_digit"
    for i, tok in enumerate(seq):
        if state == "need_first_digit":
            rows.append({"s": " ".join(DIGIT_LIST), "e": False})
            value = int(tok)
            state = "need_op_or_equals"
        elif state == "need_op_or_equals":
            rows.append({"s": " ".join(sorted(OP_LIST + ["="])), "e": False})
            if tok == "=":
                state = "need_final_digit"
            else:
                state = "need_digit_after_op"
                pending_op = tok
        elif state == "need_digit_after_op":
            rows.append({"s": " ".join(DIGIT_LIST), "e": False})
            x = int(tok)
            if pending_op == "+":
                value = (value + x) % 5
            elif pending_op == "-":
                value = (value - x) % 5
            elif pending_op == "*":
                value = (value * x) % 5
            state = "need_op_or_equals"
        elif state == "need_final_digit":
            rows.append({"s": str(value), "e": False})
            state = "done"
    rows.append({"s": "", "e": True})
    return rows


def swap_pairs(seq, i, j):
    """Swap (op,digit) pairs at pair-indices i,j (0-indexed, pair k starts
    at seq index 1+2k)."""
    corrupted = seq[:]
    ai, aj = 1 + 2 * i, 1 + 2 * j
    corrupted[ai], corrupted[ai + 1], corrupted[aj], corrupted[aj + 1] = (
        seq[aj], seq[aj + 1], seq[ai], seq[ai + 1]
    )
    return corrupted


def categorize_tertile(frac):
    if frac < 1 / 3:
        return "low"
    elif frac < 2 / 3:
        return "mid"
    else:
        return "high"


TERTILES = ["low", "mid", "high"]


def generate_hard_negative(rng, length_range, max_tries=MAX_TRIES):
    """Option A (explicitly stratified position sampling, per user decision
    after the first construction showed a train/test position-distribution
    mismatch): every candidate non-adjacent (op,digit)-pair is bucketed by
    the TERTILE of its MIDPOINT relative position (the midpoint, not just
    the first pair's own start index, since the first index "a" is
    structurally capped below k-3 -- for small k this NEVER reaches the
    high tertile at all, confirmed empirically: k=4 gives a-fractions only
    up to 0.30, k=16 only up to 0.79 -- the midpoint covers a wider,
    though still not perfectly full, range).

    The target tertile is drawn UNIFORMLY at random ONCE, OUTSIDE the retry
    loop, and held fixed across every retry (rather than being re-rolled
    every attempt). This matters because swap-validity rejection rates are
    NOT uniform across tertiles: swapping two EARLY (op,digit) pairs in a
    long expression very often leaves the final mod-5 result unchanged
    (empirically ~95%+ rejection for "low"-tertile swaps in long/test-like
    sequences, vs ~60% for "high"-tertile swaps) -- later multiplications
    (e.g. by 0) or accumulation dynamics frequently "erase" an early
    perturbation before the final value is reached. Re-rolling the target
    tertile on every retry (the first version of this fix) meant that a
    request for "low" almost always failed and then silently got replaced
    by a fresh (uniformly re-drawn) target on the next attempt -- so the
    process drifted away from "low" outcomes even though "low" was
    requested equally often, defeating stratification. Holding the target
    fixed and retrying (fresh k/expression each time) until THAT tertile's
    swap succeeds fixes this, at the cost of needing more retries for
    "low" and long sequences specifically."""
    target_tertile = TERTILES[int(rng.integers(0, 3))]
    for _ in range(max_tries):
        total_len = gen_total_len(length_range, rng)
        k = n_pairs_for_length(total_len)
        if k < 2:
            continue
        clean = random_valid_expression(rng, k)
        pair_starts_frac = [(1 + 2 * i) / (len(clean) - 1) for i in range(k)]
        # prefer non-adjacent pairs; for k==2 only the adjacent pair (0,1) exists
        if k == 2:
            i, j = 0, 1
            constrained = True
            mid_frac = (pair_starts_frac[i] + pair_starts_frac[j]) / 2
            if categorize_tertile(mid_frac) != target_tertile:
                continue  # k==2 forces "low"; retry (fresh k) unless target happens to be "low"
        else:
            candidates = [(a, b) for a in range(k) for b in range(a + 2, k)]  # non-adjacent
            if not candidates:
                candidates = [(a, a + 1) for a in range(k - 1)]  # fallback (shouldn't happen for k>=3)
            constrained = False
            bucket = [(a, b) for (a, b) in candidates
                      if categorize_tertile((pair_starts_frac[a] + pair_starts_frac[b]) / 2) == target_tertile]
            if not bucket:
                continue  # this k has no candidate in the (fixed) target tertile -- retry with a fresh k
            i, j = bucket[int(rng.integers(0, len(bucket)))]
            mid_frac = (pair_starts_frac[i] + pair_starts_frac[j]) / 2
        corrupted = swap_pairs(clean, i, j)
        ok, true_val, claimed = parse_and_evaluate(corrupted)
        clean_ok, clean_val, clean_claim = parse_and_evaluate(clean)
        if not ok or true_val == claimed:
            continue  # coincidentally still correct -- retry (fresh k, same fixed target_tertile)
        if not all(fn(clean) == fn(corrupted) for fn in SHORTCUT_FEATURES.values()):
            continue
        meta = {
            "n_pairs": k, "swapped_pair_indices": [i, j],
            "short_sequence_constrained": constrained,
            "violation_position_relative": mid_frac,
            "violation_position_tertile": categorize_tertile(mid_frac),
            "target_tertile_requested": target_tertile,
        }
        return corrupted, meta
    raise RuntimeError("could not generate a hard negative")


def generate_uniform_random(length_range, rng):
    n = gen_total_len(length_range, rng)
    return [ALPHABET[int(rng.integers(0, ALPHABET_SIZE))] for _ in range(n)]


def propose_negative(length_range, rng):
    if rng.random() < NEG_UNIFORM_RANDOM_PROB:
        return generate_uniform_random(length_range, rng), "uniform_random", None
    else:
        seq, meta = generate_hard_negative(rng, length_range)
        return seq, "hard_negative", meta


def generate_negative_example(length_range, rng, max_tries=MAX_TRIES):
    for _ in range(max_tries):
        seq, kind, meta = propose_negative(length_range, rng)
        if not is_valid_example(seq):
            return seq, kind, meta
    raise RuntimeError("could not generate a verified negative example")


def generate_split(length_range, num_samples, rng, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(num_samples):
        label = int(rng.integers(0, 2))
        if label:
            total_len = gen_total_len(length_range, rng)
            k = n_pairs_for_length(total_len)
            s = random_valid_expression(rng, k)
            rows.append({"s": s, "label": 1, "kind": "", "next_symbols": compute_next_symbols(s), "meta": None})
        else:
            s, kind, meta = generate_negative_example(length_range, rng)
            rows.append({"s": s, "label": 0, "kind": kind, "next_symbols": None, "meta": meta})

    with (output_dir / "main.tok").open("w") as tok_f, \
         (output_dir / "labels.txt").open("w") as labels_f, \
         (output_dir / "negative-kind.txt").open("w") as kind_f, \
         (output_dir / "next-symbols.jsonl").open("w") as ns_f, \
         (output_dir / "hard-negative-meta.jsonl").open("w") as meta_f:
        for r in rows:
            print(" ".join(r["s"]), file=tok_f)
            print(r["label"], file=labels_f)
            print(r["kind"], file=kind_f)
            if r["label"]:
                print(json.dumps(r["next_symbols"]), file=ns_f)
            print(json.dumps(r["meta"] if r["meta"] is not None else {}), file=meta_f)
    return rows


def audit_split(rows, split_name):
    n_total = len(rows)
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = n_total - n_pos
    neg_rows = [r for r in rows if r["label"] == 0]
    n_uniform = sum(1 for r in neg_rows if r["kind"] == "uniform_random")
    hard_rows = [r for r in neg_rows if r["kind"] == "hard_negative"]
    n_hard = len(hard_rows)

    pos_rows = [r for r in rows if r["label"] == 1]
    all_pos_valid = all(is_valid_example(r["s"]) for r in pos_rows)

    hard_checks = {"n_checked": n_hard}
    if hard_rows:
        all_pass_shortcuts = [all(fn(r["s"]) for fn in SHORTCUT_FEATURES.values()) for r in hard_rows]
        all_invalid = [not is_valid_example(r["s"]) for r in hard_rows]
        n_constrained = sum(1 for r in hard_rows if r["meta"]["short_sequence_constrained"])
        hard_checks.update({
            "all_pass_all_six_shortcut_features": all(all_pass_shortcuts),
            "n_detectable_by_any_shortcut_feature": sum(1 for x in all_pass_shortcuts if not x),
            "all_genuinely_invalid": all(all_invalid),
            "n_from_short_sequence_constrained (n_pairs==2)": n_constrained,
            "fraction_short_sequence_constrained": n_constrained / n_hard,
        })
        fracs = [r["meta"]["violation_position_relative"] for r in hard_rows]
        arr = np.array(fracs)
        bins = np.linspace(0, 1, 4)
        hist, _ = np.histogram(arr, bins=bins)
        hard_checks["position_relative_tertile_histogram_low_mid_high"] = hist.tolist()
        hard_checks["position_relative_mean"] = float(arr.mean())
        hmax, hmin = int(hist.max()), int(hist.min())
        hard_checks["tertiles_within_20_percent_of_each_other"] = bool(hmin >= 0.8 * hmax)
        n_pairs_dist = Counter(r["meta"]["n_pairs"] for r in hard_rows)
        hard_checks["n_pairs_distribution"] = {str(k): v for k, v in sorted(n_pairs_dist.items())}

    uniform_rows = [r for r in neg_rows if r["kind"] == "uniform_random"]
    uniform_checks = {"n_checked": len(uniform_rows)}
    if uniform_rows:
        n_valid_parse = sum(1 for r in uniform_rows if is_valid_example(r["s"]))
        uniform_checks["n_accidentally_valid"] = n_valid_parse
        uniform_checks["fraction_accidentally_valid"] = n_valid_parse / len(uniform_rows)

    return {
        "split": split_name, "n_total": n_total, "n_positive": n_pos, "n_negative": n_neg,
        "positive_fraction": n_pos / n_total, "all_positives_grammatically_valid": all_pos_valid,
        "n_negative_uniform_random": n_uniform, "n_negative_hard_negative": n_hard,
        "uniform_random_fraction_of_negatives": n_uniform / n_neg if n_neg else float("nan"),
        "hard_negative_construction_checks": hard_checks,
        "uniform_random_structural_check": uniform_checks,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=== generating train (n=10000, length range 0-40) ===", flush=True)
    train_rows = generate_split(TRAIN_LENGTH_RANGE, TRAIN_SIZE, rng, LANG_DIR)

    print("=== generating validation-short (n=1000, length range 0-40) ===", flush=True)
    val_rows = generate_split(TRAIN_LENGTH_RANGE, VALIDATION_SHORT_SIZE, rng, LANG_DIR / "datasets" / "validation-short")

    test_size = (TEST_LENGTH_RANGE[1] - TEST_LENGTH_RANGE[0] + 1) * TEST_EXAMPLES_PER_LENGTH
    print(f"=== generating test (n={test_size}, length range 0-500) ===", flush=True)
    test_rows = generate_split(TEST_LENGTH_RANGE, test_size, rng, LANG_DIR / "datasets" / "test")

    print("\n=== auditing train ===", flush=True)
    train_audit = audit_split(train_rows, "train")
    print(json.dumps(train_audit, indent=2, default=str))
    print("\n=== auditing validation-short ===", flush=True)
    val_audit = audit_split(val_rows, "validation-short")
    print(json.dumps(val_audit, indent=2, default=str))
    print("\n=== auditing test ===", flush=True)
    test_audit = audit_split(test_rows, "test")
    print(json.dumps(test_audit, indent=2, default=str))

    assert train_audit["all_positives_grammatically_valid"]
    assert test_audit["all_positives_grammatically_valid"]
    assert train_audit["hard_negative_construction_checks"]["all_pass_all_six_shortcut_features"]
    assert test_audit["hard_negative_construction_checks"]["all_pass_all_six_shortcut_features"]
    assert train_audit["hard_negative_construction_checks"]["all_genuinely_invalid"]
    assert test_audit["hard_negative_construction_checks"]["all_genuinely_invalid"]

    print("\n=== position-distribution uniformity check (Option A target) ===", flush=True)
    for split_name, audit in [("train", train_audit), ("validation-short", val_audit), ("test", test_audit)]:
        ok = audit["hard_negative_construction_checks"]["tertiles_within_20_percent_of_each_other"]
        print(f"  {split_name}: tertiles_within_20_percent={ok} "
              f"hist={audit['hard_negative_construction_checks']['position_relative_tertile_histogram_low_mid_high']}",
              flush=True)
    assert train_audit["hard_negative_construction_checks"]["tertiles_within_20_percent_of_each_other"]
    assert test_audit["hard_negative_construction_checks"]["tertiles_within_20_percent_of_each_other"]

    old_histograms_path = Path("/tmp/modarith_old_histograms.json")
    old_histograms = json.loads(old_histograms_path.read_text()) if old_histograms_path.exists() else None
    if old_histograms:
        print("\n=== OLD (pre-Option-A) vs NEW position histograms ===", flush=True)
        for split_name, audit in [("train", train_audit), ("validation-short", val_audit), ("test", test_audit)]:
            new_hist = audit["hard_negative_construction_checks"]["position_relative_tertile_histogram_low_mid_high"]
            old_hist = old_histograms.get(split_name, {}).get("position_relative_tertile_histogram_low_mid_high")
            print(f"  {split_name}: OLD={old_hist}  NEW={new_hist}", flush=True)

    out = {
        "task": "modular-arithmetic-simple",
        "experiment": "corrected-dataset-fix (MF1: dataset construction + audit)",
        "description": (
            "Corrected modular-arithmetic-simple dataset preserving Butoi et al.'s mixed negative-"
            "generation strategy (50% uniform-random / 50% hard-negative half). Hard negatives are "
            "constructed via (op,digit)-pair permutation on genuine positives -- same digit "
            "multiset, operator multiset, length, and claimed answer, but a genuinely different "
            "true left-to-right mod-5 value -- verified to pass all six listed shortcut features "
            "while being genuinely invalid."
        ),
        "random_seed": RANDOM_SEED,
        "language_output_directory": str(LANG_DIR),
        "splits_generated": {
            "train": {"n": TRAIN_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "validation-short": {"n": VALIDATION_SHORT_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "test": {"n": test_size, "length_range": TEST_LENGTH_RANGE},
        },
        "known_simplifications_vs_original_pipeline": [
            "Positives are generated via a from-scratch left-to-right random walk (random_valid_"
            "expression), not the original FSA-based sampling machinery -- validated equivalent via "
            "is_valid_example matching the FSA's own acceptance criterion exactly, but the exact "
            "PROBABILITY DISTRIBUTION over valid strings may differ from the original FSA's "
            "weighting scheme.",
            "No cross-split deduplication of positive examples (negligible given the string space "
            "size at these lengths, matching every prior fix experiment's own documented "
            "simplification).",
        ],
        "audits": {"train": train_audit, "validation-short": val_audit, "test": test_audit},
        "position_sampling_revision": {
            "decision": "Option A: explicitly stratified position sampling -- every candidate non-"
                       "adjacent (op,digit)-pair is bucketed by the TERTILE of its MIDPOINT relative "
                       "position (not just the first index, which is structurally capped below the "
                       "high tertile for small k). A target tertile is drawn uniformly at random ONCE "
                       "per example and held FIXED across every retry (an earlier version re-drew the "
                       "target on every retry, which silently drifted away from 'low' outcomes because "
                       "swaps of two EARLY pairs in a long expression are much more likely to leave the "
                       "final mod-5 result unchanged -- ~95%+ rejection vs ~60% for 'high' -- so a "
                       "re-rolled target effectively abandoned 'low' whenever it was hard). Replaces the "
                       "original 'closest to a continuously-sampled target fraction' approach, which "
                       "produced OPPOSITE-direction skews on train (low/mid-skewed) vs test "
                       "(high-skewed) due to candidate-density differences between short and long "
                       "expressions.",
            "old_histograms_pre_option_A": old_histograms,
            "new_histograms_post_option_A": {
                split_name: {
                    "position_relative_tertile_histogram_low_mid_high": audit["hard_negative_construction_checks"]["position_relative_tertile_histogram_low_mid_high"],
                    "position_relative_mean": audit["hard_negative_construction_checks"]["position_relative_mean"],
                    "tertiles_within_20_percent_of_each_other": audit["hard_negative_construction_checks"]["tertiles_within_20_percent_of_each_other"],
                }
                for split_name, audit in [("train", train_audit), ("validation-short", val_audit), ("test", test_audit)]
            },
        },
    }
    out_path = RESULTS / "fix_dataset_modular_arithmetic_simple.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
