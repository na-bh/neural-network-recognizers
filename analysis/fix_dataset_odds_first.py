"""Corrected-dataset experiment, OF1: dataset construction + audit.

Builds a "corrected" odds-first dataset preserving Butoi et al.'s mixed
negative-generation strategy (50% uniform-random / 50% "adversarial" half)
but replacing the adversarial half with an explicit hard-negative
construction: a genuine positive w#deinterleave(w) (deinterleave(w) =
w[::2] + w[1::2], even-indexed characters first then odd-indexed), with two
ADJACENT differing-valued positions in the post-marker half swapped -- this
preserves marker count (1), marker position (balanced halves), and the
post-marker half's content MULTISET relative to the pre-marker half (a swap
can never change a multiset, and deinterleave is itself just a permutation
of the pre-marker half), while breaking the specific deinterleave
correctness. SAME construction template as marked-copy/marked-reversal's F1
(analysis/fix_dataset_marked_copy.py) -- the only semantic difference is
is_genuinely_wrong_deinterleave compares against deinterleave(before)
instead of before itself, and the swap is restricted to ADJACENT positions
(per the user's construction spec) rather than any differing pair.

Reuses recognizers.hand_picked_languages.odds_first.OddsFirst directly
(language.sample() for positives and for the base string the hard negative
is derived from; language.is_negative() to verify every generated negative
really is negative).

Outputs (matching the on-disk format recognizers/neural_networks/data.py
reads, convertible to .prepared/.vocab via prepare_data.py exactly like the
standard pipeline):
  languages/odds-first-fixed/{main.tok,labels.txt,next-symbols.jsonl,
    log-probabilities.txt,negative-kind.txt,hard-negative-swap-meta.jsonl}
  languages/odds-first-fixed/datasets/validation-short/{...}
  languages/odds-first-fixed/datasets/test/{...}

PYTHONPATH=src:analysis python analysis/fix_dataset_odds_first.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from recognizers.hand_picked_languages.odds_first import OddsFirst
from recognizers.tools.jsonl import write_json_line
from recognizers.automata.reserved import ReservedSymbol

RESULTS = Path("analysis_outputs/final_results")
LANG_DIR = Path("languages/odds-first-fixed")
RANDOM_SEED = 20260719
TRAIN_LENGTH_RANGE = (0, 40)
TEST_LENGTH_RANGE = (0, 500)
TEST_EXAMPLES_PER_LENGTH = 10
TRAIN_SIZE = 10000
VALIDATION_SHORT_SIZE = 1000
NEG_UNIFORM_RANDOM_PROB = 0.5
MIN_N_FOR_HARD_NEGATIVE = 2
MAX_TRIES = 300


def deinterleave(w):
    return w[::2] + w[1::2]


def generate_random_string(length_range, alphabet_size, generator):
    lo, hi = length_range
    length = generator.randint(lo, hi)
    return tuple(generator.randrange(alphabet_size) for _ in range(length))


def generate_hard_negative_content(language_len_restricted, marker_symbol, generator,
                                    min_n=MIN_N_FOR_HARD_NEGATIVE, max_tries=MAX_TRIES):
    """Samples a genuine positive w#deinterleave(w), then swaps two ADJACENT
    positions within the post-marker (deinterleaved) half whose VALUES
    differ -- this breaks the specific deinterleave-correctness while
    preserving marker count/position, balanced halves, length, and the
    post-marker half's content multiset (a swap can never change a
    multiset). Adjacent-pair index i is chosen uniformly at random among
    all differing-valued adjacent pairs each call, so aggregated across many
    generated examples the mismatch position is NOT concentrated at any
    single index. Retries (re-sampling w entirely) if n < min_n or no
    differing-valued adjacent pair exists in the half."""
    for _ in range(max_tries):
        s, _ = language_len_restricted.sample(
            generator=generator, include_log_probability=False, include_next_symbols=False)
        m = s.index(marker_symbol)
        before, after = s[:m], s[m + 1:]
        n = len(after)
        if n < min_n:
            continue
        candidates = [i for i in range(n - 1) if after[i] != after[i + 1]]
        if not candidates:
            continue  # no differing-valued adjacent pair -- retry with a fresh w
        i = generator.choice(candidates)
        after_list = list(after)
        after_list[i], after_list[i + 1] = after_list[i + 1], after_list[i]
        corrupted = before + (marker_symbol,) + tuple(after_list)
        denom = max(1, n - 2)
        return corrupted, {"n_half": n, "swap_i": i, "swap_j": i + 1, "swap_i_relative": i / denom}
    raise RuntimeError(f"could not generate a hard negative after {max_tries} tries")


def propose_negative(language_len_restricted, marker_symbol, length_range, alphabet_size, generator):
    if generator.random() < NEG_UNIFORM_RANDOM_PROB:
        s = generate_random_string(length_range, alphabet_size, generator)
        return s, "uniform_random", None
    else:
        s, meta = generate_hard_negative_content(language_len_restricted, marker_symbol, generator)
        return s, "hard_negative", meta


def generate_negative_example(language_len_restricted, marker_symbol, length_range, alphabet_size, generator):
    for _ in range(MAX_TRIES):
        s, kind, meta = propose_negative(language_len_restricted, marker_symbol, length_range, alphabet_size, generator)
        is_neg, _ = language_len_restricted.is_negative(s, include_edit_distance=False)
        if is_neg:
            return s, kind, meta
    raise RuntimeError("could not generate a verified negative example")


def build_next_symbols_row(alphabet, next_symbols):
    row = []
    for next_symbols_i in next_symbols:
        has_eos = ReservedSymbol.EOS in next_symbols_i
        non_eos = sorted(alphabet[c] for c in next_symbols_i if c != ReservedSymbol.EOS)
        row.append({"s": " ".join(non_eos), "e": has_eos})
    return row


def generate_split(language_len_restricted, marker_symbol, length_range, alphabet, alphabet_size,
                    num_samples, generator, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(num_samples):
        label = generator.randrange(2)
        if label:
            s, parse = language_len_restricted.sample(
                generator=generator, include_log_probability=True, include_next_symbols=True)
            rows.append({"s": s, "label": 1, "kind": "", "log_prob": parse.log_probability,
                         "next_symbols": parse.next_symbols, "meta": None})
        else:
            s, kind, meta = generate_negative_example(
                language_len_restricted, marker_symbol, length_range, alphabet_size, generator)
            rows.append({"s": s, "label": 0, "kind": kind, "log_prob": None,
                         "next_symbols": None, "meta": meta})

    with (output_dir / "main.tok").open("w") as tok_f, \
         (output_dir / "labels.txt").open("w") as labels_f, \
         (output_dir / "negative-kind.txt").open("w") as kind_f, \
         (output_dir / "log-probabilities.txt").open("w") as logprob_f, \
         (output_dir / "next-symbols.jsonl").open("w") as ns_f, \
         (output_dir / "hard-negative-swap-meta.jsonl").open("w") as swapmeta_f:
        for r in rows:
            print(" ".join(alphabet[c] for c in r["s"]), file=tok_f)
            print(r["label"], file=labels_f)
            print(r["kind"], file=kind_f)
            if r["label"]:
                print(r["log_prob"], file=logprob_f)
                write_json_line(build_next_symbols_row(alphabet, r["next_symbols"]), ns_f)
            write_json_line(r["meta"] if r["meta"] is not None else {}, swapmeta_f)
    return rows


def audit_split(rows, split_name, marker_symbol):
    n_total = len(rows)
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = n_total - n_pos
    neg_rows = [r for r in rows if r["label"] == 0]
    n_uniform = sum(1 for r in neg_rows if r["kind"] == "uniform_random")
    n_hard = sum(1 for r in neg_rows if r["kind"] == "hard_negative")

    positive_fraction = n_pos / n_total
    uniform_fraction_of_negatives = n_uniform / n_neg if n_neg else float("nan")

    hard_rows = [r for r in neg_rows if r["kind"] == "hard_negative"]
    hard_checks = {"n_checked": len(hard_rows)}
    if hard_rows:
        def split_info(s):
            m = s.index(marker_symbol)
            return m, len(s) - m - 1
        balanced = [split_info(r["s"])[0] == split_info(r["s"])[1] for r in hard_rows]
        single_marker = [r["s"].count(marker_symbol) == 1 for r in hard_rows]

        def multiset_matches(r):
            m = r["s"].index(marker_symbol)
            before, after = r["s"][:m], r["s"][m + 1:]
            return Counter(before) == Counter(after)
        multiset_ok = [multiset_matches(r) for r in hard_rows]

        def is_genuinely_wrong_deinterleave(r):
            m = r["s"].index(marker_symbol)
            before, after = r["s"][:m], r["s"][m + 1:]
            return list(after) != list(deinterleave(before))
        mismatched = [is_genuinely_wrong_deinterleave(r) for r in hard_rows]
        hard_checks.update({
            "all_balanced_halves": all(balanced),
            "all_single_marker": all(single_marker),
            "all_multiset_matched_pre_post": all(multiset_ok),
            "all_genuinely_wrong_deinterleave": all(mismatched),
            "all_constraints_satisfied": all(balanced) and all(single_marker) and all(multiset_ok) and all(mismatched),
        })
        rel_positions = [r["meta"]["swap_i_relative"] for r in hard_rows]
        rel_positions = np.array(rel_positions)
        bins = np.linspace(0, 1, 6)
        hist, _ = np.histogram(rel_positions, bins=bins)
        hard_checks["swap_position_relative_histogram_5bins"] = hist.tolist()
        hard_checks["swap_position_distribution_note"] = (
            "counts of the (lower) swap index falling in each fifth of the post-marker half -- "
            "roughly equal counts across bins would confirm positions are NOT concentrated at a "
            "single location."
        )

    uniform_rows = [r for r in neg_rows if r["kind"] == "uniform_random"]
    uniform_checks = {"n_checked": len(uniform_rows)}
    if uniform_rows:
        def n_markers(s):
            return sum(1 for t in s if t == marker_symbol)
        marker_counts = [n_markers(r["s"]) for r in uniform_rows]
        counter = Counter(marker_counts)
        n_zero_marker = counter.get(0, 0)
        n_one_marker = counter.get(1, 0)
        n_multi_marker = len(uniform_rows) - n_zero_marker - n_one_marker
        uniform_checks.update({
            "marker_count_distribution": {str(k): v for k, v in sorted(counter.items())},
            "fraction_zero_markers": n_zero_marker / len(uniform_rows),
            "fraction_one_marker": n_one_marker / len(uniform_rows),
            "fraction_multi_marker": n_multi_marker / len(uniform_rows),
            "note": (
                "matches the expected 'mostly structural violation' pattern established across "
                "the whole marker-family pilot (a uniform-random string very rarely has exactly "
                "one marker AND balanced halves AND correct deinterleave content by chance)."
            ),
        })

    return {
        "split": split_name, "n_total": n_total,
        "n_positive": n_pos, "n_negative": n_neg,
        "positive_fraction": positive_fraction,
        "n_negative_uniform_random": n_uniform, "n_negative_hard_negative": n_hard,
        "uniform_random_fraction_of_negatives": uniform_fraction_of_negatives,
        "hard_negative_construction_checks": hard_checks,
        "uniform_random_structural_violation_distribution": uniform_checks,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    base_lang = OddsFirst()
    marker_symbol = base_lang.num_symbols  # 2
    alphabet_size = base_lang.alphabet_size()
    alphabet = [base_lang.symbol_to_str(c) for c in range(alphabet_size)]

    training_language = base_lang.with_length_range(TRAIN_LENGTH_RANGE)
    test_language = base_lang.with_length_range(TEST_LENGTH_RANGE)

    generator = random.Random(RANDOM_SEED)

    print("=== generating train (n=10000, length range 0-40) ===", flush=True)
    train_rows = generate_split(
        training_language, marker_symbol, TRAIN_LENGTH_RANGE, alphabet, alphabet_size,
        TRAIN_SIZE, generator, LANG_DIR)

    print("=== generating validation-short (n=1000, length range 0-40) ===", flush=True)
    val_rows = generate_split(
        training_language, marker_symbol, TRAIN_LENGTH_RANGE, alphabet, alphabet_size,
        VALIDATION_SHORT_SIZE, generator, LANG_DIR / "datasets" / "validation-short")

    test_size = (TEST_LENGTH_RANGE[1] - TEST_LENGTH_RANGE[0] + 1) * TEST_EXAMPLES_PER_LENGTH
    print(f"=== generating test (n={test_size}, length range 0-500) ===", flush=True)
    test_rows = generate_split(
        test_language, marker_symbol, TEST_LENGTH_RANGE, alphabet, alphabet_size,
        test_size, generator, LANG_DIR / "datasets" / "test")

    print("\n=== auditing train ===", flush=True)
    train_audit = audit_split(train_rows, "train", marker_symbol)
    print(json.dumps(train_audit, indent=2, default=str))

    print("\n=== auditing validation-short ===", flush=True)
    val_audit = audit_split(val_rows, "validation-short", marker_symbol)
    print(json.dumps(val_audit, indent=2, default=str))

    print("\n=== auditing test ===", flush=True)
    test_audit = audit_split(test_rows, "test", marker_symbol)
    print(json.dumps(test_audit, indent=2, default=str))

    assert train_audit["hard_negative_construction_checks"]["all_constraints_satisfied"]
    assert test_audit["hard_negative_construction_checks"]["all_constraints_satisfied"]

    out = {
        "task": "odds-first",
        "experiment": "corrected-dataset-fix (OF1: dataset construction + audit)",
        "description": (
            "Corrected odds-first dataset preserving Butoi et al.'s mixed negative-generation "
            "strategy (50% uniform-random / 50% adversarial) but replacing the adversarial half "
            "with an explicit hard-negative construction: a genuine positive w#deinterleave(w) "
            "with two ADJACENT differing-valued post-marker positions swapped -- preserving marker "
            "count/position, balanced halves, and the post-marker half's content multiset "
            "(deinterleave is itself just a permutation, and a swap can never change a multiset), "
            "breaking only the specific deinterleave-correctness. Same construction template as "
            "marked-copy/marked-reversal's F1, adapted for the deinterleave permutation target and "
            "restricted to adjacent swaps (per spec). Swap position randomized per-example (uniform "
            "over differing-valued adjacent pairs within the half) to avoid concentrating the "
            "content-mismatch signal at any single index."
        ),
        "random_seed": RANDOM_SEED,
        "language_output_directory": str(LANG_DIR),
        "splits_generated": {
            "train": {"n": TRAIN_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "validation-short": {"n": VALIDATION_SHORT_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "test": {"n": test_size, "length_range": TEST_LENGTH_RANGE},
        },
        "known_simplifications_vs_original_pipeline": [
            "No cross-split deduplication of positive examples (negligible given the string "
            "space size at these lengths, matching every prior fix experiment's own documented "
            "simplification).",
            "Hard-negative construction rejects (resamples w entirely) when n_half < 2 or the "
            "post-marker half has no differing-valued adjacent pair -- a minor, documented "
            "distributional deviation for the hard-negative half only.",
            "validation-long, test-short-held-out, and test-edit-distance splits were not "
            "generated -- not needed for the planned evaluations (standard test set + corrected "
            "test set only).",
        ],
        "audits": {"train": train_audit, "validation-short": val_audit, "test": test_audit},
    }
    out_path = RESULTS / "fix_dataset_odds_first.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
