"""Corrected-dataset experiment, F1: dataset construction + audit.

Builds a "corrected" marked-reversal dataset preserving Butoi et al.'s
mixed negative-generation strategy (50% uniform-random / 50% "adversarial"
half) but replacing the K-edit-perturbation adversarial half (which mostly
breaks marker STRUCTURE, not content, on marker-family tasks -- established
across the whole four-task pilot) with an explicit hard-negative
construction: a genuine positive w#reversed(w), with two positions in the
post-marker half swapped -- this preserves marker count (1), marker
position (centered), balanced halves, and the CONTENT MULTISET of the
post-marker half (a swap can never change a multiset), while breaking the
literal reversal-content match. This is the SAME swap-based construction
validated for bucket-sort's target-computation counterfactuals (Phase 3),
adapted here for wholesale DATASET generation (many examples, swap position
randomized per-example) rather than a small matched-pair test set.

Reuses recognizers.hand_picked_languages.marked_reversal.MarkedReversal
directly (language.sample() for positives and for the base string the hard
negative is derived from; language.is_negative() to verify every generated
negative really is negative, matching Butoi et al.'s own rejection-sampling
discipline).

Outputs (matching the on-disk format recognizers/neural_networks/data.py
reads, i.e. main.tok + labels.txt [+ next-symbols.jsonl for positives],
convertible to .prepared/.vocab via prepare_data.py exactly like the
standard pipeline):
  languages/marked-reversal-fixed/{main.tok,labels.txt,next-symbols.jsonl,
    log-probabilities.txt,negative-kind.txt}
  languages/marked-reversal-fixed/datasets/validation-short/{...}
  languages/marked-reversal-fixed/datasets/test/{...}

negative-kind.txt is a NEW side-channel file (analogous to the original
pipeline's num-edits.txt), one line per example: "" for positives,
"uniform_random" or "hard_negative" for negatives -- needed for F2-F4's
per-subpopulation analysis (the "hard negative" half is not detectable via
A1's provability-timing classifier the way K-edit perturbations are, since
it's constructed directly, not discovered post-hoc).

PYTHONPATH=src:analysis python analysis/fix_dataset_marked_reversal_f1.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from recognizers.hand_picked_languages.marked_reversal import MarkedReversal
from recognizers.tools.jsonl import write_json_line
from recognizers.automata.reserved import ReservedSymbol

RESULTS = Path("analysis_outputs/final_results")
LANG_DIR = Path("languages/marked-reversal-fixed")
RANDOM_SEED = 20260713
TRAIN_LENGTH_RANGE = (0, 40)
TEST_LENGTH_RANGE = (0, 500)
TEST_EXAMPLES_PER_LENGTH = 10
TRAIN_SIZE = 10000
VALIDATION_SHORT_SIZE = 1000
NEG_UNIFORM_RANDOM_PROB = 0.5
MIN_N_FOR_HARD_NEGATIVE = 2
MAX_TRIES = 300


def generate_random_string(length_range, alphabet_size, generator):
    lo, hi = length_range
    length = generator.randint(lo, hi)
    return tuple(generator.randrange(alphabet_size) for _ in range(length))


def generate_hard_negative_content(language_len_restricted, marker_symbol, generator,
                                    min_n=MIN_N_FOR_HARD_NEGATIVE, max_tries=MAX_TRIES):
    """Samples a genuine positive w#reversed(w), then swaps two positions
    within the post-marker (reversed) half whose VALUES differ -- this
    breaks the literal reversal-content match while preserving marker
    count/position, balanced halves, length, and the post-marker half's
    content multiset (a swap can never change a multiset). Position i is
    chosen uniformly at random within the half each call, so aggregated
    across many generated examples the mismatch position is NOT
    concentrated at any single index (the Phase-3 position-sweep lesson).
    Retries (re-sampling w entirely) if n < min_n (no valid swap exists for
    n<2) or if the half happens to be monochromatic (no differing pair
    exists for a 2-symbol alphabet when all bits are equal)."""
    for _ in range(max_tries):
        s, _ = language_len_restricted.sample(
            generator=generator, include_log_probability=False, include_next_symbols=False)
        m = s.index(marker_symbol)
        before, after = s[:m], s[m + 1:]
        n = len(before)
        if n < min_n:
            continue
        if len(set(after)) < 2:
            continue  # monochromatic half -- no differing pair to swap
        i = generator.randrange(n)
        candidates = [j for j in range(n) if after[j] != after[i]]
        j = generator.choice(candidates)
        after_list = list(after)
        after_list[i], after_list[j] = after_list[j], after_list[i]
        corrupted = before + (marker_symbol,) + tuple(after_list)
        return corrupted, {"n_half": n, "swap_i": i, "swap_j": j, "swap_i_relative": i / n, "swap_j_relative": j / n}
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
                # matching the standard generate_files() convention exactly:
                # log-probabilities.txt and next-symbols.jsonl get ONE LINE
                # PER POSITIVE EXAMPLE ONLY -- no placeholder lines for
                # negatives (verified against languages/marked-reversal/
                # next-symbols.jsonl's line count, which equals n_positive,
                # not n_total). Writing a line per example regardless of
                # label (the original bug here) desyncs next-symbols.prepared
                # from labels.prepared and breaks load_prepared_next_symbols_
                # file's zip(..., strict=True) at train time.
                print(r["log_prob"], file=logprob_f)
                write_json_line(build_next_symbols_row(alphabet, r["next_symbols"]), ns_f)
            # hard-negative-swap-meta.jsonl is a CUSTOM analysis-only side file
            # (not part of the standard pipeline), kept ONE LINE PER EXAMPLE
            # (aligned with main.tok/negative-kind.txt) for simplicity -- empty
            # object for positives and uniform-random negatives, real swap
            # metadata for hard negatives.
            write_json_line(r["meta"] if r["meta"] is not None else {}, swapmeta_f)
    return rows


def audit_split(rows, split_name, marker_symbol):
    n_total = len(rows)
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = n_total - n_pos
    neg_rows = [r for r in rows if r["label"] == 0]
    n_uniform = sum(1 for r in neg_rows if r["kind"] == "uniform_random")
    n_hard = sum(1 for r in neg_rows if r["kind"] == "hard_negative")

    # (a) positive fraction
    positive_fraction = n_pos / n_total

    # (b) 50/50 split within negatives
    uniform_fraction_of_negatives = n_uniform / n_neg if n_neg else float("nan")

    # (c) hard-negative half constraint verification
    hard_rows = [r for r in neg_rows if r["kind"] == "hard_negative"]
    hard_checks = {"n_checked": len(hard_rows)}
    if hard_rows:
        def split_info(s):
            m = s.index(marker_symbol)
            return m, len(s) - m - 1
        balanced = [split_info(r["s"])[0] == split_info(r["s"])[1] for r in hard_rows]
        single_marker = [r["s"].count(marker_symbol) == 1 for r in hard_rows]
        def multiset_matches_reversed_reconstruction(r):
            # the pre-marker half's multiset must equal the post-marker half's
            # multiset (guaranteed by construction -- a swap never changes it)
            m = r["s"].index(marker_symbol)
            before, after = r["s"][:m], r["s"][m + 1:]
            return Counter(before) == Counter(after)
        multiset_ok = [multiset_matches_reversed_reconstruction(r) for r in hard_rows]
        def is_genuinely_mismatched(r):
            m = r["s"].index(marker_symbol)
            before, after = r["s"][:m], r["s"][m + 1:]
            return list(after) != list(reversed(before))
        mismatched = [is_genuinely_mismatched(r) for r in hard_rows]
        hard_checks.update({
            "all_balanced_halves": all(balanced),
            "all_single_marker": all(single_marker),
            "all_multiset_matched_pre_post": all(multiset_ok),
            "all_genuinely_mismatched_reversal": all(mismatched),
            "all_constraints_satisfied": all(balanced) and all(single_marker) and all(multiset_ok) and all(mismatched),
        })
        # swap-position distribution (relative position within the half)
        rel_positions = []
        for r in hard_rows:
            meta = r["meta"]
            rel_positions.append(meta["swap_i_relative"])
            rel_positions.append(meta["swap_j_relative"])
        rel_positions = np.array(rel_positions)
        bins = np.linspace(0, 1, 6)  # 5 bins: [0-.2, .2-.4, .4-.6, .6-.8, .8-1]
        hist, _ = np.histogram(rel_positions, bins=bins)
        hard_checks["swap_position_relative_histogram_5bins"] = hist.tolist()
        hard_checks["swap_position_distribution_note"] = (
            "counts of swap positions (both indices per swap, pooled) falling in each fifth "
            "of the post-marker half [0-.2),[.2-.4),[.4-.6),[.6-.8),[.8-1.0] -- roughly equal "
            "counts across bins would confirm positions are NOT concentrated at a single "
            "location (the Phase-3 position-sweep lesson)."
        )

    # (d) uniform-random half structural violation distribution
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
                "the four-task pilot (a uniform-random string very rarely has exactly one "
                "marker AND balanced halves AND correct reversal content by chance)."
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
    base_lang = MarkedReversal()
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

    out = {
        "task": "marked-reversal",
        "experiment": "corrected-dataset-fix (F1: dataset construction + audit)",
        "description": (
            "Corrected marked-reversal dataset preserving Butoi et al.'s mixed negative-"
            "generation strategy (50% uniform-random / 50% adversarial) but replacing the "
            "K-edit-perturbation adversarial half with an explicit hard-negative construction: "
            "a genuine positive w#reversed(w) with two post-marker positions swapped -- "
            "preserving marker count/position, balanced halves, and the post-marker half's "
            "content multiset, breaking only the literal reversal-content match. Swap position "
            "randomized per-example (uniform within the half) to avoid concentrating the "
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
            "No cross-split deduplication of positive examples (original sample_dataset.py "
            "tracks generated_strings_output/excluded_strings for the held-out test split; "
            "skipped here as negligible given the string space size at these lengths).",
            "Hard-negative construction rejects (resamples w entirely) when n_half < 2 or the "
            "post-marker half is monochromatic -- a minor, documented distributional deviation "
            "for the hard-negative half only (affects only the shortest ~2-5% of the length "
            "range where monochromatic halves are non-negligible under a 2-symbol alphabet).",
            "validation-long, test-short-held-out, and test-edit-distance splits (present in "
            "the standard FLaRe pipeline) were not generated -- not needed for F2-F4's planned "
            "evaluations (standard test set + corrected test set only).",
        ],
        "audits": {"train": train_audit, "validation-short": val_audit, "test": test_audit},
    }
    out_path = RESULTS / "fix_dataset_marked_reversal.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
