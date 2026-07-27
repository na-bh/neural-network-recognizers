"""Corrected-dataset experiment, SMF1: dataset construction + audit.

Builds a "corrected" stack-manipulation dataset preserving Butoi et al.'s
mixed negative-generation strategy (50% uniform-random / 50% "adversarial"
half) but replacing the adversarial half with an explicit hard-negative
construction: a genuine positive program # correct_final_stack, with an
ADJACENT, DIFFERING-VALUE swap applied within the answer (post-marker)
portion -- preserves length, marker count/position, well-formedness (only
the answer is touched), and per-value bit counts (a swap can't change a
multiset), while breaking ONLY the true LIFO simulation result. Swap
position swept across low/mid/high tertiles with the target held fixed
across retries (Option A discipline). Reuses the exact target_computation_
pairs construction validated in the Phase 3 causal-patching pilot
(analysis/phase3_stackmanip_counterfactual_design.py), extended to
full-dataset scale.

Reuses recognizers.hand_picked_languages.stack_manipulation.
StackManipulation directly (language.sample() for positives and for the
base string the hard negative is derived from; language.is_negative() to
verify every generated negative really is negative) -- unlike modular-
arithmetic-simple/repeat-01, this task DOES have a real WeightedLanguage
sample() implementation matching the Phase 3 audit's is_positive_local
semantics exactly, so no from-scratch construction is needed.

Six shortcut features (from the Phase 3 audit, reused unchanged):
marker_count_is_one, operations_well_formed, stack_size_arithmetic_
consistent, first_token_never_pop, token_before_marker_never_push,
answer_bit_counts_subset_of_available.

Outputs (matching the on-disk format recognizers/neural_networks/data.py
reads, convertible to .prepared/.vocab via prepare_data.py exactly like the
standard pipeline):
  languages/stack-manipulation-fixed/{main.tok,labels.txt,
    next-symbols.jsonl,negative-kind.txt,hard-negative-swap-meta.jsonl}
  languages/stack-manipulation-fixed/datasets/validation-short/{...}
  languages/stack-manipulation-fixed/datasets/test/{...}

(No log-probabilities.txt: StackManipulation.supports_log_probability() is
False.)

PYTHONPATH=src:analysis python analysis/fix_dataset_stack_manipulation.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from recognizers.hand_picked_languages.stack_manipulation import StackManipulation, PUSH, POP, MARKER
from recognizers.tools.jsonl import write_json_line
from recognizers.automata.reserved import ReservedSymbol

RESULTS = Path("analysis_outputs/final_results")
LANG_DIR = Path("languages/stack-manipulation-fixed")
RANDOM_SEED = 20260723
TRAIN_LENGTH_RANGE = (0, 40)
TEST_LENGTH_RANGE = (0, 500)
TEST_EXAMPLES_PER_LENGTH = 10
TRAIN_SIZE = 10000
VALIDATION_SHORT_SIZE = 1000
NEG_UNIFORM_RANDOM_PROB = 0.5
MIN_ANSWER_LEN_FOR_HARD_NEGATIVE = 3
MAX_TRIES = 400
TERTILES = ["low", "mid", "high"]


def categorize_tertile(frac):
    if frac < 1 / 3:
        return "low"
    elif frac < 2 / 3:
        return "mid"
    else:
        return "high"


def marker_count_is_one(seq):
    return seq.count(MARKER) == 1


def operations_well_formed(seq):
    n = len(seq)
    i = 0
    stack = []
    while i < n and seq[i] in (0, 1):
        stack.append(seq[i])
        i += 1
    while i < n and seq[i] in (PUSH, POP):
        if seq[i] == PUSH:
            i += 1
            if not (i < n and seq[i] in (0, 1)):
                return False
            stack.append(seq[i])
            i += 1
        else:
            if not stack:
                return False
            stack.pop()
            i += 1
    return True


def stack_size_arithmetic_consistent(seq):
    if seq.count(MARKER) != 1 or not operations_well_formed(seq):
        return False
    i = seq.index(MARKER)
    n = len(seq)
    j = 0
    stack = []
    while j < n and seq[j] in (0, 1):
        stack.append(seq[j])
        j += 1
    while j < i:
        if seq[j] == PUSH:
            j += 2
            stack.append(None)
        else:
            if not stack:
                return False
            stack.pop()
            j += 1
    return len(seq) - (i + 1) == len(stack)


def first_token_never_pop(seq):
    return bool(seq) and seq[0] != POP


def token_before_marker_never_push(seq):
    if seq.count(MARKER) != 1:
        return False
    i = seq.index(MARKER)
    return i == 0 or seq[i - 1] != PUSH


def answer_bit_counts_subset_of_available(seq):
    if seq.count(MARKER) != 1:
        return False
    i = seq.index(MARKER)
    prefix, suffix = seq[:i], seq[i + 1:]
    avail = Counter(t for t in prefix if t in (0, 1))
    ans = Counter(t for t in suffix if t in (0, 1))
    if len(suffix) != sum(ans.values()):
        return False
    return ans[0] <= avail[0] and ans[1] <= avail[1]


SHORTCUT_FEATURES = {
    "marker_count_is_one": marker_count_is_one,
    "operations_well_formed": operations_well_formed,
    "stack_size_arithmetic_consistent": stack_size_arithmetic_consistent,
    "first_token_never_pop": first_token_never_pop,
    "token_before_marker_never_push": token_before_marker_never_push,
    "answer_bit_counts_subset_of_available": answer_bit_counts_subset_of_available,
}


def is_positive_local(seq):
    n = len(seq)
    i = 0
    stack = []
    while i < n and seq[i] in (0, 1):
        stack.append(seq[i])
        i += 1
    while i < n and seq[i] in (PUSH, POP):
        if seq[i] == PUSH:
            i += 1
            if i < n and seq[i] in (0, 1):
                stack.append(seq[i])
                i += 1
            else:
                return False
        else:
            if not stack:
                return False
            stack.pop()
            i += 1
    if i < n and seq[i] == MARKER:
        i += 1
    else:
        return False
    return seq[i:] == tuple(reversed(stack))


def generate_random_string(length_range, alphabet_size, generator):
    lo, hi = length_range
    length = generator.randint(lo, hi)
    return tuple(generator.randrange(alphabet_size) for _ in range(length))


def generate_hard_negative_content(language_len_restricted, generator,
                                    min_k=MIN_ANSWER_LEN_FOR_HARD_NEGATIVE, max_tries=MAX_TRIES):
    """Samples a genuine positive program # correct_final_stack, applies an
    ADJACENT, DIFFERING-VALUE swap within the answer at a position drawn
    from a target tertile held FIXED across retries. Preserves length,
    marker count/position, and per-value bit counts (a swap can't change a
    multiset) while breaking true LIFO order."""
    target_tertile = TERTILES[generator.randrange(3)]
    for _ in range(max_tries):
        s, _ = language_len_restricted.sample(
            generator=generator, include_log_probability=False, include_next_symbols=False)
        m = s.index(MARKER)
        answer = s[m + 1:]
        k = len(answer)
        if k < min_k:
            continue
        candidates = [j for j in range(k - 1) if answer[j] != answer[j + 1]]
        if not candidates:
            continue
        denom = max(1, k - 2)
        def frac_of(j):
            return j / denom
        bucket = [j for j in candidates if categorize_tertile(min(1.0, frac_of(j))) == target_tertile]
        if not bucket:
            continue
        j = generator.choice(bucket)
        answer_list = list(answer)
        answer_list[j], answer_list[j + 1] = answer_list[j + 1], answer_list[j]
        corrupted = s[:m + 1] + tuple(answer_list)
        meta = {
            "answer_length": k, "swap_indices": [j, j + 1], "swap_position_relative": frac_of(j),
            "swap_position_tertile": target_tertile,
        }
        return corrupted, meta
    raise RuntimeError(f"could not generate a hard negative after {max_tries} tries")


def propose_negative(language_len_restricted, length_range, alphabet_size, generator):
    if generator.random() < NEG_UNIFORM_RANDOM_PROB:
        s = generate_random_string(length_range, alphabet_size, generator)
        return s, "uniform_random", None
    else:
        s, meta = generate_hard_negative_content(language_len_restricted, generator)
        return s, "hard_negative", meta


def generate_negative_example(language_len_restricted, length_range, alphabet_size, generator):
    for _ in range(MAX_TRIES):
        s, kind, meta = propose_negative(language_len_restricted, length_range, alphabet_size, generator)
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


def generate_split(language_len_restricted, length_range, alphabet, alphabet_size,
                    num_samples, generator, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(num_samples):
        label = generator.randrange(2)
        if label:
            s, parse = language_len_restricted.sample(
                generator=generator, include_log_probability=False, include_next_symbols=True)
            rows.append({"s": s, "label": 1, "kind": "", "next_symbols": parse.next_symbols, "meta": None})
        else:
            s, kind, meta = generate_negative_example(language_len_restricted, length_range, alphabet_size, generator)
            rows.append({"s": s, "label": 0, "kind": kind, "next_symbols": None, "meta": meta})

    with (output_dir / "main.tok").open("w") as tok_f, \
         (output_dir / "labels.txt").open("w") as labels_f, \
         (output_dir / "negative-kind.txt").open("w") as kind_f, \
         (output_dir / "next-symbols.jsonl").open("w") as ns_f, \
         (output_dir / "hard-negative-swap-meta.jsonl").open("w") as swapmeta_f:
        for r in rows:
            print(" ".join(alphabet[c] for c in r["s"]), file=tok_f)
            print(r["label"], file=labels_f)
            print(r["kind"], file=kind_f)
            if r["label"]:
                write_json_line(build_next_symbols_row(alphabet, r["next_symbols"]), ns_f)
            write_json_line(r["meta"] if r["meta"] is not None else {}, swapmeta_f)
    return rows


def audit_split(rows, split_name):
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
        all_pass_shortcuts = [all(fn(r["s"]) for fn in SHORTCUT_FEATURES.values()) for r in hard_rows]
        all_fail_simulation = [not is_positive_local(r["s"]) for r in hard_rows]
        hard_checks.update({
            "all_pass_all_six_shortcut_features": all(all_pass_shortcuts),
            "n_detectable_by_any_shortcut_feature": sum(1 for x in all_pass_shortcuts if not x),
            "all_genuinely_fail_stack_simulation": all(all_fail_simulation),
            "all_constraints_satisfied": all(all_pass_shortcuts) and all(all_fail_simulation),
        })
        rel_positions = np.array([r["meta"]["swap_position_relative"] for r in hard_rows])
        bins = np.linspace(0, 1, 4)
        hist, _ = np.histogram(rel_positions, bins=bins)
        hard_checks["swap_position_tertile_histogram_low_mid_high"] = hist.tolist()
        hmax, hmin = int(hist.max()), int(hist.min())
        hard_checks["tertiles_within_20_percent_of_each_other"] = bool(hmin >= 0.8 * hmax)

    uniform_rows = [r for r in neg_rows if r["kind"] == "uniform_random"]
    uniform_checks = {"n_checked": len(uniform_rows)}
    if uniform_rows:
        marker_counts = Counter(r["s"].count(MARKER) for r in uniform_rows)
        n_accidentally_valid = sum(
            1 for r in uniform_rows
            if all(fn(r["s"]) for fn in SHORTCUT_FEATURES.values()) and is_positive_local(r["s"])
        )
        uniform_checks.update({
            "marker_count_distribution": {str(k): v for k, v in sorted(marker_counts.items())},
            "n_accidentally_positive": n_accidentally_valid,
        })

    return {
        "split": split_name, "n_total": n_total,
        "n_positive": n_pos, "n_negative": n_neg,
        "positive_fraction": positive_fraction,
        "n_negative_uniform_random": n_uniform, "n_negative_hard_negative": n_hard,
        "uniform_random_fraction_of_negatives": uniform_fraction_of_negatives,
        "hard_negative_construction_checks": hard_checks,
        "uniform_random_structural_check": uniform_checks,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    base_lang = StackManipulation()
    alphabet_size = base_lang.alphabet_size()
    alphabet = [base_lang.symbol_to_str(c) for c in range(alphabet_size)]

    training_language = base_lang.with_length_range(TRAIN_LENGTH_RANGE)
    test_language = base_lang.with_length_range(TEST_LENGTH_RANGE)

    generator = random.Random(RANDOM_SEED)

    print("=== generating train (n=10000, length range 0-40) ===", flush=True)
    train_rows = generate_split(
        training_language, TRAIN_LENGTH_RANGE, alphabet, alphabet_size,
        TRAIN_SIZE, generator, LANG_DIR)

    print("=== generating validation-short (n=1000, length range 0-40) ===", flush=True)
    val_rows = generate_split(
        training_language, TRAIN_LENGTH_RANGE, alphabet, alphabet_size,
        VALIDATION_SHORT_SIZE, generator, LANG_DIR / "datasets" / "validation-short")

    test_size = (TEST_LENGTH_RANGE[1] - TEST_LENGTH_RANGE[0] + 1) * TEST_EXAMPLES_PER_LENGTH
    print(f"=== generating test (n={test_size}, length range 0-500) ===", flush=True)
    test_rows = generate_split(
        test_language, TEST_LENGTH_RANGE, alphabet, alphabet_size,
        test_size, generator, LANG_DIR / "datasets" / "test")

    print("\n=== auditing train ===", flush=True)
    train_audit = audit_split(train_rows, "train")
    print(json.dumps(train_audit, indent=2, default=str))
    print("\n=== auditing validation-short ===", flush=True)
    val_audit = audit_split(val_rows, "validation-short")
    print(json.dumps(val_audit, indent=2, default=str))
    print("\n=== auditing test ===", flush=True)
    test_audit = audit_split(test_rows, "test")
    print(json.dumps(test_audit, indent=2, default=str))

    assert train_audit["hard_negative_construction_checks"]["all_constraints_satisfied"]
    assert test_audit["hard_negative_construction_checks"]["all_constraints_satisfied"]

    print("\n=== swap position-distribution uniformity check ===", flush=True)
    for split_name, audit in [("train", train_audit), ("validation-short", val_audit), ("test", test_audit)]:
        ok = audit["hard_negative_construction_checks"]["tertiles_within_20_percent_of_each_other"]
        print(f"  {split_name}: tertiles_within_20_percent={ok} "
              f"hist={audit['hard_negative_construction_checks']['swap_position_tertile_histogram_low_mid_high']}",
              flush=True)
    assert test_audit["hard_negative_construction_checks"]["tertiles_within_20_percent_of_each_other"]

    out = {
        "task": "stack-manipulation",
        "experiment": "corrected-dataset-fix (SMF1: dataset construction + audit)",
        "description": (
            "Corrected stack-manipulation dataset preserving Butoi et al.'s mixed negative-"
            "generation strategy (50% uniform-random / 50% hard-negative half). Hard negatives are "
            "constructed via an adjacent, differing-value swap within the answer (post-marker) "
            "portion of a genuine positive -- structurally guaranteed to preserve all six identified "
            "shortcut features (marker_count_is_one, operations_well_formed, stack_size_arithmetic_"
            "consistent, first_token_never_pop, token_before_marker_never_push, answer_bit_counts_"
            "subset_of_available), while genuinely breaking true LIFO simulation order. Swap position "
            "swept across low/mid/high tertiles with the target held fixed across retries (Option A "
            "discipline). Reuses the exact target_computation_pairs construction validated in the "
            "Phase 3 causal-patching pilot, extended to full-dataset scale. Note for downstream "
            "interpretation: Phase 3 found 6/8 causal-patching cells at floor-effect on the "
            "genuinely-hard 62-example subset (RNN/LSTM/Mamba defaulted to constant behavior; "
            "Transformer had a blanket-long-sequence-rejection confound). This corrected dataset has "
            "substantially more genuinely-hard examples (~2500+ vs 62), giving much more statistical "
            "power to detect whether corrected training can push any cell past floor."
        ),
        "random_seed": RANDOM_SEED,
        "language_output_directory": str(LANG_DIR),
        "splits_generated": {
            "train": {"n": TRAIN_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "validation-short": {"n": VALIDATION_SHORT_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "test": {"n": test_size, "length_range": TEST_LENGTH_RANGE},
        },
        "known_simplifications_vs_original_pipeline": [
            "No cross-split deduplication of positive examples.",
            "Hard-negative construction rejects (resamples entirely) when answer length < 3 or no "
            "differing-valued adjacent pair exists in the answer, or the target tertile has no "
            "candidate for that sample -- minor documented distributional deviations for the "
            "hard-negative half only.",
            "No log-probabilities.txt: StackManipulation.supports_log_probability() is False.",
        ],
        "audits": {"train": train_audit, "validation-short": val_audit, "test": test_audit},
    }
    out_path = RESULTS / "fix_dataset_stack_manipulation.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
