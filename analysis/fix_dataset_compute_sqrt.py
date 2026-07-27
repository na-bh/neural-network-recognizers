"""Corrected-dataset experiment, CSF1: dataset construction + audit.

Builds a "corrected" compute-sqrt dataset preserving Butoi et al.'s mixed
negative-generation strategy (50% uniform-random / 50% "adversarial" half)
but replacing the adversarial half with an explicit hard-negative
construction: a genuine positive u_x=u_z (LSB-first, u_z=floor(sqrt(x))),
with a SINGLE bit of u_z flipped -- this preserves marker count (1),
well-formedness, and the answer-length-sufficiency shortcut (n_z >=
ceil(bitlength(x)/2), which depends only on field WIDTHS, untouched by a
content-only bit flip), while making the claimed answer genuinely wrong
(math.isqrt(x) != z_corrupt, guaranteed since flipping any single bit of a
fixed-width binary encoding always changes its decoded integer value).
Uses the same single-bit-flip construction validated in the Phase 3
causal-patching pilot (analysis/phase3_computesqrt_counterfactual_design.py
target_computation_pairs), extended to full-dataset scale with the
modular-arithmetic-simple fix's Option A tertile-sweep discipline (target
tertile drawn once and held fixed across retries) and an explicit z==0
exclusion (per the binary-multiplication fix's z=0-edge-case lesson: z=0's
minimal-width encoding is a single '0' bit, not empty -- excluded here from
serving as a hard-negative BASE, even though the bit-flip construction
itself handles it correctly, to keep the base population unambiguous).

Reuses recognizers.hand_picked_languages.compute_sqrt.ComputeSqrt directly
(language.sample() for positives and for the base string the hard negative
is derived from; language.is_negative() to verify every generated negative
really is negative). LSB-first encoding confirmed directly from
binary_util.decode_binary/binary_encoding (see Phase 3 task audit).

Outputs (matching the on-disk format recognizers/neural_networks/data.py
reads, convertible to .prepared/.vocab via prepare_data.py exactly like the
standard pipeline):
  languages/compute-sqrt-fixed/{main.tok,labels.txt,next-symbols.jsonl,
    negative-kind.txt,hard-negative-bitflip-meta.jsonl}
  languages/compute-sqrt-fixed/datasets/validation-short/{...}
  languages/compute-sqrt-fixed/datasets/test/{...}

(No log-probabilities.txt: ComputeSqrt.supports_log_probability() is False
-- the language itself never produces one.)

PYTHONPATH=src:analysis python analysis/fix_dataset_compute_sqrt.py
"""

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from recognizers.hand_picked_languages.compute_sqrt import ComputeSqrt, EQUALS
from recognizers.hand_picked_languages.binary_util import decode_binary
from recognizers.tools.jsonl import write_json_line
from recognizers.automata.reserved import ReservedSymbol

RESULTS = Path("analysis_outputs/final_results")
LANG_DIR = Path("languages/compute-sqrt-fixed")
RANDOM_SEED = 20260721
TRAIN_LENGTH_RANGE = (0, 40)
TEST_LENGTH_RANGE = (0, 500)
TEST_EXAMPLES_PER_LENGTH = 10
TRAIN_SIZE = 10000
VALIDATION_SHORT_SIZE = 1000
NEG_UNIFORM_RANDOM_PROB = 0.5
MIN_N_Z_FOR_HARD_NEGATIVE = 2
MAX_TRIES = 400
TERTILES = ["low", "mid", "high"]


def categorize_tertile(frac):
    if frac < 1 / 3:
        return "low"
    elif frac < 2 / 3:
        return "mid"
    else:
        return "high"


def bitlength(v):
    return v.bit_length() if v > 0 else 1


def compute_z(x):
    """Matches ComputeSqrt.compute_z EXACTLY (math.floor(math.sqrt(x)), not
    math.isqrt) -- this is the actual ground-truth target definition, and
    float64's math.sqrt LOSES PRECISION for large x (confirmed empirically:
    diverges from the exact integer isqrt for ~36% of real positives at
    test-set lengths, where x can have 200-400+ bits). Every correctness
    check here MUST use this same float-based definition -- using the
    'more correct' math.isqrt instead would silently disagree with the
    language's own is_negative() ground truth for long sequences."""
    return math.floor(math.sqrt(x))


def marker_count_is_one(s):
    return s.count(EQUALS) == 1


def fields_well_formed(s):
    if s.count(EQUALS) != 1:
        return False
    i = s.index(EQUALS)
    return bool(s[:i]) and bool(s[i + 1:])


def answer_length_sufficient(s):
    if not fields_well_formed(s):
        return False
    i = s.index(EQUALS)
    u_x, u_z = s[:i], s[i + 1:]
    x = decode_binary(u_x)
    need = bitlength(compute_z(x))
    return len(u_z) >= need


SHORTCUT_FEATURES = {
    "marker_count_is_one": marker_count_is_one,
    "fields_well_formed": fields_well_formed,
    "answer_length_sufficient": answer_length_sufficient,
}


def is_genuinely_wrong_sqrt(s):
    i = s.index(EQUALS)
    u_x, u_z = s[:i], s[i + 1:]
    x = decode_binary(u_x)
    z = decode_binary(u_z)
    return compute_z(x) != z


def generate_random_string(length_range, alphabet_size, generator):
    lo, hi = length_range
    length = generator.randint(lo, hi)
    return tuple(generator.randrange(alphabet_size) for _ in range(length))


def generate_hard_negative_content(language_len_restricted, generator,
                                    min_n_z=MIN_N_Z_FOR_HARD_NEGATIVE, max_tries=MAX_TRIES):
    """Samples a genuine positive u_x=u_z, flips a single bit of u_z (LSB-
    first) at a position drawn from a target tertile held FIXED across
    retries (Option A discipline). Excludes z==0 bases (minimal-width
    encoding is a single '0' bit -- flipping it still works correctly, but
    is excluded here to keep the base population unambiguous, per the
    binary-multiplication fix's z=0 lesson). Field widths (n_x, n_z) are
    untouched -- only one bit's VALUE changes -- so marker_count_is_one,
    fields_well_formed, and answer_length_sufficient are structurally
    guaranteed to survive; z_corrupt != z_clean is guaranteed since
    flipping any single bit of a fixed-width binary encoding always changes
    the decoded integer value, and z_clean == true isqrt(x) (it came from a
    genuine positive), so z_corrupt is guaranteed genuinely wrong."""
    target_tertile = TERTILES[generator.randrange(3)]
    for _ in range(max_tries):
        s, _ = language_len_restricted.sample(
            generator=generator, include_log_probability=False, include_next_symbols=False)
        eq_idx = s.index(EQUALS)
        u_x, u_z = s[:eq_idx], s[eq_idx + 1:]
        n_z = len(u_z)
        if n_z < min_n_z:
            continue
        x = decode_binary(u_x)
        z = decode_binary(u_z)
        if z == 0:
            continue  # exclude the k=0 edge case from serving as a hard-negative base
        candidates = list(range(n_z))
        denom = max(1, n_z - 1)
        def frac_of(j):
            return j / denom
        bucket = [j for j in candidates if categorize_tertile(frac_of(j)) == target_tertile]
        if not bucket:
            continue
        bit_idx = generator.choice(bucket)
        u_z_list = list(u_z)
        u_z_list[bit_idx] = 1 - u_z_list[bit_idx]
        z_corrupt = decode_binary(u_z_list)
        corrupted = u_x + (EQUALS,) + tuple(u_z_list)
        meta = {
            "n_z": n_z, "bit_idx": bit_idx, "bit_position_relative": frac_of(bit_idx),
            "bit_position_tertile": target_tertile, "x": x, "z_clean": z, "z_corrupt": z_corrupt,
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
         (output_dir / "hard-negative-bitflip-meta.jsonl").open("w") as metaf:
        for r in rows:
            print(" ".join(alphabet[c] for c in r["s"]), file=tok_f)
            print(r["label"], file=labels_f)
            print(r["kind"], file=kind_f)
            if r["label"]:
                write_json_line(build_next_symbols_row(alphabet, r["next_symbols"]), ns_f)
            write_json_line(r["meta"] if r["meta"] is not None else {}, metaf)
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
        all_wrong_sqrt = [is_genuinely_wrong_sqrt(r["s"]) for r in hard_rows]
        hard_checks.update({
            "all_pass_all_three_shortcut_features": all(all_pass_shortcuts),
            "n_detectable_by_any_shortcut_feature": sum(1 for x in all_pass_shortcuts if not x),
            "all_genuinely_fail_sqrt_check": all(all_wrong_sqrt),
            "all_constraints_satisfied": all(all_pass_shortcuts) and all(all_wrong_sqrt),
        })
        rel_positions = np.array([r["meta"]["bit_position_relative"] for r in hard_rows])
        bins = np.linspace(0, 1, 4)
        hist, _ = np.histogram(rel_positions, bins=bins)
        hard_checks["bit_position_tertile_histogram_low_mid_high"] = hist.tolist()
        hmax, hmin = int(hist.max()), int(hist.min())
        hard_checks["tertiles_within_20_percent_of_each_other"] = bool(hmin >= 0.8 * hmax)

        # magnitude-change distribution: |z_corrupt - z_clean| == 2**bit_idx exactly
        # (report via bit_idx, since 2**bit_idx can be astronomically large for the
        # long test-set sequences -- LSB-first means 'low' tertile = small-magnitude
        # bit flips, 'high' tertile = large-magnitude flips).
        bit_idx_by_tertile = {t: [] for t in TERTILES}
        for r in hard_rows:
            bit_idx_by_tertile[r["meta"]["bit_position_tertile"]].append(r["meta"]["bit_idx"])
        magnitude_summary = {}
        for t in TERTILES:
            idxs = bit_idx_by_tertile[t]
            if idxs:
                arr = np.array(idxs)
                magnitude_summary[t] = {
                    "n": len(idxs), "bit_idx_min": int(arr.min()), "bit_idx_max": int(arr.max()),
                    "bit_idx_mean": float(arr.mean()), "bit_idx_median": float(np.median(arr)),
                    "magnitude_at_min_bit_idx": 2 ** int(arr.min()), "magnitude_at_max_bit_idx": 2 ** int(arr.max()),
                }
        hard_checks["magnitude_change_distribution_by_tertile"] = magnitude_summary
        hard_checks["magnitude_change_note"] = (
            "magnitude of the value change = 2**bit_idx exactly (flipping bit b changes the decoded "
            "integer by +-2**b). Reported via bit_idx (the exponent) rather than raw magnitude, since "
            "2**bit_idx is astronomically large for the long test-set sequences (n_z can be ~100+ "
            "bits at length 500) and not meaningfully printable in full. LSB-first encoding confirms "
            "the audit's flagged concern: 'low' tertile flips change the value by a small amount, "
            "'high' tertile flips change it by an enormous amount -- this magnitude-scaling confound "
            "is inherent to the task's encoding, not a construction artifact, and mirrors the same "
            "concern flagged for binary-addition/binary-multiplication's LSB-first fields."
        )

    uniform_rows = [r for r in neg_rows if r["kind"] == "uniform_random"]
    uniform_checks = {"n_checked": len(uniform_rows)}
    if uniform_rows:
        def n_markers(s):
            return sum(1 for t in s if t == EQUALS)
        marker_counts = [n_markers(r["s"]) for r in uniform_rows]
        from collections import Counter
        counter = Counter(marker_counts)
        n_zero_marker = counter.get(0, 0)
        n_one_marker = counter.get(1, 0)
        n_multi_marker = len(uniform_rows) - n_zero_marker - n_one_marker
        n_accidentally_valid = sum(
            1 for r in uniform_rows
            if marker_count_is_one(r["s"]) and fields_well_formed(r["s"]) and not is_genuinely_wrong_sqrt(r["s"])
        )
        uniform_checks.update({
            "marker_count_distribution": {str(k): v for k, v in sorted(counter.items())},
            "fraction_zero_markers": n_zero_marker / len(uniform_rows),
            "fraction_one_marker": n_one_marker / len(uniform_rows),
            "fraction_multi_marker": n_multi_marker / len(uniform_rows),
            "n_accidentally_correct_sqrt_among_well_formed": n_accidentally_valid,
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
    base_lang = ComputeSqrt()
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

    print("\n=== bit-flip position-distribution uniformity check ===", flush=True)
    for split_name, audit in [("train", train_audit), ("validation-short", val_audit), ("test", test_audit)]:
        ok = audit["hard_negative_construction_checks"]["tertiles_within_20_percent_of_each_other"]
        print(f"  {split_name}: tertiles_within_20_percent={ok} "
              f"hist={audit['hard_negative_construction_checks']['bit_position_tertile_histogram_low_mid_high']}",
              flush=True)
    assert test_audit["hard_negative_construction_checks"]["tertiles_within_20_percent_of_each_other"]

    out = {
        "task": "compute-sqrt",
        "experiment": "corrected-dataset-fix (CSF1: dataset construction + audit)",
        "description": (
            "Corrected compute-sqrt dataset preserving Butoi et al.'s mixed negative-generation "
            "strategy (50% uniform-random / 50% hard-negative half). Hard negatives are constructed "
            "via a single-bit flip in u_z (LSB-first) of a genuine positive -- structurally guaranteed "
            "to preserve marker_count_is_one, fields_well_formed, and answer_length_sufficient (all "
            "depend only on field widths, untouched by a content-only flip), while making the claimed "
            "answer genuinely wrong (any single-bit flip changes the decoded integer). Excludes z==0 "
            "bases from hard-negative construction. Bit-flip position swept across low/mid/high "
            "tertiles of u_z with the target held fixed across retries (Option A discipline)."
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
            "Hard-negative construction rejects (resamples entirely) when n_z < 2, z == 0, or the "
            "target tertile has no candidate bit position for that sample -- minor documented "
            "distributional deviations for the hard-negative half only.",
            "No log-probabilities.txt: ComputeSqrt.supports_log_probability() is False.",
            "The magnitude-scaling confound flagged in the Phase 3 compute-sqrt audit (LSB-first "
            "encoding means 'high'-tertile bit flips change the claimed value by an astronomically "
            "larger amount than 'low'-tertile flips) is inherent to the task's own encoding and is "
            "NOT corrected by this construction -- it is reported (magnitude_change_distribution_by_"
            "tertile) so downstream training-time analysis can account for it, matching the same "
            "documented, uncorrected confound in binary-addition/binary-multiplication's fixes.",
        ],
        "audits": {"train": train_audit, "validation-short": val_audit, "test": test_audit},
    }
    out_path = RESULTS / "fix_dataset_compute_sqrt.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
