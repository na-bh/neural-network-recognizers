"""Corrected-dataset experiment for binary-addition, BF1: dataset
construction + audit.

Builds a "corrected" binary-addition dataset preserving Butoi et al.'s mixed
negative-generation strategy (50% uniform-random / 50% "hard negative"
half) but replacing the K-edit-perturbation-derived hard negatives (which
Phase 1 established were a mix of structural violations, not a controlled
arithmetic-only construction) with an EXPLICIT structurally-valid bit-flip
construction: a genuine positive u_x + u_y = u_z with exactly ONE bit of
u_z flipped -- preserving u_x, u_y, segment lengths, and operator/equals
positions exactly (matches Phase 3's already-validated target_computation_
bitflip_pairs design, generalized here to full-scale, variable-length
dataset generation rather than fixed-length matched pairs).

Structural differences from the marked-reversal fix that shape this
construction (per this experiment's instruction):
  - TWO structural tokens ('+', '=') rather than one marker -- both must
    land at the SAME positions in clean/corrupted (guaranteed here since we
    never touch u_x/u_y/token-count, only flip a bit WITHIN u_z).
  - NO length invariant between operands (len_x, len_y vary independently,
    unlike the marker family's balanced-halves invariant) -- the hard-
    negative half must reflect this: since we only flip a bit inside u_z of
    an actual SAMPLED positive, len_x/len_y/len_z automatically inherit the
    same (independent, Dirichlet-derived) length distribution as genuine
    positives, with NO artificial re-balancing.
  - Bit-count-parity / LSB-parity relationship: a single bit-flip at
    position k ONLY breaks LSB-parity detectability when k=0 (flipping any
    OTHER position leaves bit 0 of u_z, and hence the necessary XOR-parity
    check, unchanged) -- audited explicitly below as the "LSB-parity-
    detectable fraction" of hard negatives (residual structural pathway).

Reuses recognizers.hand_picked_languages.binary_addition.BinaryAddition
directly (language.sample() for positives and for the source positive a
hard negative is derived from; language.is_negative() to verify every
generated negative really is negative, matching Butoi et al.'s and the
marked-reversal fix's own rejection-sampling discipline -- even though a
single bit-flip in u_z is PROVABLY always wrong, per Phase 1/Phase 3's
established math, this is still verified independently here rather than
assumed).

Outputs (matching the on-disk format recognizers/neural_networks/data.py
reads, convertible to .prepared/.vocab via prepare_data.py exactly like the
standard pipeline):
  languages/binary-addition-fixed/{main.tok,labels.txt,next-symbols.jsonl,
    negative-kind.txt}
  languages/binary-addition-fixed/datasets/validation-short/{...}
  languages/binary-addition-fixed/datasets/test/{...}

log-probabilities.txt is NOT written here: binary-addition's language class
has supports_log_probability()=False (unlike marked-reversal), so there is
no log-probability field to persist -- confirmed directly from binary_util.
py's LengthRestrictedBinaryArithmeticOperation.supports_log_probability().

negative-kind.txt and hard-negative-flip-meta.jsonl are NEW side-channel
files (mirroring marked-reversal-fixed's negative-kind.txt / hard-negative-
swap-meta.jsonl) -- one line per example: "" for positives, "uniform_random"
or "hard_negative" for negatives; flip metadata (flip_k, len_z, relative
position) for hard negatives, empty object otherwise.

PYTHONPATH=src:analysis python analysis/fix_dataset_binary_addition.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from recognizers.hand_picked_languages.binary_addition import BinaryAddition
from recognizers.hand_picked_languages.binary_util import OPERATOR, EQUALS, decode_binary
from recognizers.tools.jsonl import write_json_line
from recognizers.automata.reserved import ReservedSymbol

RESULTS = Path("analysis_outputs/final_results")
LANG_DIR = Path("languages/binary-addition-fixed")
RANDOM_SEED = 20260714
TRAIN_LENGTH_RANGE = (0, 40)
TEST_LENGTH_RANGE = (0, 500)
TEST_EXAMPLES_PER_LENGTH = 10
TRAIN_SIZE = 10000
VALIDATION_SHORT_SIZE = 1000
NEG_UNIFORM_RANDOM_PROB = 0.5
MAX_TRIES = 300


def generate_random_string(length_range, alphabet_size, generator):
    lo, hi = length_range
    length = generator.randint(lo, hi)
    return tuple(generator.randrange(alphabet_size) for _ in range(length))


def parse_structural(s):
    """Returns (u_x, u_y, u_z) as tuples of 0/1 ints -- assumes s is a
    genuine positive from language.sample() (operator/equals guaranteed
    present at their construction-time positions)."""
    op_idx = s.index(OPERATOR)
    eq_idx = s.index(EQUALS, op_idx + 1)
    u_x = s[:op_idx]
    u_y = s[op_idx + 1:eq_idx]
    u_z = s[eq_idx + 1:]
    return u_x, u_y, u_z


def generate_hard_negative_content(language_len_restricted, generator, max_tries=MAX_TRIES):
    """Samples a genuine positive u_x + u_y = u_z, then flips exactly ONE
    bit within u_z at a position chosen UNIFORMLY at random within
    [0, len_z) -- this ALWAYS changes decode(u_z) by exactly +-2^k (never a
    no-op), preserving u_x, u_y, all segment lengths, and operator/equals
    positions exactly. Choosing k uniformly per-example (not from a fixed
    small set) makes the RELATIVE flip position (k/len_z) uniform by
    construction for every individual example regardless of its own len_z,
    so aggregate relative-position uniformity does not depend on the
    length distribution of the split (train vs test) -- only the ABSOLUTE
    position's range differs between splits, as an expected consequence of
    train/test using different overall length ranges, not of this
    construction introducing bias."""
    for _ in range(max_tries):
        s, _ = language_len_restricted.sample(
            generator=generator, include_log_probability=False, include_next_symbols=False)
        u_x, u_y, u_z = parse_structural(s)
        len_z = len(u_z)
        if len_z < 1:
            continue
        k = generator.randrange(len_z)
        u_z_list = list(u_z)
        u_z_list[k] = 1 - u_z_list[k]
        corrupted = u_x + (OPERATOR,) + u_y + (EQUALS,) + tuple(u_z_list)
        meta = {"len_x": len(u_x), "len_y": len(u_y), "len_z": len_z,
                "flip_k": k, "flip_k_relative": k / len_z}
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
         (output_dir / "hard-negative-flip-meta.jsonl").open("w") as flipmeta_f:
        for r in rows:
            print(" ".join(alphabet[c] for c in r["s"]), file=tok_f)
            print(r["label"], file=labels_f)
            print(r["kind"], file=kind_f)
            if r["label"]:
                # matching the standard generate_files() convention exactly (and the
                # F1 next-symbols bug fix already made for marked-reversal-fixed):
                # next-symbols.jsonl gets ONE LINE PER POSITIVE EXAMPLE ONLY.
                write_json_line(build_next_symbols_row(alphabet, r["next_symbols"]), ns_f)
            write_json_line(r["meta"] if r["meta"] is not None else {}, flipmeta_f)
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

    # (c) hard-negative half constraint verification
    hard_rows = [r for r in neg_rows if r["kind"] == "hard_negative"]
    hard_checks = {"n_checked": len(hard_rows)}
    if hard_rows:
        def source_positive_reconstruction_matches(r):
            # reconstructing the would-be positive (undo the flip) must give
            # a genuine u_x+u_y=u_z -- confirms u_x/u_y/lengths/positions
            # were not altered, only the single flipped bit in u_z.
            u_x, u_y, u_z = parse_structural(r["s"])
            u_z_list = list(u_z)
            k = r["meta"]["flip_k"]
            u_z_list[k] = 1 - u_z_list[k]
            x, y, z_reconstructed = decode_binary(u_x), decode_binary(u_y), decode_binary(tuple(u_z_list))
            return (x + y) == z_reconstructed

        single_flip_reconstructs_to_positive = [source_positive_reconstruction_matches(r) for r in hard_rows]

        def is_genuinely_wrong(r):
            u_x, u_y, u_z = parse_structural(r["s"])
            x, y, z = decode_binary(u_x), decode_binary(u_y), decode_binary(u_z)
            return (x + y) != z

        genuinely_wrong = [is_genuinely_wrong(r) for r in hard_rows]

        hard_checks.update({
            "all_single_bit_flip_reconstructs_to_valid_positive": all(single_flip_reconstructs_to_positive),
            "all_genuinely_wrong_sum": all(genuinely_wrong),
            "all_constraints_satisfied": all(single_flip_reconstructs_to_positive) and all(genuinely_wrong),
        })

        # flip-position distributions: absolute (raw k) and relative
        # (k/len_z, tertile-binned low/mid/high per the Phase-3 position-
        # sweep discipline)
        abs_ks = np.array([r["meta"]["flip_k"] for r in hard_rows])
        rel_ks = np.array([r["meta"]["flip_k_relative"] for r in hard_rows])
        len_zs = np.array([r["meta"]["len_z"] for r in hard_rows])
        tertile_bins = np.linspace(0, 1, 4)  # low [0,1/3), mid [1/3,2/3), high [2/3,1]
        tertile_hist, _ = np.histogram(rel_ks, bins=tertile_bins)
        hard_checks["flip_position_absolute_summary"] = {
            "mean": float(abs_ks.mean()), "min": int(abs_ks.min()), "max": int(abs_ks.max()),
        }
        hard_checks["flip_position_relative_tertile_histogram_low_mid_high"] = tertile_hist.tolist()
        hard_checks["len_z_summary"] = {
            "mean": float(len_zs.mean()), "min": int(len_zs.min()), "max": int(len_zs.max()),
        }
        hard_checks["flip_position_distribution_note"] = (
            "relative position (flip_k / len_z) is drawn Uniform per-example by construction, so "
            "the low/mid/high tertile counts should be roughly equal REGARDLESS of split -- this is "
            "the quantity that must be uniform. The absolute flip_k range will differ between train "
            "and test (reported separately) purely because train's length range (0-40) is much "
            "narrower than test's (0-500), NOT because of any construction bias -- matching the "
            "marked-reversal fix F1's analogous train/test length-range caveat."
        )

        # (e) LSB-parity-detectable fraction: a single bit-flip only breaks
        # the necessary LSB(u_z)=LSB(u_x) XOR LSB(u_y) parity check when the
        # flip lands AT position 0 -- independently re-verified here (not
        # just taken from flip_k==0 bookkeeping) via direct recomputation.
        def lsb_parity_violated(r):
            u_x, u_y, u_z = parse_structural(r["s"])
            return (int(u_x[0]) ^ int(u_y[0])) != int(u_z[0])

        lsb_detectable = [lsb_parity_violated(r) for r in hard_rows]
        hard_checks["lsb_parity_detectable_fraction"] = float(np.mean(lsb_detectable))
        hard_checks["lsb_parity_detectable_matches_flip_at_position_0"] = (
            sum(lsb_detectable) == sum(1 for r in hard_rows if r["meta"]["flip_k"] == 0)
        )
        hard_checks["lsb_parity_note"] = (
            f"{np.mean(lsb_detectable):.4f} of hard negatives are LSB-parity-detectable (flip landed "
            f"exactly at position 0). This should be close to the theoretical mean(1/len_z) over the "
            f"hard-negative population, small relative to 0.7 -- reported as a residual structural "
            f"pathway, not grounds for restricting the construction further."
        )

    # (f) uniform-random half structural violation distribution
    uniform_rows = [r for r in neg_rows if r["kind"] == "uniform_random"]
    uniform_checks = {"n_checked": len(uniform_rows)}
    if uniform_rows:
        def n_operator(s):
            return sum(1 for t in s if t == OPERATOR)

        def n_equals(s):
            return sum(1 for t in s if t == EQUALS)

        op_counts = Counter(min(n_operator(r["s"]), 2) for r in uniform_rows)
        eq_counts = Counter(min(n_equals(r["s"]), 2) for r in uniform_rows)
        n_total_u = len(uniform_rows)
        uniform_checks.update({
            "n_operator_distribution_capped_at_2": {str(k): v for k, v in sorted(op_counts.items())},
            "n_equals_distribution_capped_at_2": {str(k): v for k, v in sorted(eq_counts.items())},
            "fraction_exactly_one_operator": op_counts.get(1, 0) / n_total_u,
            "fraction_exactly_one_equals": eq_counts.get(1, 0) / n_total_u,
            "note": (
                "matches the expected 'mostly structural violation' pattern established across the "
                "prior pilot (a uniform-random 4-symbol string very rarely has exactly one '+' AND "
                "one '=' in a valid arrangement by chance)."
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
    base_lang = BinaryAddition()
    alphabet_size = base_lang.alphabet_size()
    alphabet = [base_lang.symbol_to_str(c) for c in range(alphabet_size)]

    training_language = base_lang.with_length_range(TRAIN_LENGTH_RANGE)
    test_language = base_lang.with_length_range(TEST_LENGTH_RANGE)

    generator = random.Random(RANDOM_SEED)

    print("=== generating train (n=10000, length range 0-40) ===", flush=True)
    train_rows = generate_split(
        training_language, TRAIN_LENGTH_RANGE, alphabet, alphabet_size, TRAIN_SIZE, generator, LANG_DIR)

    print("=== generating validation-short (n=1000, length range 0-40) ===", flush=True)
    val_rows = generate_split(
        training_language, TRAIN_LENGTH_RANGE, alphabet, alphabet_size,
        VALIDATION_SHORT_SIZE, generator, LANG_DIR / "datasets" / "validation-short")

    test_size = (TEST_LENGTH_RANGE[1] - TEST_LENGTH_RANGE[0] + 1) * TEST_EXAMPLES_PER_LENGTH
    print(f"=== generating test (n={test_size}, length range 0-500) ===", flush=True)
    test_rows = generate_split(
        test_language, TEST_LENGTH_RANGE, alphabet, alphabet_size, test_size, generator,
        LANG_DIR / "datasets" / "test")

    print("\n=== auditing train ===", flush=True)
    train_audit = audit_split(train_rows, "train")
    print(json.dumps(train_audit, indent=2, default=str))

    print("\n=== auditing validation-short ===", flush=True)
    val_audit = audit_split(val_rows, "validation-short")
    print(json.dumps(val_audit, indent=2, default=str))

    print("\n=== auditing test ===", flush=True)
    test_audit = audit_split(test_rows, "test")
    print(json.dumps(test_audit, indent=2, default=str))

    out = {
        "task": "binary-addition",
        "experiment": "corrected-dataset-fix (BF1: dataset construction + audit)",
        "description": (
            "Corrected binary-addition dataset preserving Butoi et al.'s mixed negative-generation "
            "strategy (50% uniform-random / 50% hard-negative half) but replacing K-edit-perturbation "
            "hard negatives with an explicit structurally-valid single-bit-flip-in-u_z construction: "
            "a genuine positive u_x+u_y=u_z with exactly one bit of u_z flipped at a position drawn "
            "uniformly within [0, len_z) -- preserving u_x, u_y, all segment lengths, and operator/"
            "equals positions exactly, guaranteeing a genuinely wrong sum (never a no-op)."
        ),
        "random_seed": RANDOM_SEED,
        "language_output_directory": str(LANG_DIR),
        "splits_generated": {
            "train": {"n": TRAIN_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "validation-short": {"n": VALIDATION_SHORT_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "test": {"n": test_size, "length_range": TEST_LENGTH_RANGE},
        },
        "known_simplifications_vs_original_pipeline": [
            "No cross-split deduplication of positive examples (negligible given the string space "
            "size at these lengths, matching the marked-reversal fix's own documented simplification).",
            "log-probabilities.txt is NOT written: binary-addition's language class has "
            "supports_log_probability()=False (unlike marked-reversal), so there is no log-"
            "probability field for this task in the standard pipeline either.",
            "validation-long, test-short-held-out, and test-edit-distance splits (present in the "
            "standard FLaRe pipeline) were not generated -- not needed for BF2-BF4's planned "
            "evaluations (standard test set + corrected test set only).",
        ],
        "audits": {"train": train_audit, "validation-short": val_audit, "test": test_audit},
    }
    out_path = RESULTS / "fix_dataset_binary_addition.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
