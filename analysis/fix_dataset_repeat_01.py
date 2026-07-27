"""Corrected-dataset experiment, RF1: dataset construction + audit.

Builds a "corrected" repeat-01 dataset preserving Butoi et al.'s mixed
negative-generation strategy (50% uniform-random / 50% "adversarial" half)
but replacing the adversarial half with an explicit hard-negative
construction: an adjacent-symbol swap somewhere in a valid (01)^k string
that preserves all four shortcut features identified in the Phase 3 audit
(length_parity_even, first_symbol_is_0, last_symbol_is_1, count_balance)
while genuinely breaking the alternation pattern. Uses the exact
target_computation_pairs construction from the Phase 3 causal-patching
pilot (analysis/phase3_repeat01_counterfactual_design.py) as the template:
candidates = range(1, n-2) (excludes the boundary pair (0,1) and (n-2,n-1)
so first_symbol_is_0/last_symbol_is_1 are structurally guaranteed to
survive), swap position drawn from a target tertile held FIXED across
retries (the "Option A" discipline established during modular-arithmetic-
simple's fix, to avoid tertile-drift from re-rolling the target on every
retry).

No hand-picked-language .sample() exists for this DFA-based task (unlike
odds-first/compute-sqrt) -- like modular-arithmetic-simple's fix, positives
are generated via a direct from-scratch construction ((01)^k, the ENTIRE
language), which is exact (not an approximation) since the grammar is a
single repeated 2-token block.

Outputs (matching the on-disk format recognizers/neural_networks/data.py
reads, convertible to .prepared/.vocab via prepare_data.py exactly like the
standard pipeline):
  languages/repeat-01-fixed/{main.tok,labels.txt,next-symbols.jsonl,
    negative-kind.txt,hard-negative-meta.jsonl}
  languages/repeat-01-fixed/datasets/validation-short/{...}
  languages/repeat-01-fixed/datasets/test/{...}

(No log-probabilities.txt: like modular-arithmetic-simple's fix, this is
not consumed by prepare_data.py and is not needed for any of the four
training objectives.)

PYTHONPATH=src:analysis python analysis/fix_dataset_repeat_01.py
"""

import json
from pathlib import Path

import numpy as np

RESULTS = Path("analysis_outputs/final_results")
LANG_DIR = Path("languages/repeat-01-fixed")
RANDOM_SEED = 20260720
TRAIN_LENGTH_RANGE = (0, 40)
TEST_LENGTH_RANGE = (0, 500)
TEST_EXAMPLES_PER_LENGTH = 10
TRAIN_SIZE = 10000
VALIDATION_SHORT_SIZE = 1000
NEG_UNIFORM_RANDOM_PROB = 0.5
MAX_TRIES = 400
TERTILES = ["low", "mid", "high"]
ALPHABET = ["0", "1"]
ALPHABET_SIZE = 2


def categorize_tertile(frac):
    if frac < 1 / 3:
        return "low"
    elif frac < 2 / 3:
        return "mid"
    else:
        return "high"


def build_valid_positive(k):
    """Direct construction: (01)^k, length 2k."""
    tokens = []
    for _ in range(k):
        tokens += [0, 1]
    return tokens


def expected(i):
    return 0 if i % 2 == 0 else 1


def is_positive_local(seq):
    if len(seq) % 2 != 0:
        return False
    return all(seq[i] == expected(i) for i in range(len(seq)))


def length_parity_even(seq):
    return len(seq) % 2 == 0


def first_symbol_is_0(seq):
    return bool(seq) and seq[0] == 0


def last_symbol_is_1(seq):
    return bool(seq) and seq[-1] == 1


def count_balance(seq):
    return seq.count(0) == seq.count(1)


SHORTCUT_FEATURES = {
    "length_parity_even": length_parity_even,
    "first_symbol_is_0": first_symbol_is_0,
    "last_symbol_is_1": last_symbol_is_1,
    "count_balance": count_balance,
}


def gen_total_len(length_range, rng):
    """Sample a total length directly, then round to the nearest valid even
    length >= 0 (every valid string has length 2k for k>=0)."""
    lo, hi = length_range
    n = int(rng.integers(lo, hi + 1))
    if n % 2 != 0:
        n = n + 1 if n + 1 <= hi else n - 1
    return max(n, 0)


def compute_next_symbols(seq):
    """next_symbols[i] = valid continuation BEFORE consuming seq[i] (i in
    range(len(seq)+1)). The DFA has exactly one outgoing arc per state (no
    branching in what the next SYMBOL must be), so at every position the
    only valid next token is the alternation-determined symbol; EOS is also
    legal whenever the current state is accepting (i.e. an even number of
    symbols consumed so far)."""
    n = len(seq)
    rows = []
    for i in range(n + 1):
        accepting = (i % 2 == 0)
        next_tok = "0" if i % 2 == 0 else "1"
        rows.append({"s": next_tok, "e": accepting})
    return rows


def generate_hard_negative(rng, length_range, max_tries=MAX_TRIES):
    """Adjacent-symbol swap at (i, i+1) with i drawn from candidates =
    range(1, n-2) -- excludes the boundary pair (0,1) and (n-2,n-1), so
    first_symbol_is_0 and last_symbol_is_1 are structurally guaranteed to
    survive (only interior positions are touched). Since the source is
    perfectly alternating, adjacent positions ALWAYS differ, so every swap
    is a genuine corruption -- no differing-value search needed (unlike
    odds-first/marked-copy, where the source content is arbitrary).
    length_parity_even and count_balance are also structurally guaranteed
    (a swap changes neither length nor the 0/1 multiset). Target tertile is
    drawn ONCE per call and held FIXED across retries (Option A discipline
    from modular-arithmetic-simple's fix, avoiding tertile-drift)."""
    target_tertile = TERTILES[int(rng.integers(0, 3))]
    for _ in range(max_tries):
        n = gen_total_len(length_range, rng)
        if n < 4:
            continue  # need at least k=2 pairs for a non-boundary candidate to exist
        clean = build_valid_positive(n // 2)
        candidates = list(range(1, n - 2))
        if not candidates:
            continue
        denom = max(1, n - 3)
        def frac_of(j):
            return (j - 1) / denom
        bucket = [j for j in candidates if categorize_tertile(min(1.0, max(0.0, frac_of(j)))) == target_tertile]
        if not bucket:
            continue
        i = int(rng.choice(bucket))
        corrupted = clean[:]
        corrupted[i], corrupted[i + 1] = corrupted[i + 1], corrupted[i]
        if not all(fn(corrupted) for fn in SHORTCUT_FEATURES.values()):
            continue  # verify, don't assume
        if is_positive_local(corrupted):
            continue  # shouldn't happen (adjacent alternating values always differ), but verify
        meta = {
            "length": n, "swap_indices": [i, i + 1],
            "swap_position_relative": frac_of(i), "swap_position_tertile": target_tertile,
        }
        return corrupted, meta
    raise RuntimeError("could not generate a hard negative")


def generate_uniform_random(length_range, rng):
    lo, hi = length_range
    n = int(rng.integers(lo, hi + 1))
    return [int(rng.integers(0, ALPHABET_SIZE)) for _ in range(n)]


def propose_negative(length_range, rng):
    if rng.random() < NEG_UNIFORM_RANDOM_PROB:
        return generate_uniform_random(length_range, rng), "uniform_random", None
    else:
        seq, meta = generate_hard_negative(rng, length_range)
        return seq, "hard_negative", meta


def generate_negative_example(length_range, rng, max_tries=MAX_TRIES):
    for _ in range(max_tries):
        seq, kind, meta = propose_negative(length_range, rng)
        if not is_positive_local(seq):
            return seq, kind, meta
    raise RuntimeError("could not generate a verified negative example")


def generate_split(length_range, num_samples, rng, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(num_samples):
        label = int(rng.integers(0, 2))
        if label:
            n = gen_total_len(length_range, rng)
            s = build_valid_positive(n // 2)
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
            print(" ".join(ALPHABET[c] for c in r["s"]), file=tok_f)
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
    all_pos_valid = all(is_positive_local(r["s"]) for r in pos_rows)

    hard_checks = {"n_checked": n_hard}
    if hard_rows:
        all_pass_shortcuts = [all(fn(r["s"]) for fn in SHORTCUT_FEATURES.values()) for r in hard_rows]
        all_invalid = [not is_positive_local(r["s"]) for r in hard_rows]
        hard_checks.update({
            "all_pass_all_four_shortcut_features": all(all_pass_shortcuts),
            "n_detectable_by_any_shortcut_feature": sum(1 for x in all_pass_shortcuts if not x),
            "all_genuinely_fail_alternation": all(all_invalid),
        })
        fracs = [r["meta"]["swap_position_relative"] for r in hard_rows]
        arr = np.array(fracs)
        bins = np.linspace(0, 1, 4)
        hist, _ = np.histogram(arr, bins=bins)
        hard_checks["position_relative_tertile_histogram_low_mid_high"] = hist.tolist()
        hard_checks["position_relative_mean"] = float(arr.mean())
        hmax, hmin = int(hist.max()), int(hist.min())
        hard_checks["tertiles_within_20_percent_of_each_other"] = bool(hmin >= 0.8 * hmax)

    uniform_rows = [r for r in neg_rows if r["kind"] == "uniform_random"]
    uniform_checks = {"n_checked": len(uniform_rows)}
    if uniform_rows:
        n_accidentally_valid = sum(1 for r in uniform_rows if is_positive_local(r["s"]))
        uniform_checks["n_accidentally_valid"] = n_accidentally_valid
        uniform_checks["fraction_accidentally_valid"] = n_accidentally_valid / len(uniform_rows)

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
    assert train_audit["hard_negative_construction_checks"]["all_pass_all_four_shortcut_features"]
    assert test_audit["hard_negative_construction_checks"]["all_pass_all_four_shortcut_features"]
    assert train_audit["hard_negative_construction_checks"]["all_genuinely_fail_alternation"]
    assert test_audit["hard_negative_construction_checks"]["all_genuinely_fail_alternation"]

    print("\n=== position-distribution uniformity check ===", flush=True)
    for split_name, audit in [("train", train_audit), ("validation-short", val_audit), ("test", test_audit)]:
        ok = audit["hard_negative_construction_checks"]["tertiles_within_20_percent_of_each_other"]
        print(f"  {split_name}: tertiles_within_20_percent={ok} "
              f"hist={audit['hard_negative_construction_checks']['position_relative_tertile_histogram_low_mid_high']}",
              flush=True)
    assert test_audit["hard_negative_construction_checks"]["tertiles_within_20_percent_of_each_other"]

    out = {
        "task": "repeat-01",
        "experiment": "corrected-dataset-fix (RF1: dataset construction + audit)",
        "description": (
            "Corrected repeat-01 dataset preserving Butoi et al.'s mixed negative-generation "
            "strategy (50% uniform-random / 50% hard-negative half). Hard negatives are constructed "
            "via an adjacent-symbol swap at interior positions (i, i+1), i in range(1, n-2) -- "
            "structurally guaranteed to preserve all four identified shortcut features "
            "(length_parity_even, first_symbol_is_0, last_symbol_is_1, count_balance) while "
            "genuinely breaking alternation at the swapped positions. Uses the exact "
            "target_computation_pairs construction validated in the Phase 3 causal-patching pilot "
            "(analysis/phase3_repeat01_counterfactual_design.py), extended to full-dataset scale "
            "with the modular-arithmetic-simple fix's Option A tertile-sweep discipline (target "
            "tertile drawn once and held fixed across retries)."
        ),
        "random_seed": RANDOM_SEED,
        "language_output_directory": str(LANG_DIR),
        "splits_generated": {
            "train": {"n": TRAIN_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "validation-short": {"n": VALIDATION_SHORT_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "test": {"n": test_size, "length_range": TEST_LENGTH_RANGE},
        },
        "known_simplifications_vs_original_pipeline": [
            "Positives are generated via a direct from-scratch construction ((01)^k) rather than "
            "the original FSA-sampling machinery -- exact (not approximate) for this task, since "
            "the grammar is a single repeated 2-token block with no branching.",
            "Hard negatives require n>=4 (k>=2 pairs); for n<4 no non-boundary adjacent-swap "
            "candidate exists, so such lengths are excluded from the hard-negative half only "
            "(uniform-random negatives are unaffected and still cover all lengths).",
            "No cross-split deduplication of positive examples (negligible given the string space "
            "size at these lengths, matching every prior fix experiment's own documented "
            "simplification).",
            "No log-probabilities.txt: not consumed by prepare_data.py and not needed for any of "
            "the four training objectives (matching modular-arithmetic-simple's fix).",
        ],
        "audits": {"train": train_audit, "validation-short": val_audit, "test": test_audit},
    }
    out_path = RESULTS / "fix_dataset_repeat_01.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
