"""Corrected-dataset experiment for bucket-sort, BSF1: dataset construction
+ audit.

Task: w#w' (5-symbol alphabet '1'..'5' plus marker '#'), positive iff there
is exactly one marker and w' == sorted(w) (src/recognizers/hand_picked_
languages/bucket_sort.py's _is_positive/_w_to_string/_sort, confirmed
directly).

Motivation (per Phase 3's finding this experiment tests): bucket-sort is
STRUCTURALLY DIFFERENT from the marker-family tasks (marked-reversal/
marked-copy/odds-first) where target computation is uniformly absent --
Phase 3 causal patching found 5/8 cells show COMPREHENSIVE (position-
independent) sort-order verification under STANDARD training already, with
2/8 position-specific and 1/8 null. Bucket-sort's target computation is
LOCALLY VERIFIABLE (checking whether adjacent post-marker positions are in
order is an aggregate-solvable check), unlike modular-arithmetic-simple's
non-locally-verifiable order-sensitive accumulation. This experiment tests
whether corrected training pushes the 2 partial cells to comprehensive
verification and whether it produces balanced-discrimination cells
exceeding the shortcut ceiling.

Confound this construction closes (the key methodological improvement over
the original Butoi et al. distribution, per Phase 1's audit): 99.8% of
balanced-halves single-marker negatives are detectable by a content-
multiset check alone (do the two halves contain the same multiset of
symbols?) -- a NECESSARY-but-not-sufficient condition for a genuine sort
match that requires no sort-order computation at all. Every hard negative
constructed here is verified to pass is_multiset_matched (both halves have
IDENTICAL multisets) alongside the other three listed shortcut features.

Four listed shortcut features audited (each a trivial bag-of-counts/
position check, no LIFO/comparison structure): marker_count_is_one,
marker_centered (marker at the exact positional midpoint), segment_
lengths_balanced (n_before == n_after -- numerically identical to
marker_centered for exactly-one-marker sequences, kept as a separate named
feature to match the four-feature list exactly, mirroring the redundancy
already present in Phase 1's marker-family baseline reuse), is_multiset_
matched.

Negative construction (50% uniform-random / 50% hard-negative, preserving
Butoi et al.'s mixed strategy):
  uniform_random: sample uniformly from the alphabet (6 tokens), matching
    every prior fix experiment.
  hard_negative: take a genuine positive w#sorted(w), then swap TWO
    ADJACENT positions in the SORTED half whose values differ (since the
    half is sorted ascending, any such adjacent differing pair is
    NECESSARILY in strictly-increasing order before the swap, so the swap
    ALWAYS creates a genuine descent -- the "swap w'[i],w'[i+1] where
    w'[i]<w'[i+1]" condition is automatically satisfied by construction,
    not just checked). A swap of two elements can never change the
    half's multiset, and never changes length, marker count, or marker
    position -- so ALL FOUR shortcut features are preserved by
    construction (verified empirically below, not just argued).

Position-sweep discipline: the swap position (relative fraction along the
sorted half) is drawn via a FIXED-target-tertile-then-retry strategy (the
lesson learned from modular-arithmetic-simple's MF1 revision, applied here
PREEMPTIVELY rather than discovered via a second iteration): draw a target
tertile (low/mid/high) once per example, then retry with a fresh random w
(not a re-rolled target) until a distinct-adjacent-pair swap candidate
lands in that tertile. This task's OWN structural constraint on candidate
availability differs from modular-arithmetic-simple's: because the sorted
half is drawn from only 5 symbol values, a long random half has at most 4
"value-transition" adjacent-differing-pairs (candidates) total, however
their positions are governed by the RANDOM composition of the draw (not
structurally capped below any tertile the way modular-arithmetic-simple's
non-adjacent-pair first-index was) -- so retrying with fresh random draws,
not re-rolling the target, is expected to reach all three tertiles for
long sequences too. Verified empirically below.

Outputs (matching recognizers/neural_networks/data.py's expected format):
languages/bucket-sort-fixed/{main.tok,labels.txt,next-symbols.jsonl,
negative-kind.txt,hard-negative-meta.jsonl}, plus datasets/
{validation-short,test}.

PYTHONPATH=src:analysis python analysis/fix_dataset_bucket_sort.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

RESULTS = Path("analysis_outputs/final_results")
LANG_DIR = Path("languages/bucket-sort-fixed")
RANDOM_SEED = 20260718
TRAIN_LENGTH_RANGE = (0, 40)   # total sequence length (w + '#' + sorted(w))
TEST_LENGTH_RANGE = (0, 500)
TEST_EXAMPLES_PER_LENGTH = 10
TRAIN_SIZE = 10000
VALIDATION_SHORT_SIZE = 1000
NEG_UNIFORM_RANDOM_PROB = 0.5
MAX_TRIES = 400
NUM_SYMBOLS = 5
CONTENT_ALPHABET = [str(i) for i in range(1, NUM_SYMBOLS + 1)]
MARKER = "#"
ALPHABET = CONTENT_ALPHABET + [MARKER]
ALPHABET_SIZE = len(ALPHABET)
TERTILES = ["low", "mid", "high"]


def rand_symbol(rng):
    return CONTENT_ALPHABET[int(rng.integers(0, NUM_SYMBOLS))]


def gen_n(length_range, rng):
    """Matches LengthRestrictedBucketSort's own total-length -> n mapping
    exactly (src/recognizers/hand_picked_languages/bucket_sort.py):
    min_n = ceil(max(0, lo-1)/2), max_n = floor(max(0, hi-1)/2), n sampled
    uniformly in [min_n, max_n]."""
    lo, hi = length_range
    min_n = -(-max(0, lo - 1) // 2)  # ceil division
    max_n = max(0, hi - 1) // 2
    return int(rng.integers(min_n, max_n + 1))


def categorize_tertile(frac):
    if frac < 1 / 3:
        return "low"
    elif frac < 2 / 3:
        return "mid"
    else:
        return "high"


def marker_count_is_one(seq):
    return seq.count(MARKER) == 1


def _marker_index(seq):
    idx = [i for i, t in enumerate(seq) if t == MARKER]
    return idx[0] if len(idx) == 1 else None


def marker_centered(seq):
    m = _marker_index(seq)
    if m is None:
        return False
    return m == len(seq) - 1 - m


def segment_lengths_balanced(seq):
    m = _marker_index(seq)
    if m is None:
        return False
    n_before, n_after = m, len(seq) - 1 - m
    return n_before == n_after


def is_multiset_matched(seq):
    m = _marker_index(seq)
    if m is None:
        return False
    before, after = seq[:m], seq[m + 1:]
    return Counter(before) == Counter(after)


SHORTCUT_FEATURES = {
    "marker_count_is_one": marker_count_is_one,
    "marker_centered": marker_centered,
    "segment_lengths_balanced": segment_lengths_balanced,
    "is_multiset_matched": is_multiset_matched,
}


def random_valid_positive(n, rng):
    w = [rand_symbol(rng) for _ in range(n)]
    return w + [MARKER] + sorted(w)


def generate_hard_negative(rng, length_range, max_tries=MAX_TRIES):
    """Fixed-target-tertile-then-retry (see module docstring). For n==2,
    only ONE adjacent pair exists (indices 0,1); its relative position is
    forced to frac=0.0 ("low") regardless of the requested target -- this
    edge case is accepted and reported (short_sequence_constrained), not
    silently retried away, matching modular-arithmetic-simple's k==2
    handling."""
    target_tertile = TERTILES[int(rng.integers(0, 3))]
    for _ in range(max_tries):
        n = gen_n(length_range, rng)
        if n < 2:
            continue
        w = [rand_symbol(rng) for _ in range(n)]
        s = sorted(w)
        candidates = [i for i in range(n - 1) if s[i] != s[i + 1]]
        if not candidates:
            continue  # every adjacent pair equal (all n symbols identical) -- retry
        if n == 2:
            i = candidates[0]
            constrained = True
            frac = 0.0
        else:
            denom = n - 2
            def frac_of(idx):
                return idx / denom
            bucket = [idx for idx in candidates if categorize_tertile(frac_of(idx)) == target_tertile]
            if not bucket:
                continue  # this draw's transitions don't reach the target tertile -- retry
            i = int(rng.choice(bucket))
            constrained = False
            frac = frac_of(i)
        corrupted_sorted = s[:]
        corrupted_sorted[i], corrupted_sorted[i + 1] = corrupted_sorted[i + 1], corrupted_sorted[i]
        corrupted = w + [MARKER] + corrupted_sorted
        meta = {
            "n_half": n, "swap_indices_in_sorted_half": [i, i + 1],
            "short_sequence_constrained": constrained,
            "violation_position_relative": frac,
            "violation_position_tertile": categorize_tertile(frac) if not constrained else "low (forced, n==2)",
            "target_tertile_requested": target_tertile,
        }
        return corrupted, meta
    raise RuntimeError("could not generate a hard negative")


def generate_uniform_random(length_range, rng):
    total_len = 2 * gen_n(length_range, rng) + 1
    return [ALPHABET[int(rng.integers(0, ALPHABET_SIZE))] for _ in range(total_len)]


def is_positive(seq):
    m = _marker_index(seq)
    if m is None:
        return False
    return seq[m + 1:] == sorted(seq[:m])


def propose_negative(length_range, rng):
    if rng.random() < NEG_UNIFORM_RANDOM_PROB:
        return generate_uniform_random(length_range, rng), "uniform_random", None
    else:
        seq, meta = generate_hard_negative(rng, length_range)
        return seq, "hard_negative", meta


def generate_negative_example(length_range, rng, max_tries=MAX_TRIES):
    for _ in range(max_tries):
        seq, kind, meta = propose_negative(length_range, rng)
        if not is_positive(seq):
            return seq, kind, meta
    raise RuntimeError("could not generate a verified negative example")


def compute_next_symbols(seq):
    """Rows correspond to predicting each token (state BEFORE consuming it),
    plus one final row after the whole sequence -- len(seq)+1 rows total,
    matching BucketSort._w_to_next_symbols exactly: while emitting w AND
    the marker (len(w)+1 positions), ANY token is valid (the marker's
    position/whether-it-has-appeared-yet is unconstrained until it is
    actually placed); once the marker has been consumed, each remaining
    position is FULLY DETERMINED by sorted(w) (singleton valid set); EOS
    is valid ONLY at the true end (bucket-sort has no early-termination
    positive, matching _is_positive's exact-match requirement)."""
    m = seq.index(MARKER)
    w, sorted_w = seq[:m], seq[m + 1:]
    all_tokens = " ".join(sorted(ALPHABET))
    rows = [{"s": all_tokens, "e": False} for _ in range(len(w) + 1)]
    rows += [{"s": tok, "e": False} for tok in sorted_w]
    rows.append({"s": "", "e": True})
    return rows


def generate_split(length_range, num_samples, rng, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(num_samples):
        label = int(rng.integers(0, 2))
        if label:
            n = gen_n(length_range, rng)
            s = random_valid_positive(n, rng)
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

    hard_checks = {"n_checked": n_hard}
    if hard_rows:
        per_feature_pass = {
            name: [fn(r["s"]) for r in hard_rows] for name, fn in SHORTCUT_FEATURES.items()
        }
        all_pass = [all(per_feature_pass[name][i] for name in SHORTCUT_FEATURES) for i in range(n_hard)]
        all_invalid = [not is_positive(r["s"]) for r in hard_rows]
        n_from_short = sum(1 for r in hard_rows if r["meta"]["short_sequence_constrained"])
        hard_checks.update({
            "per_shortcut_feature_pass_rate": {name: float(np.mean(vals)) for name, vals in per_feature_pass.items()},
            "all_pass_all_four_shortcut_features": all(all_pass),
            "n_detectable_by_any_shortcut_feature": sum(1 for x in all_pass if not x),
            "multiset_confound_closed": all(per_feature_pass["is_multiset_matched"]),
            "all_genuinely_invalid": all(all_invalid),
            "n_from_short_sequence_constrained (n_half==2)": n_from_short,
            "fraction_short_sequence_constrained": n_from_short / n_hard,
        })
        fracs = [r["meta"]["violation_position_relative"] for r in hard_rows]
        arr = np.array(fracs)
        bins = np.linspace(0, 1, 4)
        hist, _ = np.histogram(arr, bins=bins)
        hmin, hmax = int(hist.min()), int(hist.max())
        hard_checks["position_relative_tertile_histogram_low_mid_high"] = hist.tolist()
        hard_checks["position_relative_mean"] = float(arr.mean())
        hard_checks["tertiles_within_20_percent_of_each_other"] = bool(hmin >= 0.8 * hmax)
        n_half_dist = Counter(r["meta"]["n_half"] for r in hard_rows)
        hard_checks["n_half_distribution"] = dict(sorted(n_half_dist.items()))

    uniform_rows = [r for r in neg_rows if r["kind"] == "uniform_random"]
    uniform_checks = {"n_checked": len(uniform_rows)}
    if uniform_rows:
        n_valid_parse = sum(1 for r in uniform_rows if is_positive(r["s"]))
        uniform_checks["n_accidentally_valid"] = n_valid_parse
        uniform_checks["fraction_accidentally_valid"] = n_valid_parse / len(uniform_rows)
        # expected distribution of structural violations (part e): how many
        # uniform-random negatives coincidentally satisfy EACH shortcut
        # feature (qualitative sanity check, not a pass/fail gate)
        uniform_checks["per_shortcut_feature_coincidental_satisfy_rate"] = {
            name: float(np.mean([fn(r["s"]) for r in uniform_rows])) for name, fn in SHORTCUT_FEATURES.items()
        }
        uniform_checks["coincidentally_passes_all_four_fraction"] = float(np.mean(
            [all(fn(r["s"]) for fn in SHORTCUT_FEATURES.values()) for r in uniform_rows]
        ))

    return {
        "split": split_name, "n_total": n_total, "n_positive": n_pos, "n_negative": n_neg,
        "positive_fraction": n_pos / n_total,
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

    assert train_audit["hard_negative_construction_checks"]["all_pass_all_four_shortcut_features"]
    assert test_audit["hard_negative_construction_checks"]["all_pass_all_four_shortcut_features"]
    assert train_audit["hard_negative_construction_checks"]["multiset_confound_closed"]
    assert test_audit["hard_negative_construction_checks"]["multiset_confound_closed"]
    assert train_audit["hard_negative_construction_checks"]["all_genuinely_invalid"]
    assert test_audit["hard_negative_construction_checks"]["all_genuinely_invalid"]

    print("\n=== position-distribution uniformity check (test set) ===", flush=True)
    for split_name, audit in [("train", train_audit), ("validation-short", val_audit), ("test", test_audit)]:
        hc = audit["hard_negative_construction_checks"]
        print(f"  {split_name}: tertiles_within_20_percent="
              f"{hc['tertiles_within_20_percent_of_each_other']} hist="
              f"{hc['position_relative_tertile_histogram_low_mid_high']} mean={hc['position_relative_mean']:.4f}",
              flush=True)
    assert test_audit["hard_negative_construction_checks"]["tertiles_within_20_percent_of_each_other"], (
        "test-set swap-position distribution must be uniform (within 20% tertile tolerance)"
    )

    out = {
        "task": "bucket-sort",
        "experiment": "corrected-dataset-fix (BSF1: dataset construction + audit)",
        "description": (
            "Corrected bucket-sort dataset preserving Butoi et al.'s mixed negative-generation "
            "strategy (50% uniform-random / 50% hard-negative half). Hard negatives are constructed "
            "via an adjacent-symbol swap in the sorted post-marker half that creates a genuine "
            "descent (breaks sort order) while preserving marker count/position, segment-length "
            "balance, and -- the key methodological improvement over the original distribution -- "
            "content-multiset match between the two halves (99.8% of the original balanced-halves "
            "single-marker negatives were multiset-detectable per Phase 1's audit; every hard "
            "negative here is verified multiset-matched by construction)."
        ),
        "random_seed": RANDOM_SEED,
        "num_symbols": NUM_SYMBOLS, "alphabet": ALPHABET,
        "language_output_directory": str(LANG_DIR),
        "splits_generated": {
            "train": {"n": TRAIN_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "validation-short": {"n": VALIDATION_SHORT_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "test": {"n": test_size, "length_range": TEST_LENGTH_RANGE},
        },
        "known_simplifications_vs_original_pipeline": [
            "Positives are generated via a direct from-scratch construction (w + '#' + sorted(w)), "
            "not the original WeightedLanguage sampling machinery -- validated equivalent via "
            "is_positive matching BucketSort._is_positive's exact semantics, but the exact "
            "PROBABILITY DISTRIBUTION over valid strings (BucketSort samples n uniformly then w "
            "uniformly, matched here exactly via gen_n's identical ceil/floor mapping) should be "
            "equivalent, not merely similar.",
            "No cross-split deduplication of positive examples (negligible given the string space "
            "size at these lengths, matching every prior fix experiment's own documented "
            "simplification).",
        ],
        "position_sweep_construction_note": (
            "Swap position uses a FIXED-target-tertile-then-retry strategy (drawn once per example, "
            "held fixed across retries with fresh random w draws), applied preemptively based on the "
            "lesson learned in modular-arithmetic-simple's MF1 revision (re-rolling the target on "
            "every retry silently drifts away from whichever tertile has a lower per-attempt success "
            "rate). Unlike modular-arithmetic-simple, bucket-sort's candidate swap positions are not "
            "structurally capped below any tertile for small n (see module docstring) -- the "
            "resulting distributions are reported explicitly below for both train and test."
        ),
        "audits": {"train": train_audit, "validation-short": val_audit, "test": test_audit},
    }
    out_path = RESULTS / "fix_dataset_bucket_sort.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
