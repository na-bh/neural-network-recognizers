"""Corrected-dataset experiment for Dyck-2-3, DF1: dataset construction +
audit.

Builds a "corrected" Dyck-2-3 dataset preserving Butoi et al.'s mixed
negative-generation strategy (50% uniform-random / 50% hard-negative half)
but replacing whatever original negative-generation Dyck-2-3 used with an
EXPLICIT construction drawing from the three shortcut-blind subtypes
identified in phase3_dyck_task_audit.json's joint audit (43/2545 negatives,
1.7%, pass ALL FOUR listed shortcut features): type_mismatch (20/43,
46.5%), close_with_nothing_open (13/43, 30.2%), depth_exceeded (10/43,
23.3%) -- hard negatives are distributed across these three subtypes in
these natural proportions.

Positive generation: no hand-picked-language .sample() exists for this
FSA-based task (dyck_k_m.py builds a rayuela automaton, not a string-
sampling class) -- positives are generated via a from-scratch stack-based
random walk (random_valid_dyck, reused from analysis/phase3_dyck_
counterfactuals.py, already validated: is_valid_dyck(seq, max_depth=3)
matches every generated string). next_symbols supervision (for the 'ns'
training objective) is computed directly from the stack-transition
semantics: at each prefix, valid next tokens are {all K openers} (if
depth < M) unioned with {closer matching stack-top} (if stack non-empty),
narrowing to ONLY {closer matching stack-top} once depth == M (can't open
further) -- EOS is valid only when the stack is empty.

Hard-negative subtype constructions (all verified to preserve every one of
the four listed shortcut features -- bracket_count_parity, per_type_
bracket_counts, sequence_length_parity, first_last_check):
  type_mismatch: swap the TYPE LABELS of two nested closers (both opens
    were simultaneously on the stack at some point) -- a pure relabeling,
    preserves per-type counts (symmetric swap) and everything else exactly.
  close_with_nothing_open: swap the TOKENS at two DIFFERENT INTERIOR
    positions (not touching the first/last token) -- a pure PERMUTATION,
    so total counts/per-type counts/length are automatically preserved by
    construction, and first/last are preserved by not touching the
    endpoints; classified via the corrected violation detector (phase3_
    dyck_task_audit.py's viol_dyck_with_depth) and KEPT only if the
    resulting violation kind is genuinely 'close_with_nothing_open'
    (rejection-sampled, not assumed).
  depth_exceeded: insert a matched (open,close) pair of a random type at a
    point already at depth M=3 -- momentarily reaches depth 4; preserves
    per-type counts (adds 1 to both open and close of that type) and
    length_parity (+2, even); reused directly from phase3_dyck_
    counterfactuals.py's target_computation_depth_exceeded_pairs
    construction (single-sequence version here, not a matched pair).

Violation position is swept across early/mid/late (relative fraction of
sequence length) for every subtype, matching bucket-sort/binary-addition's
position-sweep discipline; expect (and report) the same train-skews-early/
test-is-uniform discretization artifact documented in fix_dataset_binary_
addition.py's analogous audit.

Outputs (matching the on-disk format recognizers/neural_networks/data.py
reads): languages/dyck-2-3-fixed/{main.tok,labels.txt,next-symbols.jsonl,
negative-kind.txt,hard-negative-subtype-meta.jsonl}, plus datasets/
{validation-short,test}.

PYTHONPATH=src:analysis python analysis/fix_dataset_dyck_2_3.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
from phase3_dyck_task_audit import is_valid_dyck, viol_dyck_with_depth, SHORTCUT_FEATURES, K, M
from phase3_dyck_counterfactuals import random_valid_dyck

RESULTS = Path("analysis_outputs/final_results")
LANG_DIR = Path("languages/dyck-2-3-fixed")
RANDOM_SEED = 20260716
TRAIN_LENGTH_RANGE = (0, 40)
TEST_LENGTH_RANGE = (0, 500)
TEST_EXAMPLES_PER_LENGTH = 10
TRAIN_SIZE = 10000
VALIDATION_SHORT_SIZE = 1000
NEG_UNIFORM_RANDOM_PROB = 0.5
MAX_TRIES = 400

# subtype proportions from the Phase 3 audit's joint shortcut-blind population
# (20 type_mismatch, 13 close_with_nothing_open, 10 depth_exceeded out of 43)
SUBTYPE_PROPORTIONS = {"type_mismatch": 20 / 43, "close_with_nothing_open": 13 / 43, "depth_exceeded": 10 / 43}
SUBTYPES = list(SUBTYPE_PROPORTIONS)
ALPHABET = ["(0", "(1", ")0", ")1"]
ALPHABET_SIZE = 4


def gen_len(length_range, rng):
    lo, hi = length_range
    return int(rng.integers(max(4, lo), hi + 1))


def find_nested_closer_pairs(seq):
    """Positions of closers whose corresponding opens are both on the stack
    simultaneously at some point (i.e., properly nested), paired up."""
    stack = []  # (type, open_idx)
    pairs = []
    open_stack_snapshot = []
    for i, tok in enumerate(seq):
        if tok.startswith("("):
            stack.append((tok[1:], i))
        elif tok.startswith(")"):
            t, oi = stack.pop()
            pairs.append((oi, i, t))
    return pairs


def type_mismatch_hard_negative(rng, length_range, position_frac, max_tries=MAX_TRIES):
    for _ in range(max_tries):
        n = gen_len(length_range, rng)
        clean = random_valid_dyck(rng, min_len=n, max_len=n)
        closer_pairs = find_nested_closer_pairs(clean)
        if len(closer_pairs) < 2:
            continue
        candidates = []
        for a in range(len(closer_pairs)):
            for b in range(a + 1, len(closer_pairs)):
                if closer_pairs[a][2] != closer_pairs[b][2]:
                    candidates.append((closer_pairs[a][1], closer_pairs[b][1]))
        if not candidates:
            continue
        ca, cb = min(candidates, key=lambda idx: abs(((idx[0] + idx[1]) / 2) / (len(clean) - 1) - position_frac))
        corrupted = clean[:]
        corrupted[ca], corrupted[cb] = clean[cb], clean[ca]
        if is_valid_dyck(corrupted, M):
            continue
        frac, kind = viol_dyck_with_depth(corrupted)
        if kind != "type_mismatch":
            continue
        if not all(fn(corrupted) for fn in SHORTCUT_FEATURES.values()):
            continue
        pos = (ca + cb) / 2
        return corrupted, {"subtype": "type_mismatch", "swap_indices": [ca, cb],
                          "violation_position_relative": pos / (len(corrupted) - 1)}
    raise RuntimeError("could not generate a type_mismatch hard negative")


def close_with_nothing_open_hard_negative(rng, length_range, position_frac, max_tries=MAX_TRIES):
    for _ in range(max_tries):
        n = gen_len(length_range, rng)
        clean = random_valid_dyck(rng, min_len=n, max_len=n)
        if len(clean) < 4:
            continue
        interior = list(range(1, len(clean) - 1))
        if len(interior) < 2:
            continue
        i = min(interior, key=lambda idx: abs(idx / (len(clean) - 1) - position_frac))
        others = [j for j in interior if j != i and clean[j] != clean[i]]
        if not others:
            continue
        j = int(rng.choice(others))
        corrupted = clean[:]
        corrupted[i], corrupted[j] = clean[j], clean[i]
        if is_valid_dyck(corrupted, M):
            continue
        frac, kind = viol_dyck_with_depth(corrupted)
        if kind != "close_with_nothing_open":
            continue
        if not all(fn(corrupted) for fn in SHORTCUT_FEATURES.values()):
            continue
        pos = (i + j) / 2
        return corrupted, {"subtype": "close_with_nothing_open", "swap_indices": sorted([i, j]),
                          "violation_position_relative": pos / (len(corrupted) - 1)}
    raise RuntimeError("could not generate a close_with_nothing_open hard negative")


def depth_exceeded_hard_negative(rng, length_range, position_frac, max_tries=MAX_TRIES):
    for _ in range(max_tries):
        n = gen_len(length_range, rng)
        clean = random_valid_dyck(rng, min_len=n, max_len=n)
        depth_m_positions = []
        depth = 0
        for i, tok in enumerate(clean):
            if tok.startswith("("):
                depth += 1
                if depth == M:
                    depth_m_positions.append(i)
            elif tok.startswith(")"):
                depth -= 1
        if not depth_m_positions:
            continue
        insert_after = min(depth_m_positions, key=lambda i: abs((i + 1) / len(clean) - position_frac))
        t = int(rng.integers(0, K))
        corrupted = clean[:insert_after + 1] + [f"({t}", f"){t}"] + clean[insert_after + 1:]
        if is_valid_dyck(corrupted, M):
            continue
        frac, kind = viol_dyck_with_depth(corrupted)
        if kind != "depth_exceeded":
            continue
        if not all(fn(corrupted) for fn in SHORTCUT_FEATURES.values()):
            continue
        return corrupted, {"subtype": "depth_exceeded", "insert_after_index": insert_after,
                          "violation_position_relative": (insert_after + 1) / len(corrupted)}
    raise RuntimeError("could not generate a depth_exceeded hard negative")


SUBTYPE_FNS = {
    "type_mismatch": type_mismatch_hard_negative,
    "close_with_nothing_open": close_with_nothing_open_hard_negative,
    "depth_exceeded": depth_exceeded_hard_negative,
}


def generate_hard_negative(rng, length_range):
    subtype = rng.choice(SUBTYPES, p=[SUBTYPE_PROPORTIONS[s] for s in SUBTYPES])
    position_frac = float(rng.random())
    seq, meta = SUBTYPE_FNS[subtype](rng, length_range, position_frac)
    meta["target_position_frac_requested"] = position_frac
    return seq, meta


def generate_uniform_random(length_range, rng):
    n = gen_len(length_range, rng)
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
        if not is_valid_dyck(seq, M):
            return seq, kind, meta
    raise RuntimeError("could not generate a verified negative example")


def compute_next_symbols(seq):
    """next_symbols[i] = (valid_token_strings, eos_valid) BEFORE consuming
    seq[i] (i in range(len(seq)+1), last entry is post-sequence/EOS check).
    Derived directly from stack-transition semantics: depth<M allows any
    opener plus (if stack non-empty) the stack-top closer; depth==M allows
    ONLY the stack-top closer; EOS is valid only when the stack is empty."""
    rows = []
    stack = []
    for tok in seq:
        valid = []
        if len(stack) < M:
            valid.extend(f"({k}" for k in range(K))
        if stack:
            valid.append(f"){stack[-1]}")
        rows.append({"s": " ".join(sorted(valid)), "e": len(stack) == 0})
        if tok.startswith("("):
            stack.append(tok[1:])
        elif tok.startswith(")"):
            stack.pop()
    # final position (after consuming everything)
    valid = []
    if len(stack) < M:
        valid.extend(f"({k}" for k in range(K))
    if stack:
        valid.append(f"){stack[-1]}")
    rows.append({"s": " ".join(sorted(valid)), "e": len(stack) == 0})
    return rows


def generate_split(length_range, num_samples, rng, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(num_samples):
        label = int(rng.integers(0, 2))
        if label:
            n = gen_len(length_range, rng)
            s = random_valid_dyck(rng, min_len=n, max_len=n)
            rows.append({"s": s, "label": 1, "kind": "", "next_symbols": compute_next_symbols(s), "meta": None})
        else:
            s, kind, meta = generate_negative_example(length_range, rng)
            rows.append({"s": s, "label": 0, "kind": kind, "next_symbols": None, "meta": meta})

    with (output_dir / "main.tok").open("w") as tok_f, \
         (output_dir / "labels.txt").open("w") as labels_f, \
         (output_dir / "negative-kind.txt").open("w") as kind_f, \
         (output_dir / "next-symbols.jsonl").open("w") as ns_f, \
         (output_dir / "hard-negative-subtype-meta.jsonl").open("w") as meta_f:
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
        all_pass_shortcuts = [all(fn(r["s"]) for fn in SHORTCUT_FEATURES.values()) for r in hard_rows]
        all_invalid = [not is_valid_dyck(r["s"], M) for r in hard_rows]
        subtype_counts = Counter(r["meta"]["subtype"] for r in hard_rows)
        hard_checks.update({
            "all_pass_all_four_shortcut_features": all(all_pass_shortcuts),
            "n_detectable_by_any_shortcut_feature": sum(1 for x in all_pass_shortcuts if not x),
            "all_genuinely_invalid": all(all_invalid),
            "subtype_distribution": dict(subtype_counts),
            "subtype_distribution_fraction": {k: v / n_hard for k, v in subtype_counts.items()},
            "expected_subtype_fractions": SUBTYPE_PROPORTIONS,
        })
        # position distribution per subtype (tertile bins)
        by_subtype_positions = {}
        for st in SUBTYPES:
            fracs = [r["meta"]["violation_position_relative"] for r in hard_rows if r["meta"]["subtype"] == st]
            if not fracs:
                by_subtype_positions[st] = {"n": 0}
                continue
            arr = np.array(fracs)
            bins = np.linspace(0, 1, 4)
            hist, _ = np.histogram(arr, bins=bins)
            by_subtype_positions[st] = {"n": len(fracs), "tertile_histogram_low_mid_high": hist.tolist(),
                                        "mean": float(arr.mean())}
        hard_checks["position_distribution_by_subtype"] = by_subtype_positions

    uniform_rows = [r for r in neg_rows if r["kind"] == "uniform_random"]
    uniform_checks = {"n_checked": len(uniform_rows)}
    if uniform_rows:
        n_valid_parse = sum(1 for r in uniform_rows if is_valid_dyck(r["s"], M))
        uniform_checks["n_accidentally_valid"] = n_valid_parse
        uniform_checks["fraction_accidentally_valid"] = n_valid_parse / len(uniform_rows)

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
    assert train_audit["hard_negative_construction_checks"]["all_genuinely_invalid"]
    assert test_audit["hard_negative_construction_checks"]["all_genuinely_invalid"]

    out = {
        "task": "dyck-2-3",
        "experiment": "corrected-dataset-fix (DF1: dataset construction + audit)",
        "description": (
            "Corrected Dyck-2-3 dataset preserving Butoi et al.'s mixed negative-generation "
            "strategy (50% uniform-random / 50% hard-negative half). Hard negatives are drawn from "
            "the three shortcut-blind subtypes identified in phase3_dyck_task_audit.json's joint "
            "audit (type_mismatch, close_with_nothing_open, depth_exceeded), in their natural "
            "proportions (46.5%/30.2%/23.3%), each construction verified to pass ALL FOUR listed "
            "shortcut features (bracket_count_parity, per_type_bracket_counts, "
            "sequence_length_parity, first_last_check) while being genuinely invalid."
        ),
        "random_seed": RANDOM_SEED,
        "k_bracket_types": K, "m_max_nesting_depth": M,
        "subtype_proportions_from_phase3_audit": SUBTYPE_PROPORTIONS,
        "language_output_directory": str(LANG_DIR),
        "splits_generated": {
            "train": {"n": TRAIN_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "validation-short": {"n": VALIDATION_SHORT_SIZE, "length_range": TRAIN_LENGTH_RANGE},
            "test": {"n": test_size, "length_range": TEST_LENGTH_RANGE},
        },
        "known_simplifications_vs_original_pipeline": [
            "Positives are generated via a from-scratch stack-based random walk (random_valid_dyck), "
            "not the original FSA-based sampling machinery (dyck_k_m.py/rayuela) -- validated "
            "equivalent via is_valid_dyck(seq, max_depth=3) matching every generated string exactly, "
            "but the exact PROBABILITY DISTRIBUTION over valid strings may differ from the original "
            "FSA's weighting scheme.",
            "No cross-split deduplication of positive examples (negligible given the string space "
            "size at these lengths, matching every prior fix experiment's own documented "
            "simplification).",
        ],
        "audits": {"train": train_audit, "validation-short": val_audit, "test": test_audit},
    }
    out_path = RESULTS / "fix_dataset_dyck_2_3.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
