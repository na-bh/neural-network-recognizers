"""Corrected-dataset experiment for binary-addition, BF2: shortcut-only
baseline ceiling on the corrected test set.

Evaluates THREE shortcut-only baselines on the CORRECTED test set (languages/
binary-addition-fixed/datasets/test, generated in BF1): (1) the task-specific
shortcut baseline already trained in Phase 1 P2 (analysis_outputs/final_
results/phase1_checkpoints/phase1_binaryaddition_shortcut_baseline.joblib,
NOT retrained here), and (2)/(3) ngram-logreg and bagcount-ffn baselines,
which did NOT previously exist for binary-addition (phase1_baselines_train.py
was only ever run for cycle-navigation/marked-reversal) -- TRAINED HERE, on
binary-addition's STANDARD training set (matching the marked-reversal fix's
convention of evaluating standard-trained baselines on the corrected test
set, not retraining baselines on corrected data), using the exact same
task-agnostic train_candidate_A/train_candidate_B infrastructure.

Theoretical shortcut-only ceiling, generalizing marked-reversal fix F2's
grammar_only_ceiling formula with a binary-addition-specific refinement:
a negative is "structurally OBVIOUS" (shortcut-detectable via structural
features alone -- invalid parse: wrong/missing operator or equals, or a
non-binary u_z) or "structurally AMBIGUOUS" (a fully valid parse --
Phase 1 P2 proved structural features [lengths, counts] carry ZERO signal
for correctness among valid-parse examples). ALL of BF1's hard-negative
half is structurally ambiguous by construction (a single bit-flip preserves
parse validity exactly); a small number of uniform-random-half examples may
ALSO be ambiguous by chance -- counted directly, not assumed.

Within the structurally-ambiguous population, LSB-parity (LSB(u_z) =
LSB(u_x) XOR LSB(u_y), a necessary-but-not-sufficient condition, NOT itself
a feature the trained shortcut_features baseline has access to, but a
theoretically available structural signal) additionally catches whichever
subset violates it -- BF1 found this to be ~25% of train's hard negatives
and ~5% of test's (shorter len_z inflates the train fraction). The ceiling
credits this subset with perfect detection; the REMAINING ambiguous
population gets chance (0.5).

ceiling = (n_positive + n_structurally_obvious + n_ambiguous_lsb_detectable
           + 0.5 * n_ambiguous_not_lsb_detectable) / n_total

Note this ceiling is a THEORETICAL upper bound assuming a shortcut detector
that also exploits LSB parity -- the ACTUALLY TRAINED shortcut_features
baseline (Phase 1 P2's feature set: lengths, counts, positions) does NOT
include an LSB-parity feature, so its measured accuracy is expected to fall
BELOW this ceiling on the hard-negative half; the gap is itself informative
(quantifies how much of the ceiling requires parity specifically, not just
generic structural features).

PYTHONPATH=src:analysis python analysis/fix_baseline_ceiling_binary_addition.py
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase1_baselines_train import (
    train_candidate_A, eval_candidate_A, train_candidate_B, eval_candidate_B,
    BagCountFFN, count_features, load_vocab, load_split,
)
from phase1_binaryaddition_baseline_design_f2 import shortcut_features
import phase2_targets as T

RESULTS = Path("analysis_outputs/final_results")
CKPT_DIR = RESULTS / "phase1_checkpoints"
TASK = "binary-addition"
FIXED_TEST_DIR = Path("languages/binary-addition-fixed/datasets/test")
LONG_SEQ_THRESH = 400
H_GRID = [8, 32, 128, 512]


def load_fixed_test():
    toks = (FIXED_TEST_DIR / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (FIXED_TEST_DIR / "labels.txt").read_text().splitlines()]
    kinds = (FIXED_TEST_DIR / "negative-kind.txt").read_text().splitlines()
    seqs = [line.split() for line in toks]
    return seqs, labels, kinds


def split_indices(labels, kinds):
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    uniform_idx = [i for i, (l, k) in enumerate(zip(labels, kinds)) if l == 0 and k == "uniform_random"]
    hard_idx = [i for i, (l, k) in enumerate(zip(labels, kinds)) if l == 0 and k == "hard_negative"]
    assert len(pos_idx) + len(uniform_idx) + len(hard_idx) == len(labels)
    return pos_idx, uniform_idx, hard_idx


def acc_on(correct, idx):
    if not idx:
        return float("nan"), 0
    return float(np.mean(correct[idx])), len(idx)


def evaluate_baseline(name, probs, labels, pos_idx, uniform_idx, hard_idx, seqs):
    labels_arr = np.asarray(labels, dtype=bool)
    preds = probs >= 0.5
    correct = preds == labels_arr
    overall_acc = float(np.mean(correct))
    pos_acc, n_pos = acc_on(correct, pos_idx)
    uniform_acc, n_uniform = acc_on(correct, uniform_idx)
    hard_acc, n_hard = acc_on(correct, hard_idx)
    long_idx = [i for i, s in enumerate(seqs) if len(s) >= LONG_SEQ_THRESH]
    long_acc, n_long = acc_on(correct, long_idx)
    return {
        "baseline": name, "n_total": len(labels),
        "overall_accuracy": overall_acc,
        "positive_accuracy": pos_acc, "n_positive": n_pos,
        "uniform_random_half_accuracy": uniform_acc, "n_uniform_random": n_uniform,
        "hard_negative_half_accuracy": hard_acc, "n_hard_negative": n_hard,
        "long_sequence_accuracy": long_acc, "n_long_sequences": n_long,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    seqs, labels, kinds = load_fixed_test()
    print(f"corrected test set: n={len(seqs)}", flush=True)
    pos_idx, uniform_idx, hard_idx = split_indices(labels, kinds)
    print(f"  n_positive={len(pos_idx)} n_uniform_random={len(uniform_idx)} n_hard_negative={len(hard_idx)}", flush=True)

    # ------------------------------------------------------------------
    # theoretical shortcut-only ceiling (computed from actual composition)
    # ------------------------------------------------------------------
    def is_structurally_ambiguous(seq):
        return T.binaddition_parse(seq)[3]  # valid-parse flag

    def is_lsb_parity_detectable(seq):
        return T.binaddition_lsb_correctness(seq) == 0

    neg_idx_all = uniform_idx + hard_idx
    ambiguous_flags = {i: is_structurally_ambiguous(seqs[i]) for i in neg_idx_all}
    lsb_flags = {i: (is_lsb_parity_detectable(seqs[i]) if ambiguous_flags[i] else False) for i in neg_idx_all}

    n_ambiguous_from_uniform = sum(1 for i in uniform_idx if ambiguous_flags[i])
    n_ambiguous_from_hard = sum(1 for i in hard_idx if ambiguous_flags[i])
    n_obvious_from_uniform = len(uniform_idx) - n_ambiguous_from_uniform
    n_obvious_from_hard = len(hard_idx) - n_ambiguous_from_hard
    n_structurally_obvious = n_obvious_from_uniform + n_obvious_from_hard

    n_ambiguous_lsb_detectable = sum(1 for i in neg_idx_all if ambiguous_flags[i] and lsb_flags[i])
    n_ambiguous_not_lsb_detectable = sum(1 for i in neg_idx_all if ambiguous_flags[i] and not lsb_flags[i])
    n_total = len(labels)

    theoretical_ceiling = (
        len(pos_idx) + n_structurally_obvious + n_ambiguous_lsb_detectable
        + 0.5 * n_ambiguous_not_lsb_detectable
    ) / n_total

    ceiling_report = {
        "n_total": n_total, "n_positive": len(pos_idx),
        "n_structurally_obvious_negatives": n_structurally_obvious,
        "structurally_obvious_breakdown": {
            "from_hard_negative_half": n_obvious_from_hard,
            "from_uniform_random_half": n_obvious_from_uniform,
        },
        "n_structurally_ambiguous_negatives": n_ambiguous_from_uniform + n_ambiguous_from_hard,
        "structurally_ambiguous_breakdown": {
            "from_hard_negative_half": n_ambiguous_from_hard,
            "from_uniform_random_half_by_chance": n_ambiguous_from_uniform,
        },
        "n_ambiguous_lsb_parity_detectable": n_ambiguous_lsb_detectable,
        "n_ambiguous_not_lsb_parity_detectable": n_ambiguous_not_lsb_detectable,
        "lsb_parity_detectable_fraction_of_ambiguous": (
            n_ambiguous_lsb_detectable / (n_ambiguous_lsb_detectable + n_ambiguous_not_lsb_detectable)
            if (n_ambiguous_lsb_detectable + n_ambiguous_not_lsb_detectable) else float("nan")
        ),
        "formula": "(n_positive + n_structurally_obvious + n_ambiguous_lsb_parity_detectable + "
                  "0.5 * n_ambiguous_not_lsb_parity_detectable) / n_total -- perfect on positives "
                  "and every negative a shortcut-only detector (parse validity + structural "
                  "features) can catch; ADDITIONALLY perfect on the subset of structurally-"
                  "ambiguous (valid-parse) negatives that also violate the necessary LSB-parity "
                  "condition; chance (0.5) on the remainder. This is a THEORETICAL upper bound "
                  "assuming access to LSB parity -- the ACTUALLY TRAINED shortcut_features "
                  "baseline (lengths/counts/positions only, Phase 1 P2) has NO parity feature, so "
                  "its measured accuracy is expected to fall below this ceiling on the hard-"
                  "negative half.",
        "theoretical_shortcut_only_ceiling": theoretical_ceiling,
        "note": (
            f"{n_ambiguous_from_hard}/{len(hard_idx)} of the hard-negative half is structurally "
            f"ambiguous by construction (should be ALL of it, i.e. {len(hard_idx)}); "
            f"{n_ambiguous_from_uniform}/{len(uniform_idx)} of the uniform-random half is ALSO "
            f"structurally ambiguous purely by chance (expected to be small/near-zero, checked "
            f"directly rather than assumed)."
        ),
    }
    print("\n=== theoretical shortcut-only ceiling ===")
    print(json.dumps(ceiling_report, indent=2, default=str))
    assert n_ambiguous_from_hard == len(hard_idx), (
        f"expected ALL hard-negative-half examples to be structurally ambiguous by construction, "
        f"got {n_ambiguous_from_hard}/{len(hard_idx)}"
    )

    # ------------------------------------------------------------------
    # (1) shortcut baseline (already trained, Phase 1 P2 -- no retraining)
    # ------------------------------------------------------------------
    sc_ckpt = joblib.load(CKPT_DIR / "phase1_binaryaddition_shortcut_baseline.joblib")
    X_sc = shortcut_features(seqs)
    probs_sc = sc_ckpt["model"].predict_proba(X_sc)[:, 1]
    sc_result = evaluate_baseline("shortcut", probs_sc, labels, pos_idx, uniform_idx, hard_idx, seqs)
    sc_result["source_checkpoint"] = str(CKPT_DIR / "phase1_binaryaddition_shortcut_baseline.joblib")
    sc_result["trained_on"] = "standard binary-addition training set (Butoi et al. mixed strategy, K-edit perturbations)"
    print("\n=== shortcut ===")
    print(json.dumps(sc_result, indent=2, default=str))

    # ------------------------------------------------------------------
    # (2) ngram-logreg baseline -- TRAINED HERE (did not previously exist
    # for binary-addition), on the STANDARD training set
    # ------------------------------------------------------------------
    print("\n=== training ngram-logreg on standard binary-addition training data ===", flush=True)
    train_seqs, train_labels = load_split(TASK, "train")
    val_seqs, val_labels = load_split(TASK, "validation-short")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tA = train_candidate_A(TASK, train_seqs, train_labels, val_seqs, val_labels)
    print(f"  selected_C={tA['selected_C']} n_features={tA['n_features']} "
          f"convergence_issues={tA['convergence_warnings'] or ['none']}", flush=True)
    ng_ckpt_path = CKPT_DIR / "phase1_binaryaddition_ngram_logreg.joblib"
    joblib.dump({"vectorizer": tA["vectorizer"], "model": tA["model"],
                "source_data": {"train": f"languages/{TASK}/main.tok",
                                "validation": f"languages/{TASK}/datasets/validation-short/main.tok"}},
               ng_ckpt_path)
    probs_ng = eval_candidate_A(tA["vectorizer"], tA["model"], seqs, labels)
    ng_result = evaluate_baseline("ngram-logreg", probs_ng, labels, pos_idx, uniform_idx, hard_idx, seqs)
    ng_result["source_checkpoint"] = str(ng_ckpt_path)
    ng_result["trained_on"] = "standard binary-addition training set (Butoi et al. mixed strategy, K-edit perturbations)"
    ng_result["selected_C"] = tA["selected_C"]
    print("\n=== ngram-logreg (evaluated on corrected test) ===")
    print(json.dumps(ng_result, indent=2, default=str))

    # ------------------------------------------------------------------
    # (3) bagcount-ffn baseline -- TRAINED HERE, swept over h
    # ------------------------------------------------------------------
    vocab = load_vocab(TASK)
    bagcount_results = {}
    for h in H_GRID:
        print(f"\n=== training bagcount-ffn h={h} on standard binary-addition training data ===", flush=True)
        tB = train_candidate_B(TASK, h, vocab, train_seqs, train_labels, val_seqs, val_labels)
        ckpt_path = CKPT_DIR / f"phase1_binaryaddition_bagcount_ffn_h{h}.pt"
        torch.save({"state_dict": tB["model"].state_dict(), "vocab": vocab, "hidden_units": h,
                    "source_data": {"train": f"languages/{TASK}/main.tok",
                                    "validation": f"languages/{TASK}/datasets/validation-short/main.tok"}},
                   ckpt_path)
        probs_bc = eval_candidate_B(tB["model"], seqs, vocab)
        bc_result = evaluate_baseline(f"bagcount-ffn-h{h}", probs_bc, labels, pos_idx, uniform_idx, hard_idx, seqs)
        bc_result["source_checkpoint"] = str(ckpt_path)
        bc_result["trained_on"] = "standard binary-addition training set (Butoi et al. mixed strategy, K-edit perturbations)"
        bc_result["num_epochs_run"] = tB["num_epochs_run"]
        bc_result["stopped_early"] = tB["stopped_early"]
        bc_result["convergence_issues"] = tB["issues"] or ["none"]
        bagcount_results[str(h)] = bc_result
        print(f"  h={h} epochs={tB['num_epochs_run']} stopped_early={tB['stopped_early']} "
              f"issues={tB['issues'] or 'none'}", flush=True)
        print(json.dumps(bc_result, indent=2, default=str))

    # ------------------------------------------------------------------
    # cross-baseline summary vs the theoretical ceiling
    # ------------------------------------------------------------------
    all_baselines = {"shortcut": sc_result, "ngram-logreg": ng_result,
                     **{f"bagcount-ffn-h{h}": bagcount_results[str(h)] for h in H_GRID}}
    summary_rows = []
    for name, r in all_baselines.items():
        summary_rows.append({
            "baseline": name, "overall_accuracy": r["overall_accuracy"],
            "hard_negative_half_accuracy": r["hard_negative_half_accuracy"],
            "exceeds_theoretical_ceiling": r["overall_accuracy"] > theoretical_ceiling + 0.01,
        })
    print("\n=== baseline summary vs theoretical ceiling ===")
    for row in summary_rows:
        print(f"  {row['baseline']}: overall={row['overall_accuracy']:.4f} "
              f"hard_neg={row['hard_negative_half_accuracy']:.4f} "
              f"(ceiling={theoretical_ceiling:.4f}, exceeds={row['exceeds_theoretical_ceiling']})")

    out = {
        "task": TASK,
        "experiment": "corrected-dataset-fix (BF2: shortcut-only baseline ceiling on corrected test data)",
        "description": (
            "Evaluates the shortcut baseline (Phase 1 P2, no retraining) plus NEWLY-TRAINED "
            "ngram-logreg and bagcount-ffn baselines (trained here on binary-addition's STANDARD "
            "training set -- these two baselines did not previously exist for this task) on the "
            "CORRECTED test set (BF1), split into positive/uniform-random/hard-negative subsets "
            "via BF1's negative-kind.txt side-channel. Also reports the theoretical shortcut-only "
            "ceiling, refined with an explicit LSB-parity-detectable credit."
        ),
        "corrected_test_set": str(FIXED_TEST_DIR),
        "test_composition": {
            "n_total": n_total, "n_positive": len(pos_idx),
            "n_uniform_random": len(uniform_idx), "n_hard_negative": len(hard_idx),
        },
        "theoretical_shortcut_only_ceiling": ceiling_report,
        "baselines": {
            "shortcut": sc_result, "ngram_logreg": ng_result, "bagcount_ffn": bagcount_results,
        },
        "summary_vs_ceiling": summary_rows,
    }
    out_path = RESULTS / "fix_baseline_ceiling_binary_addition.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
