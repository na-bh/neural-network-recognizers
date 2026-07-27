"""Corrected-dataset experiment, OF2: shortcut-only baseline ceiling on the
corrected test set.

Odds-first has the SAME marker structure as marked-copy/marked-reversal
(single '#' separating two equal-length halves) -- only the content
transform differs (deinterleave vs copy/reversal), which the
marker_shortcut_features feature set (analysis/phase1_b3_compare.py: marker
count, marker relative offset from center) never encodes at all. Reused
directly, unchanged. No pre-existing phase1 checkpoints exist for
odds-first, so all three baselines (shortcut, ngram-logreg, bagcount-ffn)
are TRAINED HERE on odds-first's STANDARD training data.

PYTHONPATH=src:analysis python analysis/fix_baseline_ceiling_odds_first.py
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase1_b3_compare import marker_shortcut_features, train_marker_shortcut
from phase1_baselines_train import (
    train_candidate_A, eval_candidate_A, train_candidate_B, eval_candidate_B,
    load_vocab, load_split,
)

RESULTS = Path("analysis_outputs/final_results")
CKPT_DIR = RESULTS / "phase1_checkpoints"
TASK = "odds-first"
FIXED_TEST_DIR = Path("languages/odds-first-fixed/datasets/test")
LONG_SEQ_THRESH = 400
H_GRID = [8, 32, 128, 512]
MARKER = "#"


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


def is_structurally_ambiguous(seq, marker=MARKER):
    """Exactly one marker AND balanced halves -- indistinguishable from a
    genuine positive using ONLY marker_count/marker_position features."""
    hash_positions = [i for i, t in enumerate(seq) if t == marker]
    if len(hash_positions) != 1:
        return False
    m = hash_positions[0]
    n_before, n_after = m, len(seq) - m - 1
    return n_before == n_after


def provable_incapacity_counterexample():
    """Same marker count/position (marker_shortcut_features is identical),
    opposite validity -- the swap breaks deinterleave-content while leaving
    marker structure untouched. w=0110 -> deinterleave(w) = evens(01)+odds(10)
    = 0,1,1,0 -> "01" + "10" = "0110"; corrupt by swapping the last two
    post-marker positions."""
    clean = ["0", "1", "1", "0", "#", "0", "1", "1", "0"]  # w=0110, deinterleave(w)=0110 -- valid
    corrupted = ["0", "1", "1", "0", "#", "0", "1", "0", "1"]  # last two swapped -- invalid
    fc = marker_shortcut_features([clean])[0]
    fk = marker_shortcut_features([corrupted])[0]
    return {
        "clean": clean, "corrupted": corrupted,
        "clean_features": fc.tolist(), "corrupted_features": fk.tolist(),
        "features_identical": bool(np.array_equal(fc, fk)),
        "note": "IDENTICAL marker-shortcut feature vectors (marker count/relative-offset), "
               "opposite validity -- the information distinguishing them (whether the post-"
               "marker half is actually the correct deinterleave, not just correctly bracketed) "
               "is structurally absent from a marker-position-only representation.",
    }


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
        "baseline": name, "n_total": len(labels), "overall_accuracy": overall_acc,
        "positive_accuracy": pos_acc, "n_positive": n_pos,
        "uniform_random_half_accuracy": uniform_acc, "n_uniform_random": n_uniform,
        "hard_negative_half_accuracy": hard_acc, "n_hard_negative": n_hard,
        "long_sequence_accuracy": long_acc, "n_long_sequences": n_long,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    seqs, labels, kinds = load_fixed_test()
    print(f"corrected test set: n={len(seqs)}", flush=True)
    pos_idx, uniform_idx, hard_idx = split_indices(labels, kinds)
    print(f"  n_positive={len(pos_idx)} n_uniform_random={len(uniform_idx)} n_hard_negative={len(hard_idx)}", flush=True)

    counterexample = provable_incapacity_counterexample()
    print("\n=== provable incapacity counterexample ===")
    print(json.dumps(counterexample, indent=2, default=str))
    assert counterexample["features_identical"]

    # ------------------------------------------------------------------
    # theoretical shortcut-only ceiling
    # ------------------------------------------------------------------
    neg_idx_all = uniform_idx + hard_idx
    ambiguous_flags = {i: is_structurally_ambiguous(seqs[i]) for i in neg_idx_all}
    n_ambiguous_from_uniform = sum(1 for i in uniform_idx if ambiguous_flags[i])
    n_ambiguous_from_hard = sum(1 for i in hard_idx if ambiguous_flags[i])
    n_obvious_from_uniform = len(uniform_idx) - n_ambiguous_from_uniform
    n_obvious_from_hard = len(hard_idx) - n_ambiguous_from_hard
    n_total = len(labels)
    n_structurally_ambiguous = n_ambiguous_from_uniform + n_ambiguous_from_hard
    n_structurally_obvious = n_obvious_from_uniform + n_obvious_from_hard
    theoretical_ceiling = (len(pos_idx) + n_structurally_obvious + 0.5 * n_structurally_ambiguous) / n_total

    ceiling_report = {
        "n_total": n_total, "n_positive": len(pos_idx),
        "n_structurally_obvious_negatives": n_structurally_obvious,
        "structurally_obvious_breakdown": {"from_hard_negative_half": n_obvious_from_hard,
                                          "from_uniform_random_half": n_obvious_from_uniform},
        "n_structurally_ambiguous_negatives": n_structurally_ambiguous,
        "structurally_ambiguous_breakdown": {
            "from_hard_negative_half": n_ambiguous_from_hard,
            "from_uniform_random_half_by_chance": n_ambiguous_from_uniform,
        },
        "formula": "(n_positive + n_structurally_obvious + 0.5 * n_structurally_ambiguous) / n_total "
                  "-- perfect classification wherever marker_count/marker_position alone can "
                  "distinguish the example from a positive, chance (0.5) on everything single-"
                  "marker-and-balanced (indistinguishable from a positive via marker features "
                  "alone).",
        "theoretical_shortcut_only_ceiling": theoretical_ceiling,
        "note": (
            f"{n_ambiguous_from_hard}/{len(hard_idx)} of the hard-negative half is structurally "
            f"ambiguous by construction (should be ALL of it, i.e. {len(hard_idx)}); "
            f"{n_ambiguous_from_uniform}/{len(uniform_idx)} of the uniform-random half is ALSO "
            f"structurally ambiguous purely by chance."
        ),
    }
    print("\n=== theoretical shortcut-only ceiling ===")
    print(json.dumps(ceiling_report, indent=2, default=str))
    assert n_ambiguous_from_hard == len(hard_idx), (
        f"expected ALL hard-negative-half examples to be structurally ambiguous by construction, "
        f"got {n_ambiguous_from_hard}/{len(hard_idx)}"
    )

    # ------------------------------------------------------------------
    # (1) marker_shortcut baseline -- TRAINED HERE (standard training data)
    # ------------------------------------------------------------------
    print("\n=== training marker_shortcut baseline on standard odds-first training data ===", flush=True)
    train_seqs, train_labels = load_split(TASK, "train")
    val_seqs, val_labels = load_split(TASK, "validation-short")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms_model, ms_C, ms_val_bce, ms_warnings = train_marker_shortcut(train_seqs, train_labels, val_seqs, val_labels)
    ms_ckpt_path = CKPT_DIR / "phase1_odds_first_marker_shortcut.joblib"
    joblib.dump({"model": ms_model, "selected_C": ms_C,
                "source_data": {"train": f"languages/{TASK}/main.tok"}}, ms_ckpt_path)
    X_ms = marker_shortcut_features(seqs)
    probs_ms = ms_model.predict_proba(X_ms)[:, 1]
    ms_result = evaluate_baseline("marker_shortcut", probs_ms, labels, pos_idx, uniform_idx, hard_idx, seqs)
    ms_result["source_checkpoint"] = str(ms_ckpt_path)
    ms_result["trained_on"] = "standard odds-first training set"
    ms_result["selected_C"] = ms_C
    print(json.dumps(ms_result, indent=2, default=str))

    # ------------------------------------------------------------------
    # (2) ngram-logreg baseline -- TRAINED HERE
    # ------------------------------------------------------------------
    print("\n=== training ngram-logreg on standard odds-first training data ===", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tA = train_candidate_A(TASK, train_seqs, train_labels, val_seqs, val_labels)
    print(f"  selected_C={tA['selected_C']} n_features={tA['n_features']} "
          f"convergence_issues={tA['convergence_warnings'] or ['none']}", flush=True)
    ng_ckpt_path = CKPT_DIR / "phase1_odds_first_ngram_logreg.joblib"
    joblib.dump({"vectorizer": tA["vectorizer"], "model": tA["model"],
                "source_data": {"train": f"languages/{TASK}/main.tok",
                                "validation": f"languages/{TASK}/datasets/validation-short/main.tok"}},
               ng_ckpt_path)
    probs_ng = eval_candidate_A(tA["vectorizer"], tA["model"], seqs, labels)
    ng_result = evaluate_baseline("ngram-logreg", probs_ng, labels, pos_idx, uniform_idx, hard_idx, seqs)
    ng_result["source_checkpoint"] = str(ng_ckpt_path)
    ng_result["trained_on"] = "standard odds-first training set"
    ng_result["selected_C"] = tA["selected_C"]
    print(json.dumps(ng_result, indent=2, default=str))

    # ------------------------------------------------------------------
    # (3) bagcount-ffn baseline -- TRAINED HERE, swept over h
    # ------------------------------------------------------------------
    vocab = load_vocab(TASK)
    bagcount_results = {}
    for h in H_GRID:
        print(f"\n=== training bagcount-ffn h={h} on standard odds-first training data ===", flush=True)
        tB = train_candidate_B(TASK, h, vocab, train_seqs, train_labels, val_seqs, val_labels)
        ckpt_path = CKPT_DIR / f"phase1_odds_first_bagcount_ffn_h{h}.pt"
        torch.save({"state_dict": tB["model"].state_dict(), "vocab": vocab, "hidden_units": h,
                    "source_data": {"train": f"languages/{TASK}/main.tok",
                                    "validation": f"languages/{TASK}/datasets/validation-short/main.tok"}},
                   ckpt_path)
        probs_bc = eval_candidate_B(tB["model"], seqs, vocab)
        bc_result = evaluate_baseline(f"bagcount-ffn-h{h}", probs_bc, labels, pos_idx, uniform_idx, hard_idx, seqs)
        bc_result["source_checkpoint"] = str(ckpt_path)
        bc_result["trained_on"] = "standard odds-first training set"
        bc_result["num_epochs_run"] = tB["num_epochs_run"]
        bc_result["stopped_early"] = tB["stopped_early"]
        bc_result["convergence_issues"] = tB["issues"] or ["none"]
        bagcount_results[str(h)] = bc_result
        print(f"  h={h} epochs={tB['num_epochs_run']} stopped_early={tB['stopped_early']} "
              f"issues={tB['issues'] or 'none'}", flush=True)
        print(json.dumps(bc_result, indent=2, default=str))

    # ------------------------------------------------------------------
    # summary vs ceiling
    # ------------------------------------------------------------------
    all_baselines = {"marker_shortcut": ms_result, "ngram-logreg": ng_result,
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
        "task": "odds-first",
        "experiment": "corrected-dataset-fix (OF2: shortcut-only baseline ceiling on corrected test data)",
        "description": (
            "Evaluates a NEWLY-TRAINED marker_shortcut baseline (marker count/relative-position, "
            "reused unchanged from marked-copy/marked-reversal's feature set -- provably incapable "
            "of deinterleave-content verification) plus ngram-logreg and bagcount-ffn baselines "
            "(also newly trained here), all trained on STANDARD odds-first training data, evaluated "
            "on the CORRECTED test set (OF1), split into positive/uniform-random/hard-negative "
            "subsets."
        ),
        "provable_incapacity_counterexample": counterexample,
        "corrected_test_set": str(FIXED_TEST_DIR),
        "test_composition": {"n_total": n_total, "n_positive": len(pos_idx),
                             "n_uniform_random": len(uniform_idx), "n_hard_negative": len(hard_idx)},
        "theoretical_shortcut_only_ceiling": ceiling_report,
        "baselines": {"marker_shortcut": ms_result, "ngram_logreg": ng_result, "bagcount_ffn": bagcount_results},
        "summary_vs_ceiling": summary_rows,
    }
    out_path = RESULTS / "fix_baseline_ceiling_odds_first.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
