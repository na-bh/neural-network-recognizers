"""Corrected-dataset experiment for Dyck-2-3, DF2: shortcut-only baseline
ceiling on the corrected test set.

Shortcut-only baseline: a 9-feature vector per sequence -- total_length,
n_open, n_close, n_open_0, n_close_0, n_open_1, n_close_1, first_token_is_
opener, last_token_is_closer -- a pure BAG-OF-COUNTS-AND-POSITIONS
representation with NO ORDER information beyond first/last token and no
notion of a stack. PROVABLY incapable of stack simulation: the task audit's
own load-bearing counterexample (clean=[(0,(1,)1,)0], corrupted=[(0,(1,)0,
)1]) has IDENTICAL values for every one of these 9 features (same counts,
same first/last tokens) but opposite validity -- no classifier operating on
this feature vector alone can distinguish them, regardless of capacity,
because the information needed (WHICH type is on top of the stack at the
moment each closer appears) is structurally absent from the feature vector
itself, not merely hard to learn.

Also trains ngram-logreg and bagcount-ffn baselines (did not previously
exist for dyck-2-3), on the STANDARD dyck-2-3 training set -- matching the
established fix-experiment convention (baselines trained on standard data,
evaluated on the corrected test set, no retraining on corrected data).

Theoretical shortcut-only ceiling: by DF1's construction, ALL hard
negatives pass every one of the four listed shortcut features (0%
detectable, verified in DF1's audit) -- so the ceiling is simply (n_positive
+ n_structurally_obvious_negatives + 0.5 * n_structurally_ambiguous) /
n_total, where "structurally ambiguous" = passes all four shortcut features
(the ENTIRE hard-negative half by construction, plus any uniform-random
negatives that coincidentally also pass all four -- counted directly, not
assumed zero).

PYTHONPATH=src:analysis python analysis/fix_baseline_ceiling_dyck_2_3.py
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, "analysis")
from phase3_dyck_task_audit import SHORTCUT_FEATURES, K, M, is_valid_dyck
from phase1_baselines_train import (
    train_candidate_A, eval_candidate_A, train_candidate_B, eval_candidate_B,
    BagCountFFN, count_features, load_vocab, load_split,
)

RESULTS = Path("analysis_outputs/final_results")
CKPT_DIR = RESULTS / "phase1_checkpoints"
TASK = "dyck-2-3"
FIXED_TEST_DIR = Path("languages/dyck-2-3-fixed/datasets/test")
LONG_SEQ_THRESH = 400
H_GRID = [8, 32, 128, 512]
OPENERS = {f"({k}" for k in range(K)}
CLOSERS = {f"){k}" for k in range(K)}


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


def shortcut_features(seqs):
    X = []
    for seq in seqs:
        n = len(seq)
        n_open = sum(1 for t in seq if t in OPENERS)
        n_close = sum(1 for t in seq if t in CLOSERS)
        n_open_k = [sum(1 for t in seq if t == f"({k}") for k in range(K)]
        n_close_k = [sum(1 for t in seq if t == f"){k}") for k in range(K)]
        first_opener = float(bool(seq) and seq[0] in OPENERS)
        last_closer = float(bool(seq) and seq[-1] in CLOSERS)
        X.append([n, n_open, n_close, *n_open_k, *n_close_k, first_opener, last_closer])
    return np.asarray(X, dtype=np.float64)


FEATURE_NAMES = ["total_length", "n_open", "n_close", "n_open_0", "n_open_1",
                 "n_close_0", "n_close_1", "first_token_is_opener", "last_token_is_closer"]


def provable_incapacity_counterexample():
    clean = ["(0", "(1", ")1", ")0"]
    corrupted = ["(0", "(1", ")0", ")1"]
    fc, fk = shortcut_features([clean])[0], shortcut_features([corrupted])[0]
    return {
        "clean": clean, "corrupted": corrupted,
        "clean_valid": is_valid_dyck(clean, M), "corrupted_valid": is_valid_dyck(corrupted, M),
        "clean_features": dict(zip(FEATURE_NAMES, fc.tolist())),
        "corrupted_features": dict(zip(FEATURE_NAMES, fk.tolist())),
        "features_identical": bool(np.array_equal(fc, fk)),
        "note": ("IDENTICAL 9-feature vectors, opposite validity -- the information distinguishing "
                "them (WHICH type is on top of the stack when each closer appears) is structurally "
                "absent from any bag-of-counts-and-position representation, not merely hard to learn."),
    }


def train_shortcut_baseline():
    train_seqs, train_labels = load_split(TASK, "train")
    X = shortcut_features(train_seqs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=2000).fit(X, train_labels)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "phase1_dyck_2_3_shortcut_baseline.joblib"
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES,
                "source_data": {"train": f"languages/{TASK}/main.tok"}}, ckpt_path)
    return model, ckpt_path


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
    seqs, labels, kinds = load_fixed_test()
    print(f"corrected test set: n={len(seqs)}", flush=True)
    pos_idx, uniform_idx, hard_idx = split_indices(labels, kinds)
    print(f"  n_positive={len(pos_idx)} n_uniform_random={len(uniform_idx)} n_hard_negative={len(hard_idx)}", flush=True)

    # ------------------------------------------------------------------
    # provable incapacity for stack simulation
    # ------------------------------------------------------------------
    counterexample = provable_incapacity_counterexample()
    print("\n=== provable incapacity counterexample ===")
    print(json.dumps(counterexample, indent=2, default=str))
    assert counterexample["features_identical"] and counterexample["clean_valid"] and not counterexample["corrupted_valid"]

    # ------------------------------------------------------------------
    # theoretical shortcut-only ceiling
    # ------------------------------------------------------------------
    def passes_all_shortcuts(seq):
        return all(fn(seq) for fn in SHORTCUT_FEATURES.values())

    neg_idx_all = uniform_idx + hard_idx
    ambiguous_flags = {i: passes_all_shortcuts(seqs[i]) for i in neg_idx_all}
    n_ambiguous_from_uniform = sum(1 for i in uniform_idx if ambiguous_flags[i])
    n_ambiguous_from_hard = sum(1 for i in hard_idx if ambiguous_flags[i])
    n_obvious_from_uniform = len(uniform_idx) - n_ambiguous_from_uniform
    n_obvious_from_hard = len(hard_idx) - n_ambiguous_from_hard
    n_total = len(labels)
    n_structurally_obvious = n_obvious_from_uniform + n_obvious_from_hard
    n_structurally_ambiguous = n_ambiguous_from_uniform + n_ambiguous_from_hard
    theoretical_ceiling = (len(pos_idx) + n_structurally_obvious + 0.5 * n_structurally_ambiguous) / n_total

    ceiling_report = {
        "n_total": n_total, "n_positive": len(pos_idx),
        "n_structurally_obvious_negatives": n_structurally_obvious,
        "structurally_obvious_breakdown": {"from_hard_negative_half": n_obvious_from_hard,
                                          "from_uniform_random_half": n_obvious_from_uniform},
        "n_structurally_ambiguous_negatives": n_structurally_ambiguous,
        "structurally_ambiguous_breakdown": {"from_hard_negative_half": n_ambiguous_from_hard,
                                            "from_uniform_random_half_by_chance": n_ambiguous_from_uniform},
        "formula": "(n_positive + n_structurally_obvious + 0.5 * n_structurally_ambiguous) / n_total "
                  "-- perfect on positives and every negative a shortcut-only detector (the 9-feature "
                  "bag-of-counts) can catch; chance (0.5) on every negative that passes all four "
                  "listed shortcut features (by DF1's construction, this is the ENTIRE hard-negative "
                  "half, plus any uniform-random negatives that coincidentally also pass all four).",
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
        f"expected ALL hard-negative-half examples to pass all four shortcuts by construction, "
        f"got {n_ambiguous_from_hard}/{len(hard_idx)}"
    )

    # ------------------------------------------------------------------
    # (1) shortcut baseline -- TRAINED HERE (on STANDARD training data)
    # ------------------------------------------------------------------
    print("\n=== training shortcut baseline on standard dyck-2-3 training data ===", flush=True)
    sc_model, sc_ckpt = train_shortcut_baseline()
    X_sc = shortcut_features(seqs)
    probs_sc = sc_model.predict_proba(X_sc)[:, 1]
    sc_result = evaluate_baseline("shortcut", probs_sc, labels, pos_idx, uniform_idx, hard_idx, seqs)
    sc_result["source_checkpoint"] = str(sc_ckpt)
    sc_result["trained_on"] = "standard dyck-2-3 training set"
    print(json.dumps(sc_result, indent=2, default=str))

    # ------------------------------------------------------------------
    # (2) ngram-logreg baseline -- TRAINED HERE
    # ------------------------------------------------------------------
    print("\n=== training ngram-logreg on standard dyck-2-3 training data ===", flush=True)
    train_seqs, train_labels = load_split(TASK, "train")
    val_seqs, val_labels = load_split(TASK, "validation-short")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tA = train_candidate_A(TASK, train_seqs, train_labels, val_seqs, val_labels)
    print(f"  selected_C={tA['selected_C']} n_features={tA['n_features']} "
          f"convergence_issues={tA['convergence_warnings'] or ['none']}", flush=True)
    ng_ckpt_path = CKPT_DIR / "phase1_dyck_2_3_ngram_logreg.joblib"
    joblib.dump({"vectorizer": tA["vectorizer"], "model": tA["model"],
                "source_data": {"train": f"languages/{TASK}/main.tok",
                                "validation": f"languages/{TASK}/datasets/validation-short/main.tok"}},
               ng_ckpt_path)
    probs_ng = eval_candidate_A(tA["vectorizer"], tA["model"], seqs, labels)
    ng_result = evaluate_baseline("ngram-logreg", probs_ng, labels, pos_idx, uniform_idx, hard_idx, seqs)
    ng_result["source_checkpoint"] = str(ng_ckpt_path)
    ng_result["trained_on"] = "standard dyck-2-3 training set"
    ng_result["selected_C"] = tA["selected_C"]
    print(json.dumps(ng_result, indent=2, default=str))

    # ------------------------------------------------------------------
    # (3) bagcount-ffn baseline -- TRAINED HERE, swept over h
    # ------------------------------------------------------------------
    vocab = load_vocab(TASK)
    bagcount_results = {}
    for h in H_GRID:
        print(f"\n=== training bagcount-ffn h={h} on standard dyck-2-3 training data ===", flush=True)
        tB = train_candidate_B(TASK, h, vocab, train_seqs, train_labels, val_seqs, val_labels)
        ckpt_path = CKPT_DIR / f"phase1_dyck_2_3_bagcount_ffn_h{h}.pt"
        torch.save({"state_dict": tB["model"].state_dict(), "vocab": vocab, "hidden_units": h,
                    "source_data": {"train": f"languages/{TASK}/main.tok",
                                    "validation": f"languages/{TASK}/datasets/validation-short/main.tok"}},
                   ckpt_path)
        probs_bc = eval_candidate_B(tB["model"], seqs, vocab)
        bc_result = evaluate_baseline(f"bagcount-ffn-h{h}", probs_bc, labels, pos_idx, uniform_idx, hard_idx, seqs)
        bc_result["source_checkpoint"] = str(ckpt_path)
        bc_result["trained_on"] = "standard dyck-2-3 training set"
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
        "experiment": "corrected-dataset-fix (DF2: shortcut-only baseline ceiling on corrected test data)",
        "description": (
            "Evaluates a NEWLY-TRAINED shortcut-only baseline (9-feature bag-of-counts, provably "
            "incapable of stack simulation) plus ngram-logreg and bagcount-ffn baselines (also newly "
            "trained here -- none previously existed for dyck-2-3), all trained on STANDARD dyck-2-3 "
            "training data, evaluated on the CORRECTED test set (DF1), split into positive/uniform-"
            "random/hard-negative subsets."
        ),
        "provable_incapacity_counterexample": counterexample,
        "corrected_test_set": str(FIXED_TEST_DIR),
        "test_composition": {"n_total": n_total, "n_positive": len(pos_idx),
                             "n_uniform_random": len(uniform_idx), "n_hard_negative": len(hard_idx)},
        "theoretical_shortcut_only_ceiling": ceiling_report,
        "baselines": {"shortcut": sc_result, "ngram_logreg": ng_result, "bagcount_ffn": bagcount_results},
        "summary_vs_ceiling": summary_rows,
    }
    out_path = RESULTS / "fix_baseline_ceiling_dyck_2_3.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
