"""Corrected-dataset experiment, F2: shortcut-only baseline ceiling on the
corrected test set.

Loads the THREE shortcut-only baselines already trained on marked-reversal's
STANDARD training data (analysis_outputs/final_results/phase1_checkpoints/
phase1_marked_reversal_{marker_shortcut,ngram_logreg,bagcount_ffn_h*}.*, from
analysis/phase1_b3_compare.py and analysis/phase1_baselines_train.py -- no
retraining here) and evaluates them on the CORRECTED test set (languages/
marked-reversal-fixed/datasets/test, generated in F1), reporting accuracy
overall and separately on the uniform-random and hard-negative halves (split
via F1's negative-kind.txt side-channel).

Also computes the THEORETICAL shortcut-only ceiling directly from the
corrected test set's actual composition (not assumed 50/50), generalizing
the pilot's established grammar_only_ceiling formula (accept all positives +
perfect on everything marker-features can distinguish + chance on everything
they can't): a negative is "structurally ambiguous" (indistinguishable from
a positive using ONLY marker_count/marker_position -- i.e. exactly one
marker AND balanced halves) or "structurally obvious" (everything else --
zero/multiple markers, or a single off-center marker). ALL of F1's
hard-negative-half examples are structurally ambiguous by construction; a
small number of uniform-random-half examples may ALSO happen to be
structurally ambiguous by chance -- counted directly, not assumed to be
zero.

PYTHONPATH=src:analysis python analysis/fix_baseline_ceiling_marked_reversal_f2.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase1_baselines_train import BagCountFFN, count_features
from phase1_b3_compare import marker_shortcut_features

RESULTS = Path("analysis_outputs/final_results")
CKPT_DIR = RESULTS / "phase1_checkpoints"
FIXED_TEST_DIR = Path("languages/marked-reversal-fixed/datasets/test")
LONG_SEQ_THRESH = 400


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


def is_structurally_ambiguous(seq, marker="#"):
    """Exactly one marker AND balanced halves -- indistinguishable from a
    genuine positive using ONLY marker_count/marker_position features."""
    hash_positions = [i for i, t in enumerate(seq) if t == marker]
    if len(hash_positions) != 1:
        return False
    m = hash_positions[0]
    n_before, n_after = m, len(seq) - m - 1
    return n_before == n_after


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
        "baseline": name,
        "n_total": len(labels),
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
        "n_structurally_ambiguous_negatives": n_structurally_ambiguous,
        "structurally_ambiguous_breakdown": {
            "from_hard_negative_half": n_ambiguous_from_hard,
            "from_uniform_random_half_by_chance": n_ambiguous_from_uniform,
        },
        "formula": "(n_positive + n_structurally_obvious + 0.5 * n_structurally_ambiguous) / n_total "
                  "-- perfect classification wherever marker_count/marker_position alone can "
                  "distinguish the example from a positive (all positives, plus every negative "
                  "that is NOT single-marker-and-balanced), chance (0.5) on everything that is "
                  "single-marker-and-balanced (indistinguishable from a positive via marker "
                  "features alone) -- generalizes the pilot's established grammar_only_ceiling "
                  "formula to the corrected test set's actual composition.",
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
    # (1) marker_shortcut baseline
    # ------------------------------------------------------------------
    ms_ckpt = joblib.load(CKPT_DIR / "phase1_marked_reversal_marker_shortcut.joblib")
    X_ms = marker_shortcut_features(seqs)
    probs_ms = ms_ckpt["model"].predict_proba(X_ms)[:, 1]
    ms_result = evaluate_baseline("marker_shortcut", probs_ms, labels, pos_idx, uniform_idx, hard_idx, seqs)
    ms_result["source_checkpoint"] = str(CKPT_DIR / "phase1_marked_reversal_marker_shortcut.joblib")
    ms_result["trained_on"] = "standard marked-reversal training set (Butoi et al. mixed strategy, K-edit perturbations)"
    print("\n=== marker_shortcut ===")
    print(json.dumps(ms_result, indent=2, default=str))

    # ------------------------------------------------------------------
    # (2) ngram-logreg baseline
    # ------------------------------------------------------------------
    ng_ckpt = joblib.load(CKPT_DIR / "phase1_marked_reversal_ngram_logreg.joblib")
    char_strings = ["".join(s) for s in seqs]
    X_ng = ng_ckpt["vectorizer"].transform(char_strings)
    probs_ng = ng_ckpt["model"].predict_proba(X_ng)[:, 1]
    ng_result = evaluate_baseline("ngram-logreg", probs_ng, labels, pos_idx, uniform_idx, hard_idx, seqs)
    ng_result["source_checkpoint"] = str(CKPT_DIR / "phase1_marked_reversal_ngram_logreg.joblib")
    ng_result["trained_on"] = "standard marked-reversal training set (Butoi et al. mixed strategy, K-edit perturbations)"
    print("\n=== ngram-logreg ===")
    print(json.dumps(ng_result, indent=2, default=str))

    # ------------------------------------------------------------------
    # (3) bagcount-ffn baseline, all trained h values
    # ------------------------------------------------------------------
    bagcount_results = {}
    for h in [8, 32, 128, 512]:
        ckpt_path = CKPT_DIR / f"phase1_marked_reversal_bagcount_ffn_h{h}.pt"
        bc_ckpt = torch.load(ckpt_path, weights_only=False)
        model = BagCountFFN(len(bc_ckpt["vocab"]), bc_ckpt["hidden_units"])
        model.load_state_dict(bc_ckpt["state_dict"])
        model.eval()
        X_bc = torch.tensor(count_features(seqs, bc_ckpt["vocab"]))
        with torch.inference_mode():
            probs_bc = torch.sigmoid(model(X_bc)).numpy()
        bc_result = evaluate_baseline(f"bagcount-ffn-h{h}", probs_bc, labels, pos_idx, uniform_idx, hard_idx, seqs)
        bc_result["source_checkpoint"] = str(ckpt_path)
        bc_result["trained_on"] = "standard marked-reversal training set (Butoi et al. mixed strategy, K-edit perturbations)"
        bagcount_results[str(h)] = bc_result
        print(f"\n=== bagcount-ffn h={h} ===")
        print(json.dumps(bc_result, indent=2, default=str))

    out = {
        "task": "marked-reversal",
        "experiment": "corrected-dataset-fix (F2: shortcut-only baseline ceiling on corrected test data)",
        "description": (
            "Evaluates the three shortcut-only baselines already trained on marked-reversal's "
            "STANDARD training data (no retraining here) on the CORRECTED test set (F1), split "
            "into uniform-random and hard-negative halves via F1's negative-kind.txt side-"
            "channel. Also reports the theoretical shortcut-only ceiling computed directly from "
            "the corrected test set's actual structural composition."
        ),
        "corrected_test_set": str(FIXED_TEST_DIR),
        "test_composition": {
            "n_total": n_total, "n_positive": len(pos_idx),
            "n_uniform_random": len(uniform_idx), "n_hard_negative": len(hard_idx),
        },
        "theoretical_shortcut_only_ceiling": ceiling_report,
        "baselines": {
            "marker_shortcut": ms_result,
            "ngram_logreg": ng_result,
            "bagcount_ffn": bagcount_results,
        },
    }
    out_path = RESULTS / "fix_baseline_ceiling_marked_reversal.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
