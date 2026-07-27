"""B3: compare grammar-only baselines (B2) against the trained architectures'
accuracy on the standard test set, restructured per review of B2's results:

  - cycle-navigation: bagcount-ffn h=4/h=8 (from phase1_baselines_trained.json)
    is the PRIMARY comparison -- symbol counts are a sufficient statistic for
    this task (the label is a function of count('>')-count('<') mod 5, and
    order among moves is irrelevant by commutativity), so an h=4 model with
    zero positional information is not handicapped by missing information,
    only by capacity, and closely reproduces the trained architectures' long-
    sequence accuracy. ngram-logreg is reported as a secondary result with an
    explicit length-generalization caveat (see B2 notes: C=100 was selected on
    a length<=40 validation set and overfits length-specific n-gram statistics
    that do not transfer to the length-500 test tail).

  - marked-reversal: neither B2 baseline reproduced the trained architectures'
    accuracy (bagcount-ffn cannot see position at all; ngram-logreg was
    deliberately kept "pure" per instruction and lacks a length feature, so it
    under-detects marker-POSITION violations). This script adds a new,
    targeted "marker-shortcut" baseline (Candidate C) whose features are
    restricted, BY CONSTRUCTION, to marker count and marker position relative
    to sequence length -- it never sees the 0/1 content tokens at all, so it
    is trivially, structurally blind to content-mismatch hard negatives (a
    stronger guarantee than Candidates A/B's linear-separability / data-
    processing-inequality arguments: here the content simply is not part of
    the input representation).

This script also computes FULL test-set accuracy (all 5010 examples, all
lengths 0-500) for the trained architectures' selected seeds, since Part 6B
(cyclenav_shortcut_check.json, flare_a2_marked_reversal.json) only recorded
long-sequence accuracy (length>=400, used for seed selection) and hard/
scrambled-negative accuracy (capped at 300 examples per subset for runtime).
Loads the exact selected-seed checkpoints Part 6B selected, via the same
load_model_generic pattern as analysis/cyclenav_s2_shortcut_check.py.

PYTHONPATH=src:analysis python analysis/phase1_b3_compare.py
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings

from flare_a1_task_audit import TASKS as A1_TASKS
from recognizers.neural_networks.data import (
    add_data_arguments, load_vocabulary_data, load_prepared_data_from_directory,
)
from recognizers.neural_networks.model_interface import RecognitionModelInterface

RESULTS = Path("analysis_outputs/final_results")
CKPT_DIR = RESULTS / "phase1_checkpoints"
LONG_SEQ_THRESH = 400
HARD_THRESHOLD = 0.8
VARIANT = "rec+ns/validation-short"
ARCHS = ["rnn", "lstm", "transformer", "mamba"]

CYCLENAV_MOVES = {"<", "=", ">"}
CYCLENAV_DIGITS = {"0", "1", "2", "3", "4"}


def is_grammatical_cyclenav(seq):
    if len(seq) == 0:
        return False
    body, last = seq[:-1], seq[-1]
    return all(t in CYCLENAV_MOVES for t in body) and last in CYCLENAV_DIGITS


def hard_scrambled_split(task, seqs, labels):
    if task == "cycle-navigation":
        hard_idx = [i for i, (s, l) in enumerate(zip(seqs, labels)) if l == 0 and is_grammatical_cyclenav(s)]
        scrambled_idx = [i for i, (s, l) in enumerate(zip(seqs, labels)) if l == 0 and not is_grammatical_cyclenav(s)]
    else:
        viol_fn = A1_TASKS[task]["viol_fn"]
        hard_idx, scrambled_idx = [], []
        for i, (s, l) in enumerate(zip(seqs, labels)):
            if l != 0:
                continue
            f = viol_fn(s)
            (hard_idx if f >= HARD_THRESHOLD else scrambled_idx).append(i)
    return hard_idx, scrambled_idx


# ----------------------------------------------------------------------
# Trained-architecture full test-set accuracy (loads Part 6B's selected seeds)
# ----------------------------------------------------------------------

def checkpoint_root(task, arch):
    for base in ["models", "data/models"]:
        p = Path(base) / task / arch / VARIANT
        if p.exists():
            return Path(base)
    raise FileNotFoundError(f"{task}/{arch}")


def load_model_generic(task, arch, seed):
    base = checkpoint_root(task, arch)
    model_dir = base / task / arch / VARIANT / str(seed)
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    iface = RecognitionModelInterface()
    iface.add_arguments(parser)
    iface.add_forward_arguments(parser)
    tmp = tempfile.mkdtemp(prefix="phase1_b3_")
    shutil.rmtree(tmp, ignore_errors=True)
    args = parser.parse_args([
        "--output", tmp, "--training-data", f"languages/{task}",
        "--architecture", arch, "--load-model", str(model_dir), "--load-parameters", "main",
    ])
    vocab = load_vocabulary_data(args, parser)
    saver = iface.construct_saver(args, vocab)
    saver.model.eval()
    return iface, saver


def predict_all(iface, saver, test_examples):
    device = next(saver.model.parameters()).device
    preds = []
    failed_indices = []
    for i, (seq, (label, next_symbols)) in enumerate(test_examples):
        try:
            mi, _ = iface.prepare_batch([(seq, (label, next_symbols))], device)
            with torch.no_grad():
                rec, _, _ = iface.get_logits(saver.model, mi)
            preds.append(bool(rec.item() > 0))
        except Exception as e:
            preds.append(None)
            failed_indices.append((i, len(seq), type(e).__name__, str(e)[:200]))
    return preds, failed_indices


def evaluate_trained_model(task, arch, seed, test_examples, hard_idx, scrambled_idx, n_pos):
    iface, saver = load_model_generic(task, arch, seed)
    preds, failed = predict_all(iface, saver, test_examples)
    labels = [bool(label) for _, (label, _) in test_examples]
    seqs = [seq for seq, _ in test_examples]
    failed_set = {i for i, _, _, _ in failed}
    correct = [(p == l) if p is not None else None for p, l in zip(preds, labels)]

    def acc_on(idx):
        idx = [i for i in idx if i not in failed_set]
        if not idx:
            return float("nan"), 0
        return float(np.mean([correct[i] for i in idx])), len(idx)

    valid_idx = [i for i in range(len(labels)) if i not in failed_set]
    overall_acc, n_total_valid = acc_on(valid_idx)
    hard_acc, n_hard = acc_on(hard_idx)
    scrambled_acc, n_scrambled = acc_on(scrambled_idx)
    long_idx = [i for i, s in enumerate(seqs) if len(s) >= LONG_SEQ_THRESH]
    long_acc, n_long = acc_on(long_idx)
    return {
        "seed": seed,
        "n_total": n_total_valid,
        "n_excluded_due_to_inference_failure": len(failed),
        "inference_failures": [
            {"test_index": i, "sequence_length": L, "exception_type": et, "message": msg}
            for i, L, et, msg in failed
        ],
        "n_positives": n_pos,
        "n_hard_negatives": n_hard,
        "n_scrambled_negatives": n_scrambled,
        "n_long_sequences": n_long,
        "test_accuracy": overall_acc,
        "hard_negative_accuracy": hard_acc,
        "scrambled_negative_accuracy": scrambled_acc,
        "long_sequence_accuracy": long_acc,
    }


# ----------------------------------------------------------------------
# Candidate C: marker-shortcut baseline for marked-reversal
# Features are restricted to marker count/position; content tokens ('0'/'1'
# identities) are never included in the feature vector, so this baseline is
# structurally, not just empirically, incapable of detecting content-mismatch
# hard negatives.
# ----------------------------------------------------------------------

def marker_shortcut_features(seqs):
    # Features are deliberately scale-invariant (offset normalized by length,
    # not raw token distance): "# is at approximately the middle" is a
    # relative-position notion, and a raw absolute-offset/length feature pair
    # was found to produce a length-generalization artifact (a decision
    # boundary from near-cancelling large coefficients on raw length/idx_hash
    # that only holds inside the length<=40 training range -- verified by
    # probing the fitted model on synthetic length-401 vs. length-31 positives
    # with identical (perfectly centered) marker placement, which received
    # accept-probabilities of 0.06 vs. 0.93 respectively under the raw-offset
    # version of this feature set).
    X = []
    for seq in seqs:
        length = len(seq)
        hash_positions = [i for i, t in enumerate(seq) if t == "#"]
        n_hash = len(hash_positions)
        expected_idx = (length - 1) / 2.0 if length > 0 else 0.0
        if n_hash >= 1:
            idx_hash = hash_positions[0]
            rel_offset = (idx_hash - expected_idx) / max(length, 1)
        else:
            rel_offset = 1.0  # sentinel: "no marker present", outside the valid [-1,1] range
        X.append([n_hash, rel_offset, abs(rel_offset)])
    return np.asarray(X, dtype=np.float64)


def load_split(task, split):
    LANG = Path("languages")
    d = LANG / task if split == "train" else LANG / task / "datasets" / split
    toks = (d / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (d / "labels.txt").read_text().splitlines()]
    seqs = [line.split() for line in toks]
    return seqs, labels


def train_marker_shortcut(train_seqs, train_labels, val_seqs, val_labels):
    X_train = marker_shortcut_features(train_seqs)
    X_val = marker_shortcut_features(val_seqs)
    C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    convergence_warnings = []
    best_C, best_val_bce, best_model = None, float("inf"), None
    val_bce_per_C = {}
    for C in C_grid:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=5000)
            model.fit(X_train, train_labels)
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    convergence_warnings.append(f"C={C}: {str(w.message)}")
        val_probs = model.predict_proba(X_val)[:, 1]
        val_bce = log_loss(val_labels, val_probs, labels=[0, 1])
        val_bce_per_C[C] = val_bce
        if val_bce < best_val_bce:
            best_C, best_val_bce, best_model = C, val_bce, model
    return best_model, best_C, val_bce_per_C, convergence_warnings


def eval_marker_shortcut(model, seqs, labels, hard_idx, scrambled_idx, n_pos):
    X = marker_shortcut_features(seqs)
    probs = model.predict_proba(X)[:, 1]
    preds = probs >= 0.5
    labels_arr = np.asarray(labels, dtype=bool)
    correct = preds == labels_arr

    def acc_on(idx):
        if not idx:
            return float("nan"), 0
        return float(np.mean(correct[idx])), len(idx)

    overall_acc = float(np.mean(correct))
    hard_acc, n_hard = acc_on(hard_idx)
    scrambled_acc, n_scrambled = acc_on(scrambled_idx)
    long_idx = [i for i, s in enumerate(seqs) if len(s) >= LONG_SEQ_THRESH]
    long_acc, n_long = acc_on(long_idx)
    return {
        "n_total": len(labels),
        "n_positives": n_pos,
        "n_hard_negatives": n_hard,
        "n_scrambled_negatives": n_scrambled,
        "n_long_sequences": n_long,
        "test_accuracy": overall_acc,
        "hard_negative_accuracy": hard_acc,
        "scrambled_negative_accuracy": scrambled_acc,
        "long_sequence_accuracy": long_acc,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    b2 = json.loads((RESULTS / "phase1_baselines_trained.json").read_text())
    cyclenav_part6b = json.loads((RESULTS / "cyclenav_shortcut_check.json").read_text())
    marked_reversal_part6b = json.loads((RESULTS / "flare_a2_marked_reversal.json").read_text())

    comparison = {"tasks": {}}

    # ------------------------------------------------------------------
    # cycle-navigation
    # ------------------------------------------------------------------
    print("=== cycle-navigation: trained architectures, full test-set accuracy ===", flush=True)
    task = "cycle-navigation"
    test_seqs, test_labels = load_split(task, "test")
    hard_idx, scrambled_idx = hard_scrambled_split(task, test_seqs, test_labels)
    n_pos = sum(test_labels)
    test_examples = load_prepared_data_from_directory(
        Path(f"languages/{task}/datasets/test"),
        type("I", (), {"use_next_symbols_head": True})()
    )
    trained_rows = {}
    for arch, seed in cyclenav_part6b["selected_seeds"].items():
        row = evaluate_trained_model(task, arch, seed, test_examples, hard_idx, scrambled_idx, n_pos)
        trained_rows[arch] = row
        print(f"  {arch:12s} seed{seed}: test_acc={row['test_accuracy']:.4f} "
              f"hard_acc={row['hard_negative_accuracy']:.4f} "
              f"scrambled_acc={row['scrambled_negative_accuracy']:.4f} "
              f"long_acc={row['long_sequence_accuracy']:.4f}", flush=True)

    cyc = b2["tasks"]["cycle-navigation"]["candidates"]
    comparison["tasks"]["cycle-navigation"] = {
        "primary_comparison": "bagcount-ffn (h=4, h=8) vs. trained architectures' long-sequence accuracy",
        "rationale": "Symbol counts are a sufficient statistic for cycle-navigation's label "
                     "(y = 1[final_digit == (count('>')-count('<')) mod 5]; order among moves "
                     "is irrelevant by commutativity), so bagcount-ffn is limited only by "
                     "representational capacity (breakpoint count), not by missing information. "
                     "An h=4 model, unable to represent more than a few mod-5 boundaries, "
                     "collapses to an 'accept anything grammatical' rule and reproduces the "
                     "trained architectures' accuracy almost exactly.",
        "trained_architectures": trained_rows,
        "grammar_only_ceiling": 0.9666,
        "baselines": {
            "bagcount-ffn_h4": cyc["bagcount-ffn"]["4"],
            "bagcount-ffn_h8": cyc["bagcount-ffn"]["8"],
            "bagcount-ffn_h16_to_h128": {h: cyc["bagcount-ffn"][h] for h in ["16", "32", "64", "128"]},
            "ngram-logreg_SECONDARY_length_confounded": {
                **cyc["ngram-logreg"],
                "caveat": "C=100 (least-regularized option in the grid) was selected on a "
                          "length<=40 validation set and overfits length-specific n-gram "
                          "statistics; long_sequence_accuracy (0.690) is well below overall "
                          "test_accuracy (0.803), indicating a length-generalization confound "
                          "rather than a violation of the linear-separability impossibility "
                          "argument. Not used for the primary claim.",
            },
        },
    }

    # ------------------------------------------------------------------
    # marked-reversal
    # ------------------------------------------------------------------
    print("\n=== marked-reversal: training marker-shortcut baseline (Candidate C) ===", flush=True)
    task = "marked-reversal"
    train_seqs, train_labels = load_split(task, "train")
    val_seqs, val_labels = load_split(task, "validation-short")
    test_seqs, test_labels = load_split(task, "test")
    hard_idx, scrambled_idx = hard_scrambled_split(task, test_seqs, test_labels)
    n_pos = sum(test_labels)

    model, best_C, val_bce_per_C, conv_warnings = train_marker_shortcut(
        train_seqs, train_labels, val_seqs, val_labels
    )
    metrics_C = eval_marker_shortcut(model, test_seqs, test_labels, hard_idx, scrambled_idx, n_pos)
    ckpt_path = CKPT_DIR / "phase1_marked_reversal_marker_shortcut.joblib"
    import joblib
    joblib.dump({"model": model, "feature_spec": "n_hash, relative_offset=(idx_hash-expected_idx)/length, abs(relative_offset)",
                 "source_data": {"train": f"languages/{task}/main.tok",
                                 "validation": f"languages/{task}/datasets/validation-short/main.tok"}},
                ckpt_path)
    print(f"  marker-shortcut: selected_C={best_C} test_acc={metrics_C['test_accuracy']:.4f} "
          f"hard_acc={metrics_C['hard_negative_accuracy']:.4f} "
          f"scrambled_acc={metrics_C['scrambled_negative_accuracy']:.4f} "
          f"long_acc={metrics_C['long_sequence_accuracy']:.4f}", flush=True)

    print("\n=== marked-reversal: trained architectures, full test-set accuracy ===", flush=True)
    test_examples = load_prepared_data_from_directory(
        Path(f"languages/{task}/datasets/test"),
        type("I", (), {"use_next_symbols_head": True})()
    )
    trained_rows = {}
    for arch, seed in marked_reversal_part6b["selected_seeds"].items():
        row = evaluate_trained_model(task, arch, seed, test_examples, hard_idx, scrambled_idx, n_pos)
        trained_rows[arch] = row
        print(f"  {arch:12s} seed{seed}: test_acc={row['test_accuracy']:.4f} "
              f"hard_acc={row['hard_negative_accuracy']:.4f} "
              f"scrambled_acc={row['scrambled_negative_accuracy']:.4f} "
              f"long_acc={row['long_sequence_accuracy']:.4f}", flush=True)

    mr = b2["tasks"]["marked-reversal"]["candidates"]
    comparison["tasks"]["marked-reversal"] = {
        "primary_comparison": "marker-shortcut (Candidate C, new in B3) vs. trained architectures",
        "rationale": "Neither B2 baseline (bagcount-ffn, ngram-logreg) reproduced the trained "
                     "architectures' accuracy: bagcount-ffn destroys all positional information "
                     "and ngram-logreg was kept free of an explicit length feature per "
                     "instruction, so it under-detects marker-POSITION violations (only "
                     "marker-COUNT violations are locally visible to a bounded n-gram window). "
                     "Candidate C's features are restricted by construction to marker count "
                     "and marker position relative to sequence length (n_hash, and a "
                     "length-normalized offset of the marker from the expected middle index "
                     "-- normalized rather than raw-token-distance after an initial raw-offset "
                     "version was found to produce a length-generalization artifact, see code "
                     "comments) -- the 0/1 content tokens never enter the feature vector, so it "
                     "cannot detect content-mismatch hard negatives by construction, not merely "
                     "by capacity or linearity limits.",
        "marker_shortcut_selected_C": best_C,
        "marker_shortcut_val_bce_per_C": val_bce_per_C,
        "marker_shortcut_convergence_issues": conv_warnings or ["none"],
        "trained_architectures": trained_rows,
        "grammar_only_ceiling": marked_reversal_part6b["grammar_only_ceiling"],
        "marker_placement_frac_of_scrambled_negatives": 0.8733,
        "baselines": {
            "marker-shortcut": metrics_C,
            "bagcount-ffn_all_h": mr["bagcount-ffn"],
            "ngram-logreg_pure_no_length_feature": mr["ngram-logreg"],
        },
    }

    # ------------------------------------------------------------------
    # Solving-threshold context (Part 6B's own 0.95 bar): neither task's
    # architectures actually cross it, but cycle-navigation converges tightly
    # to the grammar-only ceiling across all 4 architectures/seeds, while
    # marked-reversal shows much more architecture-dependent spread.
    # ------------------------------------------------------------------
    cyc_trained = comparison["tasks"]["cycle-navigation"]["trained_architectures"]
    cyc_h4 = comparison["tasks"]["cycle-navigation"]["baselines"]["bagcount-ffn_h4"]
    mr_trained = comparison["tasks"]["marked-reversal"]["trained_architectures"]
    mr_shortcut = comparison["tasks"]["marked-reversal"]["baselines"]["marker-shortcut"]

    summary = (
        "cycle-navigation: all four trained architectures converge tightly to "
        f"{min(r['test_accuracy'] for r in cyc_trained.values()):.4f}-"
        f"{max(r['test_accuracy'] for r in cyc_trained.values()):.4f} overall test accuracy "
        f"(long-sequence accuracy identical across all four at 0.9231) with near-zero "
        "hard-negative accuracy -- the signature Part 6B identified as shortcut convergence. "
        f"A bagcount-ffn baseline with h=4 hidden units, which provably cannot maintain "
        f"positional/running-position state (its input is an order-erased 8-dimensional count "
        f"vector), reproduces this almost exactly: test accuracy {cyc_h4['test_accuracy']:.4f}, "
        f"long-sequence accuracy {cyc_h4['long_sequence_accuracy']:.4f} vs. the trained models' "
        f"uniform 0.9231, hard-negative accuracy {cyc_h4['hard_negative_accuracy']:.4f} vs. the "
        f"trained models' 0.0000-0.0090. Since symbol counts are a sufficient statistic for this "
        "task's true label (order among moves is irrelevant by commutativity), this baseline's "
        "match is not an information advantage it shouldn't have -- it is exactly as informed as "
        "the real task requires, and still cannot exceed grammar-detection accuracy because its "
        "capacity (h=4) is far below what representing the mod-5 decision boundary across the "
        "full length-500 test range would require. This is the clean, decisive result: a model "
        "that is structurally incapable of tracking position matches the trained architectures' "
        "accuracy to within 0.1-0.2 percentage points, so the trained architectures' accuracy is "
        "fully consistent with, and does not exceed, grammar-only exploitation. "
        "\n\n"
        "marked-reversal: unlike cycle-navigation, the four trained architectures do NOT "
        f"converge tightly -- test accuracy ranges from {min(r['test_accuracy'] for r in mr_trained.values()):.4f} "
        f"(transformer) to {max(r['test_accuracy'] for r in mr_trained.values()):.4f} (rnn), and none "
        "reach Part 6B's own 0.95 solving-seed threshold even at their best-of-10-seeds pick "
        "(best long-sequence accuracy: rnn 0.94, mamba 0.87, lstm 0.78, transformer 0.77) -- "
        "so this task shows architecture-dependent behavior rather than uniform convergence to a "
        "shared ceiling. Against that backdrop, the marker-shortcut baseline (features restricted "
        "by construction to marker count and length-normalized marker position; the 0/1 content "
        f"tokens never enter its input) reaches test accuracy {mr_shortcut['test_accuracy']:.4f}, "
        f"long-sequence accuracy {mr_shortcut['long_sequence_accuracy']:.4f}. This matches mamba "
        f"closely (mamba: test {mr_trained['mamba']['test_accuracy']:.4f}, long "
        f"{mr_trained['mamba']['long_sequence_accuracy']:.4f} -- long-sequence accuracy is "
        "identical to four decimal places), consistent with mamba's marked-reversal behavior "
        "being explained by marker-structure exploitation alone. lstm and transformer score "
        f"*below* the marker-shortcut baseline ({mr_trained['lstm']['test_accuracy']:.4f} and "
        f"{mr_trained['transformer']['test_accuracy']:.4f} respectively) despite having full "
        "access to content information a marker-only classifier structurally lacks -- given "
        "neither architecture's best seed exceeds 0.78 long-sequence accuracy (well short of "
        "solving), this most likely reflects weak/non-convergent training on this harder task "
        "rather than genuine content-sensitivity that happens to underperform a shortcut. rnn is "
        f"the one architecture that clearly exceeds the marker-shortcut ceiling (test "
        f"{mr_trained['rnn']['test_accuracy']:.4f}, hard-negative accuracy "
        f"{mr_trained['rnn']['hard_negative_accuracy']:.4f} vs. the marker-shortcut's "
        f"{mr_shortcut['hard_negative_accuracy']:.4f}, which is expected to be near chance since "
        "content-mismatch hard negatives are by definition indistinguishable from positives to a "
        "marker-only classifier) -- rnn's selected seed shows real, if partial, sensitivity to "
        "reversal content beyond marker-structure detection. Net pattern: marked-reversal does "
        "NOT show the same clean, architecture-uniform shortcut signature cycle-navigation does; "
        "it shows a shortcut-consistent result for mamba, an inconclusive one for lstm/"
        "transformer (confounded by incomplete convergence), and evidence against a pure-shortcut "
        "explanation for rnn."
    )
    comparison["summary"] = summary
    print(f"\n{summary}\n")

    out_path = RESULTS / "phase1_b3_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
