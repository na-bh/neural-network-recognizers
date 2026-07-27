"""B2: train and evaluate the grammar-only baselines designed in B1
(analysis_outputs/final_results/phase1_grammar_baselines_design.json).

Two candidates, both implemented as standalone feature-extraction +
shallow-classifier pipelines (not wired into RecognitionModelInterface --
see the "integration_strategy" section of the B1 design doc for why):

  Candidate A ("ngram-logreg"): char n-grams (n=1..4) -> L2-regularized
  logistic regression (sklearn). L2 strength C is selected on validation-short
  (the functional analogue of early stopping for a convex model with a direct
  solver -- there is no overfitting-via-too-many-epochs risk to guard against
  with lbfgs, so model-selection-via-validation is done over regularization
  strength instead, using the same validation-short split the main pipeline
  uses).

  Candidate B ("bagcount-ffn"): order-erased exact symbol counts -> a
  1-hidden-layer ReLU network (h hidden units, swept per the amended B1 plan),
  trained with the training protocol below.

Training/validation/test data are read directly from the same plaintext files
the main pipeline uses (languages/<task>/main.tok+labels.txt for training,
languages/<task>/datasets/validation-short for early stopping /
model-selection, languages/<task>/datasets/test for the standard test set).
No new data is generated.

Training protocol for Candidate B, matched to Butoi et al. 2025's protocol
(src/recognizers/neural_networks/training_loop.py) as closely as feasible,
with explicit, itemized deviations recorded in every output JSON's
"protocol_deviations_from_butoi_et_al" field:
  - optimizer: AdamW (Butoi et al. use plain Adam -- requested explicitly for
    this experiment; AdamW with weight_decay=0.01 is not identical to Adam)
  - LR schedule: ReduceLROnPlateau(patience=learning_rate_patience-1, factor=0.5),
    identical mechanics to Butoi et al.
  - early stopping: patience=10 checkpoints without improvement in validation
    recognition BCE loss, identical mechanics to Butoi et al.
    (rau.training.early_stopping.UpdatesWithoutImprovement, mode='min')
  - gradient clipping: L2 norm threshold 5, identical to Butoi et al.
  - max epochs: 1000, identical to Butoi et al.
  - checkpoint frequency: once per epoch (Butoi et al. checkpoint every
    examples_per_checkpoint=10,000 training examples, which equals one epoch
    for this training-set size, so this matches exactly)
  - batching: fixed minibatch size 256, shuffled per epoch (Butoi et al. use
    dynamic token-budget batching, which has no equivalent for our fixed-size
    feature vectors; this is a necessary substitution, not a free choice)
  - initial learning rate: fixed at 1e-2 (Butoi et al. randomly sample a
    per-trial initial learning rate from a search range; a fixed value is used
    here since learning rate is not the object of study for this experiment)
  - loss: BCEWithLogitsLoss on the single recognition logit, identical to
    Butoi et al.'s recognition_cross_entropy term; no language-modeling or
    next-symbols auxiliary heads (not applicable -- these baselines have no
    notion of predicting the next token)

PYTHONPATH=src:analysis python analysis/phase1_baselines_train.py
"""

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from flare_a1_task_audit import TASKS as A1_TASKS

LANG = Path("languages")
RESULTS = Path("analysis_outputs/final_results")
CKPT_DIR = RESULTS / "phase1_checkpoints"
HARD_THRESHOLD = 0.8
LONG_SEQ_THRESH = 400

CYCLENAV_MOVES = {"<", "=", ">"}
CYCLENAV_DIGITS = {"0", "1", "2", "3", "4"}

H_GRID = {
    "cycle-navigation": [4, 8, 16, 32, 64, 128],
    "marked-reversal": [8, 32, 128, 512],
}

# Candidate B training protocol (see module docstring for Butoi et al. comparison).
MAX_EPOCHS = 1000
BATCH_SIZE = 256
INITIAL_LR = 1e-2
GRAD_CLIP = 5.0
EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE = 5  # ReduceLROnPlateau uses LR_PATIENCE - 1, matching rau's convention
LR_DECAY_FACTOR = 0.5
WEIGHT_DECAY = 0.01

PROTOCOL_DEVIATIONS = [
    "optimizer: AdamW with weight_decay=0.01, vs. Butoi et al.'s plain Adam "
    "(weight_decay=0) -- requested explicitly for this experiment.",
    "batching: fixed minibatch size 256 shuffled per epoch, vs. Butoi et al.'s "
    "dynamic token-budget batching -- not applicable since inputs here are "
    "fixed-size feature vectors, not variable-length token sequences.",
    "initial learning rate fixed at 1e-2, vs. Butoi et al.'s per-trial random "
    "sampling from a search range -- learning rate is not the object of study "
    "here.",
    "Candidate A (logistic regression) is optimized to convergence with "
    "sklearn's lbfgs solver, not SGD/Adam-style epoch training -- this is a "
    "different but standard optimization approach for a convex model; there "
    "is no epoch-wise early-stopping analogue, so L2 regularization strength "
    "C is instead selected via validation-short loss, serving the same "
    "overfitting-control role that early stopping serves in Butoi et al.'s "
    "protocol.",
]


def load_split(task, split):
    d = LANG / task if split == "train" else LANG / task / "datasets" / split
    toks = (d / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (d / "labels.txt").read_text().splitlines()]
    seqs = [line.split() for line in toks]
    return seqs, labels


def is_grammatical_cyclenav(seq):
    if len(seq) == 0:
        return False
    body, last = seq[:-1], seq[-1]
    return all(t in CYCLENAV_MOVES for t in body) and last in CYCLENAV_DIGITS


def hard_scrambled_split(task, seqs, labels):
    """Classify negatives into hard vs. scrambled, matching flare_a1_task_audit.py
    / cyclenav_s2_shortcut_check.py exactly (same functions, same threshold)."""
    if task == "cycle-navigation":
        def frac(seq):
            return 0.0 if is_grammatical_cyclenav(seq) else 1.0
        # is_grammatical is a 0/1 shape check (not a graded first-violation
        # fraction); treat "grammatical" as hard (frac=0, i.e. < threshold is
        # scrambled convention would misclassify) -- use is_grammatical
        # directly instead of the generic frac>=0.8 rule for this task, exactly
        # mirroring cyclenav_s2_shortcut_check.py's hard_vs_scrambled_accuracy.
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


def grammar_only_ceiling(n_pos, n_hard, n_scrambled):
    n_total = n_pos + n_hard + n_scrambled
    return (n_pos + n_scrambled + 0.5 * n_hard) / n_total


def to_char_strings(seqs):
    return ["".join(s) for s in seqs]


def accuracy_and_bce(probs, labels):
    labels = np.asarray(labels, dtype=np.float64)
    preds = (probs >= 0.5).astype(np.float64)
    acc = float(np.mean(preds == labels)) if len(labels) else float("nan")
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    bce = float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))) if len(labels) else float("nan")
    return acc, bce


def subset_accuracy(probs, labels, idx):
    if not idx:
        return float("nan"), 0
    probs_sub = probs[idx]
    labels_sub = np.asarray(labels)[idx]
    preds = (probs_sub >= 0.5).astype(int)
    return float(np.mean(preds == labels_sub)), len(idx)


def long_seq_accuracy(probs, labels, seqs):
    idx = [i for i, s in enumerate(seqs) if len(s) >= LONG_SEQ_THRESH]
    return subset_accuracy(probs, labels, idx)


# ----------------------------------------------------------------------
# Candidate A: char n-gram (1..4) + logistic regression
# ----------------------------------------------------------------------

def train_candidate_A(task, train_seqs, train_labels, val_seqs, val_labels):
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=False)
    X_train = vectorizer.fit_transform(to_char_strings(train_seqs))
    X_val = vectorizer.transform(to_char_strings(val_seqs))

    C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    convergence_warnings = []
    results_per_C = {}
    best_C, best_val_bce, best_model = None, float("inf"), None
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
        results_per_C[C] = val_bce
        if val_bce < best_val_bce:
            best_C, best_val_bce, best_model = C, val_bce, model

    return {
        "vectorizer": vectorizer,
        "model": best_model,
        "selected_C": best_C,
        "val_bce_per_C": results_per_C,
        "n_features": X_train.shape[1],
        "convergence_warnings": convergence_warnings,
    }


def eval_candidate_A(vectorizer, model, seqs, labels):
    X = vectorizer.transform(to_char_strings(seqs))
    probs = model.predict_proba(X)[:, 1]
    return probs


# ----------------------------------------------------------------------
# Candidate B: bag-of-symbol-counts + small FFN
# ----------------------------------------------------------------------

class BagCountFFN(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_features(seqs, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for i, s in enumerate(seqs):
        for t in s:
            X[i, idx[t]] += 1.0
    return X


def train_candidate_B(task, h, vocab, train_seqs, train_labels, val_seqs, val_labels):
    X_train = torch.tensor(count_features(train_seqs, vocab))
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    X_val = torch.tensor(count_features(val_seqs, vocab))
    y_val = torch.tensor(val_labels, dtype=torch.float32)

    model = BagCountFFN(len(vocab), h)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE - 1, factor=LR_DECAY_FACTOR
    )
    loss_fn = nn.BCEWithLogitsLoss()

    n = X_train.shape[0]
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epoch_log = []
    issues = []
    stopped_early = False
    nan_detected = False

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        epoch_loss_sum, epoch_n = 0.0, 0
        for start in range(0, n, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            xb, yb = X_train[batch_idx], y_train[batch_idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss_sum += loss.item() * len(batch_idx)
            epoch_n += len(batch_idx)
        train_loss = epoch_loss_sum / epoch_n

        model.eval()
        with torch.inference_mode():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val).item()

        if math.isnan(train_loss) or math.isnan(val_loss):
            nan_detected = True
            issues.append(f"NaN loss encountered at epoch {epoch}; stopping.")
            break

        scheduler.step(val_loss)
        epoch_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                           "lr": optimizer.param_groups[0]["lr"]})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                stopped_early = True
                break

    if best_state is None:
        issues.append("training never improved on validation; using last-epoch weights.")
        best_state = model.state_dict()
    else:
        model.load_state_dict(best_state)

    if not stopped_early and not nan_detected and len(epoch_log) >= MAX_EPOCHS:
        issues.append(f"reached max_epochs={MAX_EPOCHS} without early stopping triggering.")

    return {
        "model": model,
        "best_val_bce": best_val_loss,
        "num_epochs_run": len(epoch_log),
        "stopped_early": stopped_early,
        "nan_detected": nan_detected,
        "issues": issues,
        "epoch_log_tail": epoch_log[-5:],
    }


def eval_candidate_B(model, seqs, vocab):
    X = torch.tensor(count_features(seqs, vocab))
    model.eval()
    with torch.inference_mode():
        probs = torch.sigmoid(model(X)).numpy()
    return probs


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def load_vocab(task):
    vd = torch.load(LANG / task / "main.vocab", weights_only=False)
    return vd["tokens"]


def evaluate_all(task_name, probs, labels, seqs, hard_idx, scrambled_idx, n_pos):
    overall_acc, overall_bce = accuracy_and_bce(probs, labels)
    hard_acc, n_hard = subset_accuracy(probs, labels, hard_idx)
    scrambled_acc, n_scrambled = subset_accuracy(probs, labels, scrambled_idx)
    long_acc, n_long = long_seq_accuracy(probs, labels, seqs)
    ceiling = grammar_only_ceiling(n_pos, len(hard_idx), len(scrambled_idx))
    return {
        "n_total": len(labels),
        "n_positives": n_pos,
        "n_hard_negatives": n_hard,
        "n_scrambled_negatives": n_scrambled,
        "n_long_sequences": n_long,
        "test_accuracy": overall_acc,
        "test_bce": overall_bce,
        "hard_negative_accuracy": hard_acc,
        "scrambled_negative_accuracy": scrambled_acc,
        "long_sequence_accuracy": long_acc,
        "grammar_only_ceiling": ceiling,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {"protocol_deviations_from_butoi_et_al": PROTOCOL_DEVIATIONS,
                   "training_protocol_candidate_B": {
                       "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
                       "initial_learning_rate": INITIAL_LR, "max_epochs": MAX_EPOCHS,
                       "batch_size": BATCH_SIZE, "gradient_clipping_threshold": GRAD_CLIP,
                       "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                       "lr_scheduler": "ReduceLROnPlateau", "lr_patience": LR_PATIENCE,
                       "lr_decay_factor": LR_DECAY_FACTOR,
                   },
                   "training_protocol_candidate_A": {
                       "solver": "lbfgs", "penalty": "l2", "max_iter": 5000,
                       "C_grid": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                       "model_selection": "validation-short BCE loss (early-stopping analogue)",
                   },
                   "tasks": {}}

    for task in ["cycle-navigation", "marked-reversal"]:
        print(f"\n=== {task} ===", flush=True)
        t0 = time.time()
        vocab = load_vocab(task)
        train_seqs, train_labels = load_split(task, "train")
        val_seqs, val_labels = load_split(task, "validation-short")
        test_seqs, test_labels = load_split(task, "test")

        test_hard_idx, test_scrambled_idx = hard_scrambled_split(task, test_seqs, test_labels)
        n_pos_test = sum(test_labels)
        print(f"  test: n={len(test_labels)} n_pos={n_pos_test} "
              f"n_hard_neg={len(test_hard_idx)} n_scrambled_neg={len(test_scrambled_idx)}", flush=True)

        task_results = {"vocab": vocab, "candidates": {}}

        # --- Candidate A ---
        print("  training candidate A (ngram-logreg)...", flush=True)
        tA = train_candidate_A(task, train_seqs, train_labels, val_seqs, val_labels)
        probs_test = eval_candidate_A(tA["vectorizer"], tA["model"], test_seqs, test_labels)
        metrics = evaluate_all(task, probs_test, test_labels, test_seqs, test_hard_idx, test_scrambled_idx, n_pos_test)
        ckpt_path = CKPT_DIR / f"phase1_{task.replace('-', '_')}_ngram_logreg.joblib"
        import joblib
        joblib.dump({"vectorizer": tA["vectorizer"], "model": tA["model"],
                     "source_data": {"train": f"languages/{task}/main.tok",
                                     "validation": f"languages/{task}/datasets/validation-short/main.tok"}},
                    ckpt_path)
        task_results["candidates"]["ngram-logreg"] = {
            "selected_C": tA["selected_C"],
            "val_bce_per_C": tA["val_bce_per_C"],
            "n_features": tA["n_features"],
            "convergence_issues": tA["convergence_warnings"] or ["none"],
            "checkpoint_path": str(ckpt_path),
            **metrics,
        }
        print(f"    selected_C={tA['selected_C']} test_acc={metrics['test_accuracy']:.4f} "
              f"hard_acc={metrics['hard_negative_accuracy']:.4f} "
              f"scrambled_acc={metrics['scrambled_negative_accuracy']:.4f} "
              f"ceiling={metrics['grammar_only_ceiling']:.4f}", flush=True)

        # --- Candidate B, swept over h ---
        task_results["candidates"]["bagcount-ffn"] = {}
        for h in H_GRID[task]:
            print(f"  training candidate B (bagcount-ffn, h={h})...", flush=True)
            tB = train_candidate_B(task, h, vocab, train_seqs, train_labels, val_seqs, val_labels)
            probs_test = eval_candidate_B(tB["model"], test_seqs, vocab)
            metrics = evaluate_all(task, probs_test, test_labels, test_seqs, test_hard_idx, test_scrambled_idx, n_pos_test)
            ckpt_path = CKPT_DIR / f"phase1_{task.replace('-', '_')}_bagcount_ffn_h{h}.pt"
            torch.save({"state_dict": tB["model"].state_dict(), "vocab": vocab, "hidden_units": h,
                        "source_data": {"train": f"languages/{task}/main.tok",
                                        "validation": f"languages/{task}/datasets/validation-short/main.tok"}},
                       ckpt_path)
            task_results["candidates"]["bagcount-ffn"][str(h)] = {
                "hidden_units": h,
                "best_val_bce": tB["best_val_bce"],
                "num_epochs_run": tB["num_epochs_run"],
                "stopped_early": tB["stopped_early"],
                "convergence_issues": tB["issues"] or ["none"],
                "checkpoint_path": str(ckpt_path),
                **metrics,
            }
            print(f"    h={h} epochs={tB['num_epochs_run']} stopped_early={tB['stopped_early']} "
                  f"test_acc={metrics['test_accuracy']:.4f} hard_acc={metrics['hard_negative_accuracy']:.4f} "
                  f"scrambled_acc={metrics['scrambled_negative_accuracy']:.4f} "
                  f"long_acc={metrics['long_sequence_accuracy']:.4f} "
                  f"issues={tB['issues'] or 'none'}", flush=True)

        all_results["tasks"][task] = task_results
        print(f"  {task} done in {time.time()-t0:.1f}s", flush=True)

    out_path = RESULTS / "phase1_baselines_trained.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
