"""Corrected-dataset experiment, F3: train RNN/LSTM/Transformer/Mamba on the
corrected marked-reversal training set (languages/marked-reversal-fixed/),
matching FLaRe's standard 'rec+ns' training procedure exactly, including
PER-TRIAL hyperparameter sampling (batch token budget, initial learning
rate, next-symbols loss coefficient) via the same random_sample.py logic
train_and_evaluate.bash uses -- NOT fixed hyperparameters across the 10
seeds per architecture.

For each of 4 architectures x 10 trials (40 runs total):
  1. Sample hyperparameters via subprocess calls to recognizers/neural_
     networks/random_sample.py (same distributions as train_and_evaluate.
     bash: max-tokens-per-batch ~ int-loguniform(128,4096); initial-
     learning-rate ~ loguniform(0.0001,0.01); next-symbols-loss-coefficient
     ~ loguniform(0.01,10)).
  2. Get architecture-specific model hyperparameters via recognizers/
     neural_networks/get_architecture_args.py (parameter budget 64000,
     matching the pilot's calibration).
  3. Train via recognizers/neural_networks/train.py (rec+ns objective:
     --use-next-symbols-head, no --use-language-modeling-head), matching
     train_and_evaluate.bash's other fixed hyperparameters exactly
     (init-scale 0.1, max-epochs 1000, optimizer Adam, gradient-clipping 5,
     early-stopping-patience 10, learning-rate-patience 5, decay-factor 0.5,
     examples-per-checkpoint 10000).
  4. Evaluate the trained model's recognition accuracy on BOTH the standard
     FLaRe test set (languages/marked-reversal/datasets/test) and the
     corrected test set (languages/marked-reversal-fixed/datasets/test),
     the latter split into uniform-random/hard-negative halves via the
     corrected test set's negative-kind.txt side-channel.

Results are written incrementally (after each of the 40 runs) so partial
progress is never lost.

Models are saved under models/marked-reversal-fixed/<arch>/rec+ns/
validation-short/<trial_no>, matching the pilot's established directory
convention (recognizers/functions.bash's get_model_dir), so later
mechanistic-verification steps (F5/F6, if approved) can reuse the existing
RNNPatchingHarness/PatchingHarness/TransformerPatchingHarness machinery
directly with task="marked-reversal-fixed".

PYTHONPATH=src:analysis python analysis/fix_train_models_f3.py
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SUBPROCESS_ENV = os.environ.copy()
SUBPROCESS_ENV["PYTHONPATH"] = "."

import numpy as np
import torch

sys.path.insert(0, "src")
from recognizers.neural_networks.data import add_data_arguments, load_vocabulary_data
from recognizers.neural_networks.model_interface import RecognitionModelInterface, ModelInput

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS = REPO_ROOT / "analysis_outputs" / "final_results"
FIXED_LANG_DIR = REPO_ROOT / "languages" / "marked-reversal-fixed"
STANDARD_LANG_DIR = REPO_ROOT / "languages" / "marked-reversal"
MODELS_ROOT = REPO_ROOT / "models" / "marked-reversal-fixed"

ARCHITECTURES = ["rnn", "lstm", "transformer", "mamba"]
N_TRIALS = 10
PARAMETER_BUDGET = 64000
OUT_PATH = RESULTS / "fix_trained_models_marked_reversal.json"


def random_sample(lo, hi, log=False, as_int=False):
    cmd = [sys.executable, "recognizers/neural_networks/random_sample.py"]
    if log:
        cmd.append("--log")
    if as_int:
        cmd.append("--int")
    cmd.extend([str(lo), str(hi)])
    out = subprocess.run(cmd, cwd=SRC_DIR, capture_output=True, text=True, check=True)
    val = out.stdout.strip()
    return int(val) if as_int else float(val)


def get_architecture_args(arch):
    cmd = [sys.executable, "recognizers/neural_networks/get_architecture_args.py",
           "--architecture", arch, "--parameter-budget", str(PARAMETER_BUDGET),
           "--training-data", str(FIXED_LANG_DIR)]
    out = subprocess.run(cmd, cwd=SRC_DIR, capture_output=True, text=True, check=True,
                         env=SUBPROCESS_ENV)
    return out.stdout.strip().split()


def train_one(arch, trial_no):
    model_dir = MODELS_ROOT / arch / "rec+ns" / "validation-short" / str(trial_no)
    max_tokens_per_batch = random_sample(128, 4096, log=True, as_int=True)
    initial_learning_rate = random_sample(0.0001, 0.01, log=True)
    next_symbols_loss_coefficient = random_sample(0.01, 10, log=True)
    arch_flags = get_architecture_args(arch)

    cmd = [
        sys.executable, "recognizers/neural_networks/train.py",
        "--output", str(model_dir),
        "--training-data", str(FIXED_LANG_DIR),
        "--validation-data", "validation-short",
        *arch_flags,
        "--init-scale", "0.1",
        "--use-next-symbols-head",
        "--next-symbols-loss-coefficient", str(next_symbols_loss_coefficient),
        "--max-epochs", "1000",
        "--max-tokens-per-batch", str(max_tokens_per_batch),
        "--optimizer", "Adam",
        "--initial-learning-rate", str(initial_learning_rate),
        "--gradient-clipping-threshold", "5",
        "--early-stopping-patience", "10",
        "--learning-rate-patience", "5",
        "--learning-rate-decay-factor", "0.5",
        "--examples-per-checkpoint", "10000",
    ]
    hyperparams = {
        "max_tokens_per_batch": max_tokens_per_batch,
        "initial_learning_rate": initial_learning_rate,
        "next_symbols_loss_coefficient": next_symbols_loss_coefficient,
        "architecture_flags": arch_flags,
    }
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=SRC_DIR, env=SUBPROCESS_ENV,
                          capture_output=True, text=True)
    duration = time.time() - t0
    log_tail = (proc.stdout or "")[-3000:] + "\n---STDERR---\n" + (proc.stderr or "")[-3000:]
    return model_dir, hyperparams, duration, proc.returncode, log_tail


# ---------------------------------------------------------------------------
# lightweight in-process evaluation (recognition accuracy only)
# ---------------------------------------------------------------------------

def load_model(model_dir, task_dir, arch):
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    iface = RecognitionModelInterface()
    iface.add_arguments(parser)
    iface.add_forward_arguments(parser)
    tmp = tempfile.mkdtemp(prefix="fix_f3_eval_")
    shutil.rmtree(tmp, ignore_errors=True)
    args = parser.parse_args([
        "--output", tmp, "--training-data", str(task_dir),
        "--architecture", arch,
        "--load-model", str(model_dir), "--load-parameters", "main",
    ])
    vocab = load_vocabulary_data(args, parser)
    saver = iface.construct_saver(args, vocab)
    saver.model.eval()
    return iface, saver


def load_split(task_dir):
    toks = (task_dir / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (task_dir / "labels.txt").read_text().splitlines()]
    seqs = [line.split() for line in toks]
    return seqs, labels


def evaluate_recognition(iface, saver, seqs, labels, vocab_tokens):
    tok2idx = {t: i for i, t in enumerate(vocab_tokens)}
    device = next(saver.model.parameters()).device
    preds = []
    n_failed = 0
    for seq, label in zip(seqs, labels):
        try:
            idx = [tok2idx[t] for t in seq]
            eos_index = saver.kwargs["eos_index"]
            x = torch.tensor([idx + [eos_index]], dtype=torch.long, device=device)
            last_index = torch.tensor([len(idx)], dtype=torch.long, device=device)
            positive_mask = torch.zeros(1, dtype=torch.bool, device=device)
            mi = ModelInput(x, last_index, positive_mask)
            with torch.no_grad():
                rec, _, _ = iface.get_logits(saver.model, mi)
            preds.append(bool(rec.item() > 0))
        except Exception:
            preds.append(None)
            n_failed += 1
    correct = [(p == bool(l)) if p is not None else None for p, l in zip(preds, labels)]
    return correct, n_failed


def subset_acc(correct, idx):
    vals = [correct[i] for i in idx if correct[i] is not None]
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def evaluate_model_both_test_sets(model_dir, arch):
    vocab_tokens = torch.load(FIXED_LANG_DIR / "main.vocab", weights_only=False)["tokens"]

    # standard FLaRe test set
    std_seqs, std_labels = load_split(STANDARD_LANG_DIR / "datasets" / "test")
    iface, saver = load_model(model_dir, STANDARD_LANG_DIR, arch)
    std_correct, std_failed = evaluate_recognition(iface, saver, std_seqs, std_labels, vocab_tokens)
    std_acc, std_n = subset_acc(std_correct, list(range(len(std_labels))))

    # corrected test set, split by negative kind
    fixed_test_dir = FIXED_LANG_DIR / "datasets" / "test"
    fixed_seqs, fixed_labels = load_split(fixed_test_dir)
    fixed_kinds = (fixed_test_dir / "negative-kind.txt").read_text().splitlines()
    iface2, saver2 = load_model(model_dir, FIXED_LANG_DIR, arch)
    fixed_correct, fixed_failed = evaluate_recognition(iface2, saver2, fixed_seqs, fixed_labels, vocab_tokens)
    pos_idx = [i for i, l in enumerate(fixed_labels) if l == 1]
    uniform_idx = [i for i, (l, k) in enumerate(zip(fixed_labels, fixed_kinds)) if l == 0 and k == "uniform_random"]
    hard_idx = [i for i, (l, k) in enumerate(zip(fixed_labels, fixed_kinds)) if l == 0 and k == "hard_negative"]
    fixed_overall_acc, fixed_n = subset_acc(fixed_correct, list(range(len(fixed_labels))))
    fixed_pos_acc, n_pos = subset_acc(fixed_correct, pos_idx)
    fixed_uniform_acc, n_uniform = subset_acc(fixed_correct, uniform_idx)
    fixed_hard_acc, n_hard = subset_acc(fixed_correct, hard_idx)

    return {
        "standard_flare_test_set": {
            "accuracy": std_acc, "n": std_n, "n_inference_failed": std_failed,
        },
        "corrected_test_set": {
            "overall_accuracy": fixed_overall_acc, "n": fixed_n, "n_inference_failed": fixed_failed,
            "positive_accuracy": fixed_pos_acc, "n_positive": n_pos,
            "uniform_random_half_accuracy": fixed_uniform_acc, "n_uniform_random": n_uniform,
            "hard_negative_half_accuracy": fixed_hard_acc, "n_hard_negative": n_hard,
        },
    }


def save_results(results):
    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    results = {
        "task": "marked-reversal",
        "experiment": "corrected-dataset-fix (F3: training on corrected data)",
        "description": (
            "Trains RNN/LSTM/Transformer/Mamba on the corrected marked-reversal training set "
            "(languages/marked-reversal-fixed), rec+ns objective, matching FLaRe's standard "
            "training procedure exactly INCLUDING per-trial hyperparameter sampling (batch "
            "token budget, initial learning rate, next-symbols loss coefficient), via the same "
            "random_sample.py logic train_and_evaluate.bash uses. 10 independent trials per "
            "architecture (40 runs total). Evaluated on both the standard FLaRe test set and "
            "the corrected test set (split into uniform-random/hard-negative halves)."
        ),
        "parameter_budget": PARAMETER_BUDGET,
        "n_trials_per_architecture": N_TRIALS,
        "hyperparameter_sampling_distributions": {
            "max_tokens_per_batch": "int, log-uniform(128, 4096)",
            "initial_learning_rate": "log-uniform(0.0001, 0.01)",
            "next_symbols_loss_coefficient": "log-uniform(0.01, 10)",
        },
        "fixed_hyperparameters": {
            "init_scale": 0.1, "max_epochs": 1000, "optimizer": "Adam",
            "gradient_clipping_threshold": 5, "early_stopping_patience": 10,
            "learning_rate_patience": 5, "learning_rate_decay_factor": 0.5,
            "examples_per_checkpoint": 10000,
        },
        "runs": [],
    }
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text())
            if existing.get("runs"):
                results["runs"] = existing["runs"]
                print(f"resuming: found {len(results['runs'])} completed runs already", flush=True)
        except Exception:
            pass

    done_keys = {(r["arch"], r["trial_no"]) for r in results["runs"]}

    for arch in ARCHITECTURES:
        for trial_no in range(1, N_TRIALS + 1):
            if (arch, trial_no) in done_keys:
                print(f"skipping {arch} trial {trial_no} (already done)", flush=True)
                continue
            print(f"\n=== training {arch} trial {trial_no} ===", flush=True)
            model_dir, hyperparams, duration, returncode, log_tail = train_one(arch, trial_no)
            run_record = {
                "arch": arch, "trial_no": trial_no, "model_dir": str(model_dir),
                "hyperparameters": hyperparams, "training_duration_seconds": duration,
                "train_returncode": returncode,
            }
            if returncode != 0:
                run_record["train_failed"] = True
                run_record["log_tail"] = log_tail
                print(f"  TRAINING FAILED (returncode={returncode}), see log_tail in output", flush=True)
            else:
                print(f"  training done in {duration:.1f}s, evaluating...", flush=True)
                try:
                    eval_results = evaluate_model_both_test_sets(model_dir, arch)
                    run_record.update(eval_results)
                    std_acc = eval_results["standard_flare_test_set"]["accuracy"]
                    fx = eval_results["corrected_test_set"]
                    print(f"  standard_test_acc={std_acc:.4f}  corrected_overall={fx['overall_accuracy']:.4f} "
                          f"pos={fx['positive_accuracy']:.4f} uniform={fx['uniform_random_half_accuracy']:.4f} "
                          f"hard={fx['hard_negative_half_accuracy']:.4f}", flush=True)
                except Exception as e:
                    run_record["eval_failed"] = True
                    run_record["eval_error"] = repr(e)
                    print(f"  EVAL FAILED: {e!r}", flush=True)
            results["runs"].append(run_record)
            save_results(results)

    print(f"\nAll runs complete. Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
