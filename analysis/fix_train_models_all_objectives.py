"""Corrected-dataset experiments, ALL-OBJECTIVES training: trains RNN/LSTM/
Transformer/Mamba on a corrected training set under ALL FOUR of FLaRe's
loss-term combinations (rec, rec+lm, rec+ns, rec+lm+ns), not rec+ns alone.

Motivated by analysis/fix_objective_methodology_audit.py's finding that
rec+ns -- the ONLY objective this whole pilot's Phase 1-3 work and the
original marked-reversal fix F3 used -- is essentially NEVER FLaRe's own
selected-best loss term for marked-reversal (1/6 architecture x validation-
set cells) and is beaten specifically on RNN for binary-addition (rec+lm
wins by ~8 points). Per explicit user decision, BOTH corrected-dataset fix
experiments are now expanded to the FULL FLaRe grid: 4 architectures x 4
loss terms x 10 trials = 160 runs per task.

For marked-reversal-fixed, the 40 rec+ns runs already completed (F3, prior
session) are CARRIED FORWARD as-is (not retrained) -- only the 120 new
(rec, rec+lm, rec+lm+ns) x 4 archs x 10 trials runs are executed here.
For binary-addition-fixed, all 160 runs are new (BF3 had not started
training before this scope expansion).

Loss-term -> train.py flag mapping (matching train_and_evaluate.bash
exactly): rec -> no auxiliary head; lm -> --use-language-modeling-head
--language-modeling-loss-coefficient <log-uniform(0.01,10)>; ns ->
--use-next-symbols-head --next-symbols-loss-coefficient <log-uniform
(0.01,10)>. max-tokens-per-batch ~ int-log-uniform(128,4096); initial-
learning-rate ~ log-uniform(0.0001,0.01) -- SAMPLED PER TRIAL regardless
of loss term, matching train_and_evaluate.bash's random_sample calls.

Models saved under models/<task>-fixed/<arch>/<loss_terms>/validation-short/
<trial_no>, matching FLaRe's own get_model_dir convention exactly (so any
future mechanistic-verification work can reuse the existing patching
harnesses unchanged with task="<task>-fixed").

Results written incrementally after every run (never loses partial progress
across a ~20-hour, 280-run background job).

PYTHONPATH=src:analysis python analysis/fix_train_models_all_objectives.py --task binary-addition
PYTHONPATH=src:analysis python analysis/fix_train_models_all_objectives.py --task marked-reversal
"""

import argparse as py_argparse
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

ARCHITECTURES = ["rnn", "lstm", "transformer", "mamba"]
LOSS_TERM_COMBOS = ["rec", "rec+lm", "rec+ns", "rec+lm+ns"]
N_TRIALS = 10
PARAMETER_BUDGET = 64000

TASK_CONFIG = {
    "marked-reversal": {
        "fixed_lang_dir": REPO_ROOT / "languages" / "marked-reversal-fixed",
        "standard_lang_dir": REPO_ROOT / "languages" / "marked-reversal",
        "models_root": REPO_ROOT / "models" / "marked-reversal-fixed",
        "out_path": RESULTS / "fix_trained_models_marked_reversal_all_objectives.json",
        "prior_rec_ns_only_results": RESULTS / "fix_trained_models_marked_reversal.json",
    },
    "binary-addition": {
        "fixed_lang_dir": REPO_ROOT / "languages" / "binary-addition-fixed",
        "standard_lang_dir": REPO_ROOT / "languages" / "binary-addition",
        "models_root": REPO_ROOT / "models" / "binary-addition-fixed",
        "out_path": RESULTS / "fix_trained_models_binary_addition_all_objectives.json",
        "prior_rec_ns_only_results": None,
    },
}


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


def get_architecture_args(arch, fixed_lang_dir):
    cmd = [sys.executable, "recognizers/neural_networks/get_architecture_args.py",
           "--architecture", arch, "--parameter-budget", str(PARAMETER_BUDGET),
           "--training-data", str(fixed_lang_dir)]
    out = subprocess.run(cmd, cwd=SRC_DIR, capture_output=True, text=True, check=True,
                         env=SUBPROCESS_ENV)
    return out.stdout.strip().split()


def train_one(arch, loss_terms, trial_no, fixed_lang_dir, models_root):
    model_dir = models_root / arch / loss_terms / "validation-short" / str(trial_no)
    max_tokens_per_batch = random_sample(128, 4096, log=True, as_int=True)
    initial_learning_rate = random_sample(0.0001, 0.01, log=True)
    arch_flags = get_architecture_args(arch, fixed_lang_dir)

    loss_term_flags = []
    hyperparams = {
        "max_tokens_per_batch": max_tokens_per_batch,
        "initial_learning_rate": initial_learning_rate,
        "architecture_flags": arch_flags,
    }
    for term in loss_terms.split("+"):
        if term == "rec":
            continue
        elif term == "lm":
            coef = random_sample(0.01, 10, log=True)
            loss_term_flags += ["--use-language-modeling-head", "--language-modeling-loss-coefficient", str(coef)]
            hyperparams["language_modeling_loss_coefficient"] = coef
        elif term == "ns":
            coef = random_sample(0.01, 10, log=True)
            loss_term_flags += ["--use-next-symbols-head", "--next-symbols-loss-coefficient", str(coef)]
            hyperparams["next_symbols_loss_coefficient"] = coef
        else:
            raise ValueError(f"invalid loss term {term!r}")

    cmd = [
        sys.executable, "recognizers/neural_networks/train.py",
        "--output", str(model_dir),
        "--training-data", str(fixed_lang_dir),
        "--validation-data", "validation-short",
        *arch_flags,
        "--init-scale", "0.1",
        *loss_term_flags,
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
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=SRC_DIR, env=SUBPROCESS_ENV, capture_output=True, text=True)
    duration = time.time() - t0
    log_tail = (proc.stdout or "")[-3000:] + "\n---STDERR---\n" + (proc.stderr or "")[-3000:]
    return model_dir, hyperparams, duration, proc.returncode, log_tail


# ---------------------------------------------------------------------------
# lightweight in-process evaluation (recognition accuracy only)
# ---------------------------------------------------------------------------

def load_model(model_dir, task_dir, arch):
    parser = py_argparse.ArgumentParser()
    add_data_arguments(parser)
    iface = RecognitionModelInterface()
    iface.add_arguments(parser)
    iface.add_forward_arguments(parser)
    tmp = tempfile.mkdtemp(prefix="fix_all_obj_eval_")
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
    eos_index = saver.kwargs["eos_index"]
    for seq, label in zip(seqs, labels):
        try:
            idx = [tok2idx[t] for t in seq]
            # eos_index is None whenever the model has NEITHER a language-modeling
            # NOR a next-symbols head (plain "rec"-only training) -- pad_sequences()
            # (model_interface.py) only appends EOS when eos is not None, matching
            # here exactly rather than assuming every model always has an EOS slot.
            # last_index must match: it is a GATHER index into the input tensor, so
            # when EOS is appended (valid indices 0..len(idx)), the appended EOS
            # sits at index len(idx); when EOS is NOT appended (valid indices
            # 0..len(idx)-1), the last valid index is len(idx)-1. Using len(idx)
            # unconditionally (the original bug) is one past the end whenever
            # eos_index is None -- confirmed empirically to either hard-crash
            # (transformer: CUDA "scatter gather kernel index out of bounds") or
            # silently return a garbage CONSTANT logit independent of the input
            # (rnn: last_index=len(idx) returned identical logits for every
            # different test sequence, vs. genuinely input-varying logits at
            # len(idx)-1) -- i.e. every "rec"-only (no ns/lm head) evaluation
            # done before this fix is corrupted, not just the transformer crash.
            if eos_index is not None:
                content = idx + [eos_index]
                last_index_val = len(idx)
            elif len(idx) == 0:
                # a zero-length input with no EOS slot has no valid gather
                # position at all (last_index would be -1, ALSO failing the
                # CUDA bounds assertion and poisoning the whole process) --
                # every task here structurally requires non-empty content to
                # be a valid positive, so an empty string is unambiguously a
                # negative; skip the forward pass and predict reject directly
                # rather than attempt an ill-defined gather.
                preds.append(False)
                continue
            else:
                content = idx
                last_index_val = len(idx) - 1
            x = torch.tensor([content], dtype=torch.long, device=device)
            last_index = torch.tensor([last_index_val], dtype=torch.long, device=device)
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


def evaluate_model_both_test_sets(model_dir, arch, fixed_lang_dir, standard_lang_dir):
    vocab_tokens = torch.load(fixed_lang_dir / "main.vocab", weights_only=False)["tokens"]

    std_seqs, std_labels = load_split(standard_lang_dir / "datasets" / "test")
    iface, saver = load_model(model_dir, standard_lang_dir, arch)
    std_correct, std_failed = evaluate_recognition(iface, saver, std_seqs, std_labels, vocab_tokens)
    std_acc, std_n = subset_acc(std_correct, list(range(len(std_labels))))

    fixed_test_dir = fixed_lang_dir / "datasets" / "test"
    fixed_seqs, fixed_labels = load_split(fixed_test_dir)
    fixed_kinds = (fixed_test_dir / "negative-kind.txt").read_text().splitlines()
    iface2, saver2 = load_model(model_dir, fixed_lang_dir, arch)
    fixed_correct, fixed_failed = evaluate_recognition(iface2, saver2, fixed_seqs, fixed_labels, vocab_tokens)
    pos_idx = [i for i, l in enumerate(fixed_labels) if l == 1]
    uniform_idx = [i for i, (l, k) in enumerate(zip(fixed_labels, fixed_kinds)) if l == 0 and k == "uniform_random"]
    hard_idx = [i for i, (l, k) in enumerate(zip(fixed_labels, fixed_kinds)) if l == 0 and k == "hard_negative"]
    fixed_overall_acc, fixed_n = subset_acc(fixed_correct, list(range(len(fixed_labels))))
    fixed_pos_acc, n_pos = subset_acc(fixed_correct, pos_idx)
    fixed_uniform_acc, n_uniform = subset_acc(fixed_correct, uniform_idx)
    fixed_hard_acc, n_hard = subset_acc(fixed_correct, hard_idx)

    return {
        "standard_flare_test_set": {"accuracy": std_acc, "n": std_n, "n_inference_failed": std_failed},
        "corrected_test_set": {
            "overall_accuracy": fixed_overall_acc, "n": fixed_n, "n_inference_failed": fixed_failed,
            "positive_accuracy": fixed_pos_acc, "n_positive": n_pos,
            "uniform_random_half_accuracy": fixed_uniform_acc, "n_uniform_random": n_uniform,
            "hard_negative_half_accuracy": fixed_hard_acc, "n_hard_negative": n_hard,
        },
    }


def save_results(results, out_path):
    out_path.write_text(json.dumps(results, indent=2, default=str))


def main():
    p = py_argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_CONFIG))
    args = p.parse_args()
    cfg = TASK_CONFIG[args.task]
    fixed_lang_dir, standard_lang_dir = cfg["fixed_lang_dir"], cfg["standard_lang_dir"]
    models_root, out_path = cfg["models_root"], cfg["out_path"]

    RESULTS.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    results = {
        "task": args.task,
        "experiment": "corrected-dataset-fix, ALL-OBJECTIVES expansion (per user decision after "
                     "fix_objective_methodology_audit.json's finding that rec+ns is rarely FLaRe's "
                     "own selected-best loss term)",
        "description": (
            "Trains RNN/LSTM/Transformer/Mamba on the corrected training set under ALL FOUR FLaRe "
            "loss-term combinations (rec, rec+lm, rec+ns, rec+lm+ns), 10 trials each (160 runs). "
            "Per-trial hyperparameter sampling matches train_and_evaluate.bash exactly. Evaluated on "
            "both the standard FLaRe test set and the corrected test set."
        ),
        "parameter_budget": PARAMETER_BUDGET,
        "architectures": ARCHITECTURES, "loss_term_combos": LOSS_TERM_COMBOS, "n_trials": N_TRIALS,
        "runs": [],
    }

    done_keys = set()
    # carry forward prior rec+ns-only results (marked-reversal only) rather than retraining
    prior_path = cfg["prior_rec_ns_only_results"]
    if prior_path is not None and prior_path.exists():
        prior = json.loads(prior_path.read_text())
        for r in prior.get("runs", []):
            r = dict(r)
            r["loss_terms"] = "rec+ns"
            r["carried_forward_from_prior_rec_ns_only_run"] = True
            results["runs"].append(r)
            done_keys.add((r["arch"], "rec+ns", r["trial_no"]))
        print(f"carried forward {len(done_keys)} prior rec+ns runs from {prior_path}", flush=True)

    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            for r in existing.get("runs", []):
                key = (r["arch"], r.get("loss_terms", "rec+ns"), r["trial_no"])
                if key not in done_keys:
                    results["runs"].append(r)
                    done_keys.add(key)
            print(f"resuming: {len(done_keys)} total runs already done", flush=True)
        except Exception:
            pass

    for arch in ARCHITECTURES:
        for loss_terms in LOSS_TERM_COMBOS:
            for trial_no in range(1, N_TRIALS + 1):
                if (arch, loss_terms, trial_no) in done_keys:
                    continue
                print(f"\n=== training {args.task} {arch} {loss_terms} trial {trial_no} ===", flush=True)
                model_dir, hyperparams, duration, returncode, log_tail = train_one(
                    arch, loss_terms, trial_no, fixed_lang_dir, models_root)
                run_record = {
                    "arch": arch, "loss_terms": loss_terms, "trial_no": trial_no,
                    "model_dir": str(model_dir), "hyperparameters": hyperparams,
                    "training_duration_seconds": duration, "train_returncode": returncode,
                }
                if returncode != 0:
                    run_record["train_failed"] = True
                    run_record["log_tail"] = log_tail
                    print(f"  TRAINING FAILED (returncode={returncode})", flush=True)
                else:
                    print(f"  training done in {duration:.1f}s, evaluating...", flush=True)
                    try:
                        eval_results = evaluate_model_both_test_sets(model_dir, arch, fixed_lang_dir, standard_lang_dir)
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
                save_results(results, out_path)

    print(f"\nAll runs complete for {args.task}. Saved {out_path}")


if __name__ == "__main__":
    main()
