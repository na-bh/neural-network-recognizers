"""Contrastive-pairing experiment, CPF3: trains RNN/LSTM/Transformer/Mamba
x 10 seeds (40 runs) under ONE specified loss-term objective on the PAIRED
marked-reversal training set (languages/marked-reversal-fixed-contrastive),
matching FLaRe's per-trial hyperparameter sampling exactly, with ONE
addition: --pair-id-file is passed to train.py so batches respect pair
boundaries (see src/recognizers/neural_networks/contrastive_training_loop.py).
Each run is evaluated on BOTH the standard FLaRe test set and the
(unmodified) corrected marked-reversal test set, exactly like every other
fix experiment -- plus a CONTRASTIVE-SPECIFIC per-pair accuracy metric on
the test set (see evaluate_pair_accuracy below): for every hard negative in
the test set, its exact source positive is reconstructed via swap
inversion (same technique as CPF1), and we report how often the model gets
BOTH members of the pair correct.

Per-run outputs also copy over contrastive_batch_stats.json (written by
train.py into the model directory) into the results JSON, so overflow-vs-
outcome correlation can be checked post-hoc without re-opening 160
individual model directories.

PYTHONPATH=src:analysis python analysis/fix_train_models_marked_reversal_contrastive_single_objective.py --objective rec
PYTHONPATH=src:analysis python analysis/fix_train_models_marked_reversal_contrastive_single_objective.py --objective rec+lm
PYTHONPATH=src:analysis python analysis/fix_train_models_marked_reversal_contrastive_single_objective.py --objective rec+ns
PYTHONPATH=src:analysis python analysis/fix_train_models_marked_reversal_contrastive_single_objective.py --objective rec+lm+ns
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SUBPROCESS_ENV = os.environ.copy()
SUBPROCESS_ENV["PYTHONPATH"] = "."

sys.path.insert(0, "analysis")
from fix_train_models_all_objectives import (
    random_sample, get_architecture_args, evaluate_model_both_test_sets,
    load_model, load_split, evaluate_recognition, subset_acc,
    ARCHITECTURES, N_TRIALS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS = REPO_ROOT / "analysis_outputs" / "final_results"
FIXED_LANG_DIR = REPO_ROOT / "languages" / "marked-reversal-fixed-contrastive"
STANDARD_LANG_DIR = REPO_ROOT / "languages" / "marked-reversal"
MODELS_ROOT = REPO_ROOT / "models" / "marked-reversal-fixed-contrastive"
PAIR_ID_FILE = FIXED_LANG_DIR / "pair-id.txt"
TASK = "marked-reversal"
MARKER = "#"


def objective_slug(objective):
    return objective.replace("+", "_")


def train_one_contrastive(arch, loss_terms, trial_no, fixed_lang_dir, models_root):
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
        "--pair-id-file", str(PAIR_ID_FILE),
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
# pair-preserved test-set evaluation (contrastive-specific metric)
# ---------------------------------------------------------------------------

def reconstruct_source_positive(seq, meta):
    m = seq.index(MARKER)
    before, after = seq[:m], seq[m + 1:]
    i, j = meta["swap_i"], meta["swap_j"]
    after_orig = list(after)
    after_orig[i], after_orig[j] = after_orig[j], after_orig[i]
    return before + [MARKER] + after_orig


def is_positive_local(seq):
    if MARKER not in seq:
        return False
    m = seq.index(MARKER)
    before, after = seq[:m], seq[m + 1:]
    return seq.count(MARKER) == 1 and len(before) == len(after) and after == list(reversed(before))


def build_test_pairs(test_dir):
    """Reconstructs, for every hard negative in the (UNMODIFIED) test set,
    its exact source positive via swap inversion -- same technique as CPF1,
    applied here purely for analysis (the on-disk test set is never
    touched). Returns a list of (positive_tokens, hard_negative_tokens)
    pairs."""
    seqs, labels = load_split(test_dir)
    kinds = (test_dir / "negative-kind.txt").read_text().splitlines()
    metas = [json.loads(line) for line in (test_dir / "hard-negative-swap-meta.jsonl").read_text().splitlines()]
    pairs = []
    n_failed = 0
    for seq, label, kind, meta in zip(seqs, labels, kinds, metas):
        if label == 0 and kind == "hard_negative":
            recon = reconstruct_source_positive(seq, meta)
            if is_positive_local(recon):
                pairs.append((recon, seq))
            else:
                n_failed += 1
    return pairs, n_failed


def evaluate_pair_accuracy(iface, saver, pairs, vocab_tokens):
    pos_seqs = [p[0] for p in pairs]
    hard_seqs = [p[1] for p in pairs]
    pos_correct, pos_failed = evaluate_recognition(iface, saver, pos_seqs, [1] * len(pos_seqs), vocab_tokens)
    hard_correct, hard_failed = evaluate_recognition(iface, saver, hard_seqs, [0] * len(hard_seqs), vocab_tokens)
    n = len(pairs)
    both_correct = sum(
        1 for p, h in zip(pos_correct, hard_correct)
        if p is True and h is True
    )
    pos_only = sum(1 for p, h in zip(pos_correct, hard_correct) if p is True and h is not True)
    hard_only = sum(1 for p, h in zip(pos_correct, hard_correct) if p is not True and h is True)
    neither = sum(1 for p, h in zip(pos_correct, hard_correct) if p is not True and h is not True)
    return {
        "n_pairs": n,
        "both_correct": both_correct,
        "positive_only_correct": pos_only,
        "hard_negative_only_correct": hard_only,
        "neither_correct": neither,
        "both_correct_fraction": both_correct / n if n else float("nan"),
        "positive_accuracy": (both_correct + pos_only) / n if n else float("nan"),
        "hard_negative_accuracy": (both_correct + hard_only) / n if n else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, choices=["rec", "rec+lm", "rec+ns", "rec+lm+ns"])
    args = p.parse_args()
    objective = args.objective
    slug = objective_slug(objective)
    out_path = RESULTS / f"fix_trained_models_marked_reversal_contrastive_{slug}.json"

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    test_dir = FIXED_LANG_DIR / "datasets" / "test"
    test_pairs, n_pair_recon_failed = build_test_pairs(test_dir)
    print(f"test-set pair reconstruction: {len(test_pairs)} pairs "
          f"({n_pair_recon_failed} reconstruction failures)", flush=True)
    assert n_pair_recon_failed == 0

    results = {
        "task": TASK, "objective": objective,
        "experiment": "contrastive-pairing (CPF3: single-objective training on paired corrected data)",
        "description": (
            f"Trains RNN/LSTM/Transformer/Mamba on the PAIRED marked-reversal training set "
            f"(languages/marked-reversal-fixed-contrastive) under the '{objective}' objective, "
            f"10 trials each (40 runs), with --pair-id-file passed to train.py so batches "
            f"respect pair boundaries. Evaluated on the standard FLaRe test set, the corrected "
            f"test set, and pair-preserved accuracy on the {len(test_pairs)} reconstructed "
            f"test-set pairs."
        ),
        "n_test_pairs": len(test_pairs),
        "corrected_trained_runs": [],
    }
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            results["corrected_trained_runs"] = existing.get("corrected_trained_runs", [])
            print(f"resuming: {len(results['corrected_trained_runs'])} corrected-trained runs already done", flush=True)
        except Exception:
            pass

    done_keys = {(r["arch"], r["trial_no"]) for r in results["corrected_trained_runs"]}
    for arch in ARCHITECTURES:
        for trial_no in range(1, N_TRIALS + 1):
            if (arch, trial_no) in done_keys:
                continue
            print(f"\n=== training marked-reversal (contrastive) {arch} {objective} trial {trial_no} ===", flush=True)
            model_dir, hyperparams, duration, returncode, log_tail = train_one_contrastive(
                arch, objective, trial_no, FIXED_LANG_DIR, MODELS_ROOT)
            run_record = {
                "arch": arch, "objective": objective, "trial_no": trial_no,
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
                    eval_results = evaluate_model_both_test_sets(model_dir, arch, FIXED_LANG_DIR, STANDARD_LANG_DIR)
                    run_record.update(eval_results)
                    std_acc = eval_results["standard_flare_test_set"]["accuracy"]
                    fx = eval_results["corrected_test_set"]
                    print(f"  standard_test_acc={std_acc:.4f}  corrected_overall={fx['overall_accuracy']:.4f} "
                          f"pos={fx['positive_accuracy']:.4f} uniform={fx['uniform_random_half_accuracy']:.4f} "
                          f"hard={fx['hard_negative_half_accuracy']:.4f}", flush=True)

                    vocab_tokens_path = FIXED_LANG_DIR / "main.vocab"
                    import torch
                    vocab_tokens = torch.load(vocab_tokens_path, weights_only=False)["tokens"]
                    iface, saver = load_model(model_dir, FIXED_LANG_DIR, arch)
                    pair_acc = evaluate_pair_accuracy(iface, saver, test_pairs, vocab_tokens)
                    run_record["pair_preserved_test_accuracy"] = pair_acc
                    print(f"  pair-preserved: both_correct={pair_acc['both_correct_fraction']:.4f} "
                          f"pos_only={pair_acc['positive_only_correct']} hard_only={pair_acc['hard_negative_only_correct']} "
                          f"neither={pair_acc['neither_correct']}", flush=True)

                    batch_stats_path = Path(model_dir) / "contrastive_batch_stats.json"
                    if batch_stats_path.exists():
                        run_record["contrastive_batch_stats"] = json.loads(batch_stats_path.read_text())
                except Exception as e:
                    run_record["eval_failed"] = True
                    run_record["eval_error"] = repr(e)
                    print(f"  EVAL FAILED: {e!r}", flush=True)
            results["corrected_trained_runs"].append(run_record)
            out_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nAll runs complete for objective={objective}. Saved {out_path}")


if __name__ == "__main__":
    main()
