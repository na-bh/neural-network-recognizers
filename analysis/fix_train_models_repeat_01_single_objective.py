"""Corrected-dataset experiment for repeat-01, RF3: trains RNN/LSTM/
Transformer/Mamba x 10 seeds (40 runs) under ONE specified loss-term
objective on the corrected training set (languages/repeat-01-fixed),
matching FLaRe's per-trial hyperparameter sampling exactly (reuses
fix_train_models_all_objectives.py's train_one/evaluate_model_both_test_
sets directly -- no new training-loop logic here). Each run is evaluated
on BOTH the standard FLaRe test set and the corrected test set.

Standard-FLaRe-trained checkpoint evaluation on the corrected test set is a
SEPARATE deliverable (fix_standard_baseline_on_corrected_test_all_
objectives_repeat_01.py), not embedded here.

PYTHONPATH=src:analysis python analysis/fix_train_models_repeat_01_single_objective.py --objective rec
PYTHONPATH=src:analysis python analysis/fix_train_models_repeat_01_single_objective.py --objective rec+lm
PYTHONPATH=src:analysis python analysis/fix_train_models_repeat_01_single_objective.py --objective rec+ns
PYTHONPATH=src:analysis python analysis/fix_train_models_repeat_01_single_objective.py --objective rec+lm+ns
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, "analysis")
from fix_train_models_all_objectives import (
    train_one, evaluate_model_both_test_sets, ARCHITECTURES, N_TRIALS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "analysis_outputs" / "final_results"
FIXED_LANG_DIR = REPO_ROOT / "languages" / "repeat-01-fixed"
STANDARD_LANG_DIR = REPO_ROOT / "languages" / "repeat-01"
MODELS_ROOT = REPO_ROOT / "models" / "repeat-01-fixed"
TASK = "repeat-01"


def objective_slug(objective):
    return objective.replace("+", "_")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, choices=["rec", "rec+lm", "rec+ns", "rec+lm+ns"])
    args = p.parse_args()
    objective = args.objective
    slug = objective_slug(objective)
    out_path = RESULTS / f"fix_trained_models_repeat_01_{slug}.json"

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    results = {
        "task": TASK, "objective": objective,
        "experiment": "corrected-dataset-fix (RF3: single-objective training on corrected data)",
        "description": (
            f"Trains RNN/LSTM/Transformer/Mamba on the corrected repeat-01 training set "
            f"(languages/repeat-01-fixed) under the '{objective}' objective, 10 trials each (40 "
            f"runs). Per-trial hyperparameter sampling matches train_and_evaluate.bash exactly. "
            f"Evaluated on both the standard FLaRe test set and the corrected test set."
        ),
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
            print(f"\n=== training repeat-01 (corrected) {arch} {objective} trial {trial_no} ===", flush=True)
            model_dir, hyperparams, duration, returncode, log_tail = train_one(
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
                except Exception as e:
                    run_record["eval_failed"] = True
                    run_record["eval_error"] = repr(e)
                    print(f"  EVAL FAILED: {e!r}", flush=True)
            results["corrected_trained_runs"].append(run_record)
            out_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nAll runs complete for objective={objective}. Saved {out_path}")


if __name__ == "__main__":
    main()
