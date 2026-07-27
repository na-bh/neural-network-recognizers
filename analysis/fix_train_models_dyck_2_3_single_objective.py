"""Corrected-dataset experiment for Dyck-2-3, DF3: trains RNN/LSTM/
Transformer/Mamba x 10 seeds (40 runs) under ONE specified loss-term
objective on the corrected training set (languages/dyck-2-3-fixed),
matching FLaRe's per-trial hyperparameter sampling exactly (reuses
fix_train_models_all_objectives.py's train_one/evaluate_model_both_test_
sets directly -- no new training-loop logic here).

Also evaluates the EXISTING standard-FLaRe-trained checkpoints for the SAME
objective (data/models/dyck-2-3/<arch>/<objective>/validation-short/
<1..10>) on the corrected test set -- no retraining, pure evaluation,
included in the same output file for direct comparison.

Run ONE objective at a time (per explicit user instruction: stop after each
objective so unexpected results can be investigated before the next runs).

PYTHONPATH=src:analysis python analysis/fix_train_models_dyck_2_3_single_objective.py --objective rec
PYTHONPATH=src:analysis python analysis/fix_train_models_dyck_2_3_single_objective.py --objective rec+lm
PYTHONPATH=src:analysis python analysis/fix_train_models_dyck_2_3_single_objective.py --objective rec+ns
PYTHONPATH=src:analysis python analysis/fix_train_models_dyck_2_3_single_objective.py --objective rec+lm+ns
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
FIXED_LANG_DIR = REPO_ROOT / "languages" / "dyck-2-3-fixed"
STANDARD_LANG_DIR = REPO_ROOT / "languages" / "dyck-2-3"
STANDARD_MODELS_ROOT = REPO_ROOT / "data" / "models" / "dyck-2-3"
MAMBA_STANDARD_MODELS_ROOT = REPO_ROOT / "models" / "dyck-2-3"
MODELS_ROOT = REPO_ROOT / "models" / "dyck-2-3-fixed"
TASK = "dyck-2-3"


def objective_slug(objective):
    return objective.replace("+", "_")


def evaluate_standard_checkpoint(arch, seed, objective):
    if arch == "mamba":
        model_dir = MAMBA_STANDARD_MODELS_ROOT / arch / objective / "validation-short" / str(seed)
    else:
        model_dir = STANDARD_MODELS_ROOT / arch / objective / "validation-short" / str(seed)
    if not model_dir.exists():
        return None
    eval_results = evaluate_model_both_test_sets(model_dir, arch, FIXED_LANG_DIR, STANDARD_LANG_DIR)
    return {"arch": arch, "seed": seed, "objective": objective, "model_dir": str(model_dir), **eval_results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, choices=["rec", "rec+lm", "rec+ns", "rec+lm+ns"])
    args = p.parse_args()
    objective = args.objective
    slug = objective_slug(objective)
    out_path = RESULTS / f"fix_trained_models_dyck_2_3_{slug}.json"

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    results = {
        "task": TASK, "objective": objective,
        "experiment": "corrected-dataset-fix (DF3: single-objective training on corrected data)",
        "description": (
            f"Trains RNN/LSTM/Transformer/Mamba on the corrected Dyck-2-3 training set "
            f"(languages/dyck-2-3-fixed) under the '{objective}' objective, 10 trials each (40 "
            f"runs). Per-trial hyperparameter sampling matches train_and_evaluate.bash exactly. "
            f"Evaluated on both the standard FLaRe test set and the corrected test set. Also "
            f"includes the EXISTING standard-FLaRe-trained checkpoints for this same objective, "
            f"evaluated on the corrected test set (no retraining), for direct comparison."
        ),
        "corrected_trained_runs": [], "standard_trained_on_corrected_test": [],
    }
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            results["corrected_trained_runs"] = existing.get("corrected_trained_runs", [])
            results["standard_trained_on_corrected_test"] = existing.get("standard_trained_on_corrected_test", [])
            print(f"resuming: {len(results['corrected_trained_runs'])} corrected-trained runs, "
                  f"{len(results['standard_trained_on_corrected_test'])} standard-eval runs already done", flush=True)
        except Exception:
            pass

    done_keys = {(r["arch"], r["trial_no"]) for r in results["corrected_trained_runs"]}
    for arch in ARCHITECTURES:
        for trial_no in range(1, N_TRIALS + 1):
            if (arch, trial_no) in done_keys:
                continue
            print(f"\n=== training dyck-2-3 (corrected) {arch} {objective} trial {trial_no} ===", flush=True)
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

    # ------------------------------------------------------------------
    # standard-FLaRe-trained checkpoints (same objective), evaluated on
    # corrected test set -- no retraining
    # ------------------------------------------------------------------
    std_done_keys = {(r["arch"], r["seed"]) for r in results["standard_trained_on_corrected_test"] if r is not None}
    for arch in ARCHITECTURES:
        seeds = range(0, N_TRIALS) if arch == "mamba" else range(1, N_TRIALS + 1)
        for seed in seeds:
            if (arch, seed) in std_done_keys:
                continue
            print(f"\n=== evaluating standard-trained {arch} {objective} seed{seed} on corrected test ===", flush=True)
            try:
                r = evaluate_standard_checkpoint(arch, seed, objective)
            except Exception as e:
                r = {"arch": arch, "seed": seed, "objective": objective, "eval_failed": True, "eval_error": repr(e)}
                print(f"  EVAL FAILED: {e!r}", flush=True)
            if r is not None:
                if "corrected_test_set" in r:
                    fx = r["corrected_test_set"]
                    print(f"  corrected_overall={fx['overall_accuracy']:.4f} pos={fx['positive_accuracy']:.4f} "
                          f"hard={fx['hard_negative_half_accuracy']:.4f}", flush=True)
                results["standard_trained_on_corrected_test"].append(r)
                out_path.write_text(json.dumps(results, indent=2, default=str))
            else:
                print("  no standard checkpoint at this path, skipping", flush=True)

    print(f"\nAll runs complete for objective={objective}. Saved {out_path}")


if __name__ == "__main__":
    main()
