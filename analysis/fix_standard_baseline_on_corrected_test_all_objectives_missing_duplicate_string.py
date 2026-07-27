"""Evaluates the EXISTING standard-FLaRe-trained missing-duplicate-string
models (data/models/missing-duplicate-string/<arch>/<loss_terms>/
validation-short/<trial> for rnn/lstm/transformer, models/missing-
duplicate-string/mamba/<loss_terms>/validation-short/<trial> for mamba) on
the CORRECTED test set. Pure evaluation, no training.

Mamba checkpoints use seeds 0-9; rnn/lstm/transformer use seeds 1-10
(confirmed directly against the checkpoint directories -- NOT a uniform
1-10 range for every architecture).

PYTHONPATH=src:analysis python analysis/fix_standard_baseline_on_corrected_test_all_objectives_missing_duplicate_string.py
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, "analysis")
from fix_train_models_all_objectives import evaluate_model_both_test_sets, LOSS_TERM_COMBOS

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "analysis_outputs" / "final_results"
STANDARD_MODELS_ROOT = REPO_ROOT / "data" / "models" / "missing-duplicate-string"
MAMBA_MODELS_ROOT = REPO_ROOT / "models" / "missing-duplicate-string"
FIXED_LANG_DIR = REPO_ROOT / "languages" / "missing-duplicate-string-fixed"
STANDARD_LANG_DIR = REPO_ROOT / "languages" / "missing-duplicate-string"
OUT_PATH = RESULTS / "fix_standard_baseline_on_corrected_test_all_objectives_missing_duplicate_string.json"
ARCHITECTURES = ["rnn", "lstm", "transformer", "mamba"]
N_TRIALS = 10


def main():
    results = {"task": "missing-duplicate-string",
               "description": "Standard-FLaRe-trained models (all 4 loss terms, no retraining) "
                              "evaluated on the corrected test set.",
               "runs": []}
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
        except Exception:
            pass
    done_keys = {(r["arch"], r["loss_terms"], r["trial_no"]) for r in results["runs"] if not r.get("eval_failed")}

    for arch in ARCHITECTURES:
        seeds = range(0, N_TRIALS) if arch == "mamba" else range(1, N_TRIALS + 1)
        for loss_terms in LOSS_TERM_COMBOS:
            for trial in seeds:
                if (arch, loss_terms, trial) in done_keys:
                    continue
                base = MAMBA_MODELS_ROOT if arch == "mamba" else STANDARD_MODELS_ROOT
                model_dir = base / arch / loss_terms / "validation-short" / str(trial)
                if not model_dir.exists():
                    continue
                try:
                    eval_results = evaluate_model_both_test_sets(model_dir, arch, FIXED_LANG_DIR, STANDARD_LANG_DIR)
                    run_record = {"arch": arch, "loss_terms": loss_terms, "trial_no": trial,
                                  "model_dir": str(model_dir), **eval_results}
                    fx = eval_results["corrected_test_set"]
                    print(f"{arch:12s} {loss_terms:10s} trial{trial}: "
                          f"corrected_overall={fx['overall_accuracy']:.4f} pos={fx['positive_accuracy']:.4f} "
                          f"uniform={fx['uniform_random_half_accuracy']:.4f} hard={fx['hard_negative_half_accuracy']:.4f}",
                          flush=True)
                except Exception as e:
                    run_record = {"arch": arch, "loss_terms": loss_terms, "trial_no": trial,
                                  "model_dir": str(model_dir), "eval_failed": True, "eval_error": repr(e)}
                    print(f"{arch} {loss_terms} trial{trial}: EVAL FAILED {e!r}", flush=True)
                results["runs"].append(run_record)
                OUT_PATH.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
