"""Evaluates the EXISTING standard-FLaRe-trained models (data/models/
marked-reversal/<arch>/<loss_terms>/validation-short/<trial>, already fully
trained under all 4 loss-term combinations x 10 trials -- no training here)
on the CORRECTED test set, for ALL FOUR loss terms (fix_standard_baseline_
on_corrected_test.json only covered rec+ns). Pure evaluation, reuses the
eos_index=None fix from fix_train_models_all_objectives.py.

PYTHONPATH=src:analysis python analysis/fix_standard_baseline_on_corrected_test_all_objectives.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "analysis")
from fix_train_models_all_objectives import evaluate_model_both_test_sets, LOSS_TERM_COMBOS

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "analysis_outputs" / "final_results"
STANDARD_MODELS_ROOT = REPO_ROOT / "data" / "models" / "marked-reversal"
# mamba is NOT part of FLaRe's own experiment grid (confirmed absent from
# experiments/include.bash's ARCHITECTURES) -- this pilot's own mamba
# addition lives under models/ (not data/models/) and was only ever trained
# under rec+ns, seeds 0-9 (not 1-10), matching the original narrower
# fix_standard_baseline_on_corrected_test.py's STANDARD_MODELS dict.
MAMBA_MODELS_ROOT = REPO_ROOT / "models" / "marked-reversal"
FIXED_LANG_DIR = REPO_ROOT / "languages" / "marked-reversal-fixed"
STANDARD_LANG_DIR = REPO_ROOT / "languages" / "marked-reversal"
OUT_PATH = RESULTS / "fix_standard_baseline_on_corrected_test_all_objectives.json"
ARCHITECTURES = ["rnn", "lstm", "transformer", "mamba"]
N_TRIALS = 10


def main():
    results = {"task": "marked-reversal",
               "description": "Standard-FLaRe-trained models (all 4 loss terms, no retraining) "
                              "evaluated on the corrected test set.",
               "runs": []}
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
        except Exception:
            pass
    # drop any previously-recorded entries that either (a) failed due to the
    # eos_index/last_index bug or the CUDA-context poisoning it caused in
    # every subsequent call within that same process, or (b) "succeeded" but
    # are loss_terms=="rec" entries computed with the BUGGY (pre-fix)
    # last_index logic -- confirmed empirically to silently return a
    # garbage constant logit independent of the actual input for rnn, not
    # just crash for transformer. Both categories need re-evaluation.
    before = len(results["runs"])
    results["runs"] = [r for r in results["runs"] if not r.get("eval_failed") and r.get("loss_terms") != "rec"]
    if len(results["runs"]) != before:
        print(f"dropped {before - len(results['runs'])} previously-failed/corrupted entries for "
              f"re-evaluation with the fixed code", flush=True)
    done_keys = {(r["arch"], r["loss_terms"], r["trial_no"]) for r in results["runs"]}

    for arch in ARCHITECTURES:
        for loss_terms in LOSS_TERM_COMBOS:
            trials = range(0, N_TRIALS) if arch == "mamba" else range(1, N_TRIALS + 1)
            for trial in trials:
                if (arch, loss_terms, trial) in done_keys:
                    continue
                if arch == "mamba":
                    if loss_terms != "rec+ns":
                        continue  # only rec+ns was ever trained for mamba on the standard dataset
                    model_dir = MAMBA_MODELS_ROOT / arch / loss_terms / "validation-short" / str(trial)
                else:
                    model_dir = STANDARD_MODELS_ROOT / arch / loss_terms / "validation-short" / str(trial)
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
