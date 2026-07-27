"""One-off repair pass: fix_train_models_all_objectives.py's evaluate_
recognition() had an off-by-one last_index bug affecting EVERY "rec"-only
(no ns/lm head, eos_index=None) evaluation -- last_index=len(idx) is one
past the valid range when no EOS is appended, which either hard-crashes
(transformer: CUDA scatter-gather out-of-bounds assert) or silently returns
a garbage CONSTANT logit independent of the actual input (rnn: confirmed
empirically -- last_index=len(idx) returned the IDENTICAL logit for
different test sequences, vs. genuinely input-varying logits at the
correct len(idx)-1). This corrupted every "rec"-only accuracy number
computed before the fix in fix_train_models_all_objectives.py.

This script RE-EVALUATES (no retraining -- the actual trained checkpoints
are fine, only the lightweight in-process evaluation was buggy) every
recorded "rec"-only run in the training-results JSONs, overwriting just the
eval fields in place.

PYTHONPATH=src:analysis python analysis/fix_repair_rec_only_evaluations.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "analysis")
from fix_train_models_all_objectives import evaluate_model_both_test_sets, TASK_CONFIG

RESULTS = Path("analysis_outputs/final_results")
FILES_TO_REPAIR = [
    ("binary-addition", RESULTS / "fix_trained_models_binary_addition_all_objectives.json"),
    ("marked-reversal", RESULTS / "fix_trained_models_marked_reversal_all_objectives.json"),
]


def main():
    for task, path in FILES_TO_REPAIR:
        if not path.exists():
            print(f"skip {path} (does not exist)")
            continue
        cfg = TASK_CONFIG[task]
        d = json.loads(path.read_text())
        n_repaired, n_skipped_no_model = 0, 0
        for r in d["runs"]:
            if r.get("loss_terms") != "rec":
                continue
            if r.get("train_failed"):
                continue
            model_dir = Path(r["model_dir"])
            if not model_dir.exists():
                n_skipped_no_model += 1
                print(f"  SKIP (no model dir): {task} {r['arch']} rec trial{r['trial_no']}")
                continue
            eval_results = evaluate_model_both_test_sets(model_dir, r["arch"], cfg["fixed_lang_dir"], cfg["standard_lang_dir"])
            old_std = r.get("standard_flare_test_set", {}).get("accuracy")
            r.pop("eval_failed", None)
            r.pop("eval_error", None)
            r.update(eval_results)
            new_std = eval_results["standard_flare_test_set"]["accuracy"]
            print(f"  repaired: {task} {r['arch']} rec trial{r['trial_no']}: "
                  f"std_acc {old_std} -> {new_std:.4f}")
            n_repaired += 1
        path.write_text(json.dumps(d, indent=2, default=str))
        print(f"{task}: repaired {n_repaired} 'rec' entries, {n_skipped_no_model} skipped (no model dir yet)\n")


if __name__ == "__main__":
    main()
