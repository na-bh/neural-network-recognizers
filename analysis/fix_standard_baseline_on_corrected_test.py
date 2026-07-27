"""Corrected-dataset experiment, F4 addition: evaluate the pilot's EXISTING
standard-FLaRe-trained marked-reversal checkpoints (ALL available seeds per
architecture, not just the ones used in the pilot's own three-phase
analysis) on the corrected test set, and assemble the three-way comparison
table (standard-trained-on-corrected-test / corrected-trained-on-corrected-
test / corrected-trained-on-standard-test).

METHODOLOGICAL CAVEAT (reported explicitly, not glossed over): the pilot's
standard-trained checkpoints were TRAINED under Butoi et al.'s original
random-hyperparameter-per-trial procedure same as F3's corrected-trained
checkpoints -- but the specific seeds this pilot SELECTED for its own
probing/patching work (Phase 2/3 of the four-task pilot) were chosen as the
best-performing seeds per architecture (see e.g. phase2_p1_targets.py-style
selected_seeds). Evaluating ALL 10 standard-trained seeds here (not just the
pilot's selected best ones) means this comparison mixes both strong and
weak standard-trained seeds, whereas F3's 10 corrected-trained trials are
UNSELECTED random hyperparameter draws by construction (no cherry-picking
occurred in either direction for F3). This asymmetry -- pilot-selected-best
vs. F3-random-draw -- is a real difference in how the two seed sets were
assembled and is reported here rather than treated as an apples-to-apples
comparison.

PYTHONPATH=src:analysis python analysis/fix_standard_baseline_on_corrected_test.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "analysis")
from fix_train_models_f3 import evaluate_model_both_test_sets

RESULTS = Path("analysis_outputs/final_results")
STANDARD_MODELS = {
    "rnn": (Path("data/models/marked-reversal/rnn/rec+ns/validation-short"), list(range(1, 11))),
    "lstm": (Path("data/models/marked-reversal/lstm/rec+ns/validation-short"), list(range(1, 11))),
    "transformer": (Path("data/models/marked-reversal/transformer/rec+ns/validation-short"), list(range(1, 11))),
    "mamba": (Path("models/marked-reversal/mamba/rec+ns/validation-short"), list(range(0, 10))),
}
OUT_PATH = RESULTS / "fix_standard_baseline_on_corrected_test.json"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    results = {"runs": []}
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text())
            if existing.get("runs"):
                results["runs"] = existing["runs"]
                print(f"resuming: {len(results['runs'])} already done", flush=True)
        except Exception:
            pass
    done_keys = {(r["arch"], r["seed"]) for r in results["runs"]}

    for arch, (base_dir, seeds) in STANDARD_MODELS.items():
        for seed in seeds:
            if (arch, seed) in done_keys:
                print(f"skipping {arch} seed {seed} (already done)", flush=True)
                continue
            model_dir = base_dir / str(seed)
            print(f"\n=== evaluating {arch} seed {seed} (standard-trained checkpoint) ===", flush=True)
            try:
                eval_results = evaluate_model_both_test_sets(model_dir, arch)
                run_record = {"arch": arch, "seed": seed, "model_dir": str(model_dir), **eval_results}
                std_acc = eval_results["standard_flare_test_set"]["accuracy"]
                fx = eval_results["corrected_test_set"]
                print(f"  standard_test_acc={std_acc:.4f}  corrected_overall={fx['overall_accuracy']:.4f} "
                      f"pos={fx['positive_accuracy']:.4f} uniform={fx['uniform_random_half_accuracy']:.4f} "
                      f"hard={fx['hard_negative_half_accuracy']:.4f}", flush=True)
            except Exception as e:
                run_record = {"arch": arch, "seed": seed, "model_dir": str(model_dir),
                             "eval_failed": True, "eval_error": repr(e)}
                print(f"  EVAL FAILED: {e!r}", flush=True)
            results["runs"].append(run_record)
            OUT_PATH.write_text(json.dumps(results, indent=2, default=str))

    # ------------------------------------------------------------------
    # hard-negative-half accuracy summary vs. chance (0.5), per architecture
    # ------------------------------------------------------------------
    import numpy as np
    per_arch_hard = {}
    for arch in STANDARD_MODELS:
        arch_runs = [r for r in results["runs"] if r["arch"] == arch and "corrected_test_set" in r]
        hard_accs = [r["corrected_test_set"]["hard_negative_half_accuracy"] for r in arch_runs]
        per_arch_hard[arch] = {
            "n_seeds": len(hard_accs),
            "hard_negative_accuracy_per_seed": {r["seed"]: r["corrected_test_set"]["hard_negative_half_accuracy"] for r in arch_runs},
            "min": float(np.min(hard_accs)) if hard_accs else None,
            "max": float(np.max(hard_accs)) if hard_accs else None,
            "mean_reported_for_context_only_not_the_primary_metric": float(np.mean(hard_accs)) if hard_accs else None,
        }
    print("\n=== hard-negative-half accuracy by architecture (standard-trained checkpoints) ===")
    print(json.dumps(per_arch_hard, indent=2, default=str))

    results["methodological_caveat"] = (
        "Standard-trained checkpoints evaluated here are ALL 10 available seeds per architecture "
        "(unselected), whereas the pilot's own Phase 2/3 probing/patching work used only the BEST-"
        "performing 1-2 seeds per architecture (selected_seeds). F3's corrected-trained checkpoints "
        "are 10 RANDOM hyperparameter draws per architecture with no selection in either direction. "
        "This means: (a) this evaluation includes standard-trained seeds that never solved the "
        "original task well at all (chance-level long-sequence accuracy), unlike the pilot's own "
        "curated cell roster; (b) the F3 vs. standard-trained comparison is not a clean apples-to-"
        "apples 'same seed-selection procedure, different training data' comparison -- it compares "
        "an unselected population (standard) to another unselected population (F3), which IS "
        "internally consistent, but neither matches the pilot's PREVIOUS best-seed-selected "
        "characterization of marked-reversal's mechanisms. Read the standard-trained numbers here "
        "as 'what the full population of standard training runs does on the corrected test,' not "
        "as 'what the pilot's previously-characterized best seeds do.'"
    )
    results["hard_negative_accuracy_by_architecture"] = per_arch_hard
    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
