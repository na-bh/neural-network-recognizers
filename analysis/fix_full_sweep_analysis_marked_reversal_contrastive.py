"""Contrastive-pairing experiment, CPF4: full-sweep decomposition analysis.

Aggregates the 160 contrastive-trained cells (10 seeds x 4 architectures x
4 objectives, from fix_trained_models_marked_reversal_contrastive_
[objective].json) using the SAME balanced-discrimination criteria as every
prior fix experiment (balanced_any, balanced_strong, exceeds_ceiling_
balanced against the theoretical shortcut-only ceiling from
fix_baseline_ceiling_marked_reversal.json), PLUS the contrastive-specific
per-pair test accuracy metric (already computed per-run by CPF3's
evaluate_pair_accuracy: both_correct / positive_only / hard_negative_only /
neither, on the 1261 reconstructed test-set pairs).

Unlike prior full-sweep scripts, this does NOT re-load each model to
recompute per-example correctness from scratch -- CPF3 already saved
per-run overall/positive/hard-negative accuracy on the corrected test set
(via evaluate_model_both_test_sets) and the pair-preserved metric, so this
script is pure aggregation over already-computed data.

No NEW standard-baseline population is computed here: the comparison point
is the ALREADY-EXISTING marked-reversal fix experiment's results
(fix_full_sweep_marked_reversal_analysis.json), which found 0/160
corrected-trained + 0/130 standard-trained balanced_any cells (0/290
combined) -- explicitly the baseline this contrastive-pairing experiment is
testing against.

PYTHONPATH=src:analysis python analysis/fix_full_sweep_analysis_marked_reversal_contrastive.py
"""

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "analysis_outputs" / "final_results"
TASK = "marked-reversal"
OBJECTIVES = ["rec", "rec+lm", "rec+ns", "rec+lm+ns"]
ARCHITECTURES = ["rnn", "lstm", "transformer", "mamba"]
BASELINE_FULL_SWEEP_PATH = RESULTS / "fix_full_sweep_marked_reversal_analysis.json"


def find_ceiling(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "theoretical_shortcut_only_ceiling" and isinstance(v, (int, float)):
                return v
            r = find_ceiling(v)
            if r is not None:
                return r
    return None


def make_cell_record(arch, objective, seed, corrected_test_set, pair_acc, theoretical_ceiling):
    overall = corrected_test_set["overall_accuracy"]
    positive = corrected_test_set["positive_accuracy"]
    hard = corrected_test_set["hard_negative_half_accuracy"]
    balanced_any = positive > 0.5 and hard > 0.5
    balanced_strong = positive > 0.7 and hard > 0.7
    exceeds_ceiling_balanced = overall > theoretical_ceiling and balanced_any
    return {
        "arch": arch, "objective": objective, "seed": seed,
        "overall_accuracy": overall, "positive_accuracy": positive, "hard_negative_accuracy": hard,
        "uniform_random_accuracy": corrected_test_set.get("uniform_random_half_accuracy"),
        "balanced_any": balanced_any, "balanced_strong": balanced_strong,
        "exceeds_ceiling_balanced": exceeds_ceiling_balanced,
        "pair_preserved_test_accuracy": pair_acc,
    }


def aggregate_by_arch_objective(cells):
    keys = [f"{a}_{o}" for a in ARCHITECTURES for o in OBJECTIVES]
    out = {}
    for key in keys:
        arch, objective = key.split("_", 1)
        subset = [c for c in cells if c["arch"] == arch and c["objective"] == objective]
        out[key] = {
            "n": len(subset),
            "n_balanced_any": sum(1 for c in subset if c["balanced_any"]),
            "n_balanced_strong": sum(1 for c in subset if c["balanced_strong"]),
            "n_exceeds_ceiling_balanced": sum(1 for c in subset if c["exceeds_ceiling_balanced"]),
            "mean_overall_accuracy": float(np.mean([c["overall_accuracy"] for c in subset])) if subset else None,
            "mean_hard_negative_accuracy": float(np.mean([c["hard_negative_accuracy"] for c in subset])) if subset else None,
            "mean_pair_both_correct_fraction": (
                float(np.mean([c["pair_preserved_test_accuracy"]["both_correct_fraction"] for c in subset
                               if c["pair_preserved_test_accuracy"] is not None]))
                if any(c["pair_preserved_test_accuracy"] is not None for c in subset) else None
            ),
        }
    return out


def winning_objective_per_architecture(cells):
    out = {}
    for arch in ARCHITECTURES:
        best_obj, best_score = None, -1.0
        per_obj = {}
        for objective in OBJECTIVES:
            subset = [c for c in cells if c["arch"] == arch and c["objective"] == objective]
            n_balanced_any = sum(1 for c in subset if c["balanced_any"])
            mean_hard = float(np.mean([c["hard_negative_accuracy"] for c in subset])) if subset else 0.0
            per_obj[objective] = {"n_balanced_any": n_balanced_any, "mean_hard_negative_accuracy": mean_hard}
            score = n_balanced_any + mean_hard
            if score > best_score:
                best_score, best_obj = score, objective
        out[arch] = {"winning_objective": best_obj, "per_objective": per_obj}
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    ceiling_data = json.loads((RESULTS / "fix_baseline_ceiling_marked_reversal.json").read_text())
    theoretical_ceiling = find_ceiling(ceiling_data)
    print(f"theoretical shortcut-only ceiling: {theoretical_ceiling:.4f}", flush=True)

    cells = []
    n_failed_runs = 0
    for objective in OBJECTIVES:
        run_file = RESULTS / f"fix_trained_models_marked_reversal_contrastive_{objective.replace('+', '_')}.json"
        if not run_file.exists():
            print(f"  {run_file} not found yet, skipping objective={objective}", flush=True)
            continue
        d = json.loads(run_file.read_text())
        for r in d["corrected_trained_runs"]:
            if r.get("train_failed") or r.get("eval_failed"):
                n_failed_runs += 1
                continue
            cell = make_cell_record(
                r["arch"], objective, r["trial_no"],
                r["corrected_test_set"], r.get("pair_preserved_test_accuracy"),
                theoretical_ceiling,
            )
            cells.append(cell)

    print(f"assembled {len(cells)} cells ({n_failed_runs} failed runs excluded)", flush=True)

    # ------------------------------------------------------------------
    # compare against the EXISTING marked-reversal fix experiment
    # ------------------------------------------------------------------
    baseline_comparison = None
    if BASELINE_FULL_SWEEP_PATH.exists():
        b = json.loads(BASELINE_FULL_SWEEP_PATH.read_text())
        b_corr = b["corrected_trained"]
        b_std = b["standard_trained"]
        n_bal_corr = sum(1 for c in b_corr["cells"] if c["balanced_any"])
        n_bal_std = sum(1 for c in b_std["cells"] if c["balanced_any"])
        baseline_comparison = {
            "source": str(BASELINE_FULL_SWEEP_PATH),
            "corrected_trained_balanced_any": f"{n_bal_corr}/{b_corr['n_cells']}",
            "standard_trained_balanced_any": f"{n_bal_std}/{b_std['n_cells']}",
            "combined_balanced_any": f"{n_bal_corr + n_bal_std}/{b_corr['n_cells'] + b_std['n_cells']}",
        }
        print(f"baseline (existing, non-contrastive, marked-reversal fix experiment): "
              f"{baseline_comparison['combined_balanced_any']} balanced_any combined", flush=True)

    n_bal_any = sum(1 for c in cells if c["balanced_any"])
    n_bal_strong = sum(1 for c in cells if c["balanced_strong"])
    n_exceed = sum(1 for c in cells if c["exceeds_ceiling_balanced"])
    pair_both_correct_fracs = [
        c["pair_preserved_test_accuracy"]["both_correct_fraction"] for c in cells
        if c["pair_preserved_test_accuracy"] is not None
    ]

    out = {
        "task": TASK,
        "experiment": "contrastive-pairing (CPF4: full-sweep decomposition analysis)",
        "description": (
            "Full-sweep decomposition of the 160 contrastive-trained cells (paired batching "
            "via --pair-id-file), using the same balanced-discrimination criteria as every "
            "prior fix experiment, plus per-pair test accuracy (does the model get BOTH the "
            "positive and its constructed-from hard negative correct?) -- compared against the "
            "existing (non-contrastive) marked-reversal fix experiment's 0/290 balanced_any "
            "baseline."
        ),
        "theoretical_shortcut_only_ceiling": theoretical_ceiling,
        "criteria": {
            "a_balanced_any": "positive_accuracy > 0.5 AND hard_negative_accuracy > 0.5",
            "b_balanced_strong": "positive_accuracy > 0.7 AND hard_negative_accuracy > 0.7",
            "c_exceeds_ceiling_balanced": f"overall_accuracy > {theoretical_ceiling:.4f} AND balanced_any",
        },
        "n_cells": len(cells), "n_failed_runs_excluded": n_failed_runs,
        "summary": {
            "n_balanced_any": n_bal_any, "n_balanced_strong": n_bal_strong,
            "n_exceeds_ceiling_balanced": n_exceed,
            "mean_pair_both_correct_fraction": float(np.mean(pair_both_correct_fracs)) if pair_both_correct_fracs else None,
        },
        "baseline_comparison_existing_non_contrastive_experiment": baseline_comparison,
        "by_architecture_objective": aggregate_by_arch_objective(cells),
        "winning_objective_per_architecture": winning_objective_per_architecture(cells),
        "cells": cells,
    }
    out_path = RESULTS / "fix_full_sweep_analysis_marked_reversal_contrastive.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")
    print(f"\ncontrastive-trained: {n_bal_any}/{len(cells)} balanced_any, "
          f"{n_bal_strong}/{len(cells)} balanced_strong, {n_exceed} exceed_ceiling_balanced")
    if baseline_comparison:
        print(f"baseline (existing fix experiment): {baseline_comparison['combined_balanced_any']} balanced_any")


if __name__ == "__main__":
    main()
