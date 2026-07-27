"""Full-sweep per-cell decomposition analysis for modular-arithmetic-simple,
matching the marked-reversal/binary-addition/dyck-2-3 template exactly: for
each of the 160 corrected-trained cells (10 seeds x 4 architectures x 4
objectives) and each of the 160 standard-FLaRe-trained cells, re-evaluates
on the corrected test set to recover PER-EXAMPLE correctness (the
training-time JSONs only stored aggregate accuracy), then reports
positive/hard-negative/overall accuracy plus three balanced-discrimination
criteria:
  (a) balanced_any:    positive_accuracy > 0.5 AND hard_negative_accuracy > 0.5
  (b) balanced_strong: positive_accuracy > 0.7 AND hard_negative_accuracy > 0.7
  (c) exceeds_ceiling_balanced: overall_accuracy > 0.8745 (theoretical
      shortcut-only ceiling from fix_baseline_ceiling_modular_arithmetic_
      simple.json) AND balanced_any

Also decomposes hard-negative accuracy by PERMUTATION POSITION TERTILE
(low/mid/high -- from hard-negative-meta.jsonl's violation_position_tertile,
MF1's Option-A stratified field) per cell, to check whether any fix effect
(where present) is uniform across early/mid/late swap positions or
concentrated in some -- analogous to Dyck-2-3's subtype decomposition, but
along this task's own structural axis (position in a long left-to-right
accumulation) rather than violation type.

Finally reports the winning objective (highest mean corrected-trained
overall accuracy on the corrected test set) per architecture.

PYTHONPATH=src:analysis python analysis/fix_full_sweep_analysis_modular_arithmetic_simple.py
"""

import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "analysis")
from fix_train_models_all_objectives import load_model, load_split, evaluate_recognition

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "analysis_outputs" / "final_results"
FIXED_LANG_DIR = REPO_ROOT / "languages" / "modular-arithmetic-simple-fixed"
FIXED_TEST_DIR = FIXED_LANG_DIR / "datasets" / "test"
TASK = "modular-arithmetic-simple"
OBJECTIVES = ["rec", "rec+lm", "rec+ns", "rec+lm+ns"]
ARCHITECTURES = ["rnn", "lstm", "transformer", "mamba"]
THEORETICAL_CEILING = 0.8744510978043912
TERTILES = ["low", "mid", "high"]


def load_fixed_test():
    seqs, labels = load_split(FIXED_TEST_DIR)
    kinds = (FIXED_TEST_DIR / "negative-kind.txt").read_text().splitlines()
    tertiles = []
    with (FIXED_TEST_DIR / "hard-negative-meta.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            tertiles.append(d.get("violation_position_tertile"))
    assert len(seqs) == len(labels) == len(kinds) == len(tertiles)
    return seqs, labels, kinds, tertiles


def build_indices(labels, kinds, tertiles):
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    uniform_idx = [i for i, (l, k) in enumerate(zip(labels, kinds)) if l == 0 and k == "uniform_random"]
    hard_idx = [i for i, (l, k) in enumerate(zip(labels, kinds)) if l == 0 and k == "hard_negative"]
    tertile_idx = {t: [i for i in hard_idx if tertiles[i] == t] for t in TERTILES}
    n_no_tertile = sum(1 for i in hard_idx if tertiles[i] not in TERTILES)
    return pos_idx, uniform_idx, hard_idx, tertile_idx, n_no_tertile


def subset_acc(correct, idx):
    vals = [correct[i] for i in idx if correct[i] is not None]
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def evaluate_cell(model_dir, arch, seqs, labels, vocab_tokens, pos_idx, uniform_idx, hard_idx, tertile_idx):
    iface, saver = load_model(model_dir, FIXED_LANG_DIR, arch)
    correct, n_failed = evaluate_recognition(iface, saver, seqs, labels, vocab_tokens)
    overall_acc, n = subset_acc(correct, list(range(len(labels))))
    pos_acc, n_pos = subset_acc(correct, pos_idx)
    uniform_acc, n_uniform = subset_acc(correct, uniform_idx)
    hard_acc, n_hard = subset_acc(correct, hard_idx)
    tertile_accs = {}
    for t in TERTILES:
        acc, n_t = subset_acc(correct, tertile_idx[t])
        tertile_accs[t] = {"accuracy": acc, "n": n_t}
    return {
        "overall_accuracy": overall_acc, "n": n, "n_inference_failed": n_failed,
        "positive_accuracy": pos_acc, "n_positive": n_pos,
        "uniform_random_accuracy": uniform_acc, "n_uniform_random": n_uniform,
        "hard_negative_accuracy": hard_acc, "n_hard_negative": n_hard,
        "hard_negative_by_position_tertile": tertile_accs,
    }


def make_cell_record(population, arch, objective, seed_or_trial, ev):
    balanced_any = ev["positive_accuracy"] > 0.5 and ev["hard_negative_accuracy"] > 0.5
    balanced_strong = ev["positive_accuracy"] > 0.7 and ev["hard_negative_accuracy"] > 0.7
    exceeds_ceiling_balanced = ev["overall_accuracy"] > THEORETICAL_CEILING and balanced_any
    return {
        "population": population, "arch": arch, "objective": objective, "seed": seed_or_trial,
        "overall_accuracy": ev["overall_accuracy"],
        "positive_accuracy": ev["positive_accuracy"],
        "hard_negative_accuracy": ev["hard_negative_accuracy"],
        "uniform_random_accuracy": ev["uniform_random_accuracy"],
        "hard_negative_by_position_tertile": ev["hard_negative_by_position_tertile"],
        "balanced_any": balanced_any, "balanced_strong": balanced_strong,
        "exceeds_ceiling_balanced": exceeds_ceiling_balanced,
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
            "mean_hard_negative_by_position_tertile": {
                t: float(np.mean([c["hard_negative_by_position_tertile"][t]["accuracy"] for c in subset
                                  if not np.isnan(c["hard_negative_by_position_tertile"][t]["accuracy"])]))
                if subset else None
                for t in TERTILES
            },
        }
    return out


def winning_objective_per_arch(corrected_by_arch_obj):
    winners = {}
    for arch in ARCHITECTURES:
        rows = [(obj, corrected_by_arch_obj[f"{arch}_{obj}"]["mean_overall_accuracy"]) for obj in OBJECTIVES]
        best_obj, best_acc = max(rows, key=lambda x: x[1])
        winners[arch] = {"winning_objective": best_obj, "mean_overall_accuracy": best_acc,
                         "all_objectives": {obj: acc for obj, acc in rows}}
    return winners


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    vocab_tokens = torch.load(FIXED_LANG_DIR / "main.vocab", weights_only=False)["tokens"]
    seqs, labels, kinds, tertiles = load_fixed_test()
    pos_idx, uniform_idx, hard_idx, tertile_idx, n_no_tertile = build_indices(labels, kinds, tertiles)
    print(f"corrected test set: n={len(seqs)} n_pos={len(pos_idx)} n_uniform={len(uniform_idx)} "
          f"n_hard={len(hard_idx)} n_no_tertile_among_hard={n_no_tertile}", flush=True)
    for t in TERTILES:
        print(f"  tertile {t}: n={len(tertile_idx[t])}", flush=True)

    out_path = RESULTS / "fix_full_sweep_analysis_modular_arithmetic_simple.json"
    corrected_cells, standard_cells = [], []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            corrected_cells = existing["corrected_trained"]["cells"]
            standard_cells = existing["standard_trained"]["cells"]
            print(f"resuming: {len(corrected_cells)} corrected cells, {len(standard_cells)} standard cells already done", flush=True)
        except Exception:
            pass
    corrected_done = {(c["arch"], c["objective"], c["seed"]) for c in corrected_cells}
    standard_done = {(c["arch"], c["objective"], c["seed"]) for c in standard_cells}

    def save():
        crit = {
            "a_balanced_any": "positive_accuracy > 0.5 AND hard_negative_accuracy > 0.5",
            "b_balanced_strong": "positive_accuracy > 0.7 AND hard_negative_accuracy > 0.7",
            "c_exceeds_ceiling_balanced": f"overall_accuracy > {THEORETICAL_CEILING:.4f} AND balanced_any",
        }
        corrected_by_ao = aggregate_by_arch_objective(corrected_cells)
        standard_by_ao = aggregate_by_arch_objective(standard_cells)
        out = {
            "task": TASK,
            "experiment": (
                "Full-sweep per-cell decomposition analysis: does overall-accuracy improvement in "
                "corrected-trained models (vs standard-trained) reflect genuine balanced "
                "discrimination, and is any such effect uniform across the three hard-negative "
                "permutation-position tertiles (low/mid/high) or concentrated in some?"
            ),
            "theoretical_shortcut_only_ceiling": THEORETICAL_CEILING,
            "criteria": crit,
            "position_tertiles": TERTILES,
            "corrected_trained": {
                "n_cells": len(corrected_cells), "cells": corrected_cells,
                "by_architecture_objective": corrected_by_ao,
            },
            "standard_trained": {
                "n_cells": len(standard_cells), "cells": standard_cells,
                "by_architecture_objective": standard_by_ao,
            },
            "winning_objective_per_architecture_corrected_trained_on_corrected_test": (
                winning_objective_per_arch(corrected_by_ao) if len(corrected_cells) == 160 else None
            ),
        }
        out_path.write_text(json.dumps(out, indent=2, default=str))

    # ------------------------------------------------------------------
    # corrected-trained cells
    # ------------------------------------------------------------------
    for objective in OBJECTIVES:
        d = json.loads((RESULTS / f"fix_trained_models_modular_arithmetic_simple_{objective.replace('+', '_')}.json").read_text())
        for r in d["corrected_trained_runs"]:
            arch, trial_no = r["arch"], r["trial_no"]
            if (arch, objective, trial_no) in corrected_done:
                continue
            if r.get("train_failed") or r.get("eval_failed"):
                print(f"  skipping corrected {arch}/{objective}/trial{trial_no}: failed run", flush=True)
                continue
            print(f"=== corrected-trained {arch} {objective} trial {trial_no} ===", flush=True)
            ev = evaluate_cell(r["model_dir"], arch, seqs, labels, vocab_tokens, pos_idx, uniform_idx, hard_idx, tertile_idx)
            cell = make_cell_record("corrected_trained", arch, objective, trial_no, ev)
            print(f"  overall={cell['overall_accuracy']:.4f} pos={cell['positive_accuracy']:.4f} "
                  f"hard={cell['hard_negative_accuracy']:.4f} "
                  f"tertiles={{low:{ev['hard_negative_by_position_tertile']['low']['accuracy']:.3f}, "
                  f"mid:{ev['hard_negative_by_position_tertile']['mid']['accuracy']:.3f}, "
                  f"high:{ev['hard_negative_by_position_tertile']['high']['accuracy']:.3f}}}", flush=True)
            corrected_cells.append(cell)
            save()

    # ------------------------------------------------------------------
    # standard-FLaRe-trained cells
    # ------------------------------------------------------------------
    for objective in OBJECTIVES:
        d = json.loads((RESULTS / f"fix_trained_models_modular_arithmetic_simple_{objective.replace('+', '_')}.json").read_text())
        for r in d["standard_trained_on_corrected_test"]:
            arch, seed = r["arch"], r["seed"]
            if (arch, objective, seed) in standard_done:
                continue
            print(f"=== standard-trained {arch} {objective} seed{seed} ===", flush=True)
            ev = evaluate_cell(r["model_dir"], arch, seqs, labels, vocab_tokens, pos_idx, uniform_idx, hard_idx, tertile_idx)
            cell = make_cell_record("standard_trained", arch, objective, seed, ev)
            print(f"  overall={cell['overall_accuracy']:.4f} pos={cell['positive_accuracy']:.4f} "
                  f"hard={cell['hard_negative_accuracy']:.4f} "
                  f"tertiles={{low:{ev['hard_negative_by_position_tertile']['low']['accuracy']:.3f}, "
                  f"mid:{ev['hard_negative_by_position_tertile']['mid']['accuracy']:.3f}, "
                  f"high:{ev['hard_negative_by_position_tertile']['high']['accuracy']:.3f}}}", flush=True)
            standard_cells.append(cell)
            save()

    print(f"\nSaved {out_path}")
    n_bal_corrected = sum(1 for c in corrected_cells if c["balanced_any"])
    n_bal_standard = sum(1 for c in standard_cells if c["balanced_any"])
    print(f"corrected_trained: {n_bal_corrected}/{len(corrected_cells)} balanced_any")
    print(f"standard_trained: {n_bal_standard}/{len(standard_cells)} balanced_any")


if __name__ == "__main__":
    main()
