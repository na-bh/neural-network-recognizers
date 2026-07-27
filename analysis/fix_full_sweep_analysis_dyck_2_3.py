"""Full-sweep per-cell decomposition analysis for Dyck-2-3, matching the
marked-reversal/binary-addition template exactly: for each of the 160
corrected-trained cells (10 seeds x 4 architectures x 4 objectives) and each
of the 160 standard-FLaRe-trained cells, re-evaluates on the corrected test
set to recover PER-EXAMPLE correctness (the training-time JSONs only stored
aggregate accuracy), then reports positive/hard-negative/overall accuracy
plus three balanced-discrimination criteria:
  (a) balanced_any:    positive_accuracy > 0.5 AND hard_negative_accuracy > 0.5
  (b) balanced_strong: positive_accuracy > 0.7 AND hard_negative_accuracy > 0.7
  (c) exceeds_ceiling_balanced: overall_accuracy > 0.8761 (theoretical
      shortcut-only ceiling from fix_baseline_ceiling_dyck_2_3.json) AND
      balanced_any

Also decomposes hard-negative accuracy by SUBTYPE (type_mismatch,
close_with_nothing_open, depth_exceeded -- from hard-negative-subtype-meta.
jsonl) per cell, to check whether the fix's effect (where present) is
uniform across subtypes or concentrated in some.

PYTHONPATH=src:analysis python analysis/fix_full_sweep_analysis_dyck_2_3.py
"""

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "analysis")
from fix_train_models_all_objectives import load_model, load_split, evaluate_recognition
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "analysis_outputs" / "final_results"
FIXED_LANG_DIR = REPO_ROOT / "languages" / "dyck-2-3-fixed"
FIXED_TEST_DIR = FIXED_LANG_DIR / "datasets" / "test"
TASK = "dyck-2-3"
OBJECTIVES = ["rec", "rec+lm", "rec+ns", "rec+lm+ns"]
ARCHITECTURES = ["rnn", "lstm", "transformer", "mamba"]
THEORETICAL_CEILING = 0.8761477045908184
SUBTYPES = ["type_mismatch", "close_with_nothing_open", "depth_exceeded"]


def load_fixed_test():
    seqs, labels = load_split(FIXED_TEST_DIR)
    kinds = (FIXED_TEST_DIR / "negative-kind.txt").read_text().splitlines()
    subtype_meta = []
    with (FIXED_TEST_DIR / "hard-negative-subtype-meta.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            subtype_meta.append(d.get("subtype"))
    assert len(seqs) == len(labels) == len(kinds) == len(subtype_meta)
    return seqs, labels, kinds, subtype_meta


def build_indices(labels, kinds, subtype_meta):
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    uniform_idx = [i for i, (l, k) in enumerate(zip(labels, kinds)) if l == 0 and k == "uniform_random"]
    hard_idx = [i for i, (l, k) in enumerate(zip(labels, kinds)) if l == 0 and k == "hard_negative"]
    subtype_idx = {st: [i for i in hard_idx if subtype_meta[i] == st] for st in SUBTYPES}
    n_no_subtype = sum(1 for i in hard_idx if subtype_meta[i] not in SUBTYPES)
    return pos_idx, uniform_idx, hard_idx, subtype_idx, n_no_subtype


def subset_acc(correct, idx):
    vals = [correct[i] for i in idx if correct[i] is not None]
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def evaluate_cell(model_dir, arch, seqs, labels, vocab_tokens, pos_idx, uniform_idx, hard_idx, subtype_idx):
    iface, saver = load_model(model_dir, FIXED_LANG_DIR, arch)
    correct, n_failed = evaluate_recognition(iface, saver, seqs, labels, vocab_tokens)
    overall_acc, n = subset_acc(correct, list(range(len(labels))))
    pos_acc, n_pos = subset_acc(correct, pos_idx)
    uniform_acc, n_uniform = subset_acc(correct, uniform_idx)
    hard_acc, n_hard = subset_acc(correct, hard_idx)
    subtype_accs = {}
    for st in SUBTYPES:
        acc, n_st = subset_acc(correct, subtype_idx[st])
        subtype_accs[st] = {"accuracy": acc, "n": n_st}
    return {
        "overall_accuracy": overall_acc, "n": n, "n_inference_failed": n_failed,
        "positive_accuracy": pos_acc, "n_positive": n_pos,
        "uniform_random_accuracy": uniform_acc, "n_uniform_random": n_uniform,
        "hard_negative_accuracy": hard_acc, "n_hard_negative": n_hard,
        "hard_negative_by_subtype": subtype_accs,
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
        "hard_negative_by_subtype": ev["hard_negative_by_subtype"],
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
            "mean_hard_negative_by_subtype": {
                st: float(np.mean([c["hard_negative_by_subtype"][st]["accuracy"] for c in subset
                                   if not np.isnan(c["hard_negative_by_subtype"][st]["accuracy"])]))
                if subset else None
                for st in SUBTYPES
            },
        }
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    vocab_tokens = torch.load(FIXED_LANG_DIR / "main.vocab", weights_only=False)["tokens"]
    seqs, labels, kinds, subtype_meta = load_fixed_test()
    pos_idx, uniform_idx, hard_idx, subtype_idx, n_no_subtype = build_indices(labels, kinds, subtype_meta)
    print(f"corrected test set: n={len(seqs)} n_pos={len(pos_idx)} n_uniform={len(uniform_idx)} "
          f"n_hard={len(hard_idx)} n_no_subtype_among_hard={n_no_subtype}", flush=True)
    for st in SUBTYPES:
        print(f"  subtype {st}: n={len(subtype_idx[st])}", flush=True)

    out_path = RESULTS / "fix_full_sweep_analysis_dyck_2_3.json"
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
        out = {
            "task": TASK,
            "experiment": (
                "Full-sweep per-cell decomposition analysis: does overall-accuracy improvement in "
                "corrected-trained models (vs standard-trained) reflect genuine balanced "
                "discrimination, and is any such effect uniform across the three hard-negative "
                "subtypes (type_mismatch, close_with_nothing_open, depth_exceeded) or concentrated "
                "in some?"
            ),
            "theoretical_shortcut_only_ceiling": THEORETICAL_CEILING,
            "criteria": crit,
            "subtypes": SUBTYPES,
            "corrected_trained": {
                "n_cells": len(corrected_cells), "cells": corrected_cells,
                "by_architecture_objective": aggregate_by_arch_objective(corrected_cells),
            },
            "standard_trained": {
                "n_cells": len(standard_cells), "cells": standard_cells,
                "by_architecture_objective": aggregate_by_arch_objective(standard_cells),
            },
        }
        out_path.write_text(json.dumps(out, indent=2, default=str))

    # ------------------------------------------------------------------
    # corrected-trained cells
    # ------------------------------------------------------------------
    for objective in OBJECTIVES:
        d = json.loads((RESULTS / f"fix_trained_models_dyck_2_3_{objective.replace('+', '_')}.json").read_text())
        for r in d["corrected_trained_runs"]:
            arch, trial_no = r["arch"], r["trial_no"]
            if (arch, objective, trial_no) in corrected_done:
                continue
            if r.get("train_failed") or r.get("eval_failed"):
                print(f"  skipping corrected {arch}/{objective}/trial{trial_no}: failed run", flush=True)
                continue
            print(f"=== corrected-trained {arch} {objective} trial {trial_no} ===", flush=True)
            ev = evaluate_cell(r["model_dir"], arch, seqs, labels, vocab_tokens, pos_idx, uniform_idx, hard_idx, subtype_idx)
            cell = make_cell_record("corrected_trained", arch, objective, trial_no, ev)
            print(f"  overall={cell['overall_accuracy']:.4f} pos={cell['positive_accuracy']:.4f} "
                  f"hard={cell['hard_negative_accuracy']:.4f} "
                  f"subtypes={{tm:{ev['hard_negative_by_subtype']['type_mismatch']['accuracy']:.3f}, "
                  f"cwno:{ev['hard_negative_by_subtype']['close_with_nothing_open']['accuracy']:.3f}, "
                  f"de:{ev['hard_negative_by_subtype']['depth_exceeded']['accuracy']:.3f}}}", flush=True)
            corrected_cells.append(cell)
            save()

    # ------------------------------------------------------------------
    # standard-FLaRe-trained cells
    # ------------------------------------------------------------------
    for objective in OBJECTIVES:
        d = json.loads((RESULTS / f"fix_trained_models_dyck_2_3_{objective.replace('+', '_')}.json").read_text())
        for r in d["standard_trained_on_corrected_test"]:
            arch, seed = r["arch"], r["seed"]
            if (arch, objective, seed) in standard_done:
                continue
            print(f"=== standard-trained {arch} {objective} seed{seed} ===", flush=True)
            ev = evaluate_cell(r["model_dir"], arch, seqs, labels, vocab_tokens, pos_idx, uniform_idx, hard_idx, subtype_idx)
            cell = make_cell_record("standard_trained", arch, objective, seed, ev)
            print(f"  overall={cell['overall_accuracy']:.4f} pos={cell['positive_accuracy']:.4f} "
                  f"hard={cell['hard_negative_accuracy']:.4f} "
                  f"subtypes={{tm:{ev['hard_negative_by_subtype']['type_mismatch']['accuracy']:.3f}, "
                  f"cwno:{ev['hard_negative_by_subtype']['close_with_nothing_open']['accuracy']:.3f}, "
                  f"de:{ev['hard_negative_by_subtype']['depth_exceeded']['accuracy']:.3f}}}", flush=True)
            standard_cells.append(cell)
            save()

    print(f"\nSaved {out_path}")
    n_bal_corrected = sum(1 for c in corrected_cells if c["balanced_any"])
    n_bal_standard = sum(1 for c in standard_cells if c["balanced_any"])
    print(f"corrected_trained: {n_bal_corrected}/{len(corrected_cells)} balanced_any")
    print(f"standard_trained: {n_bal_standard}/{len(standard_cells)} balanced_any")


if __name__ == "__main__":
    main()
