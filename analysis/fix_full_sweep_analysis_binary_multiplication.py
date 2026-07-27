"""Full-sweep per-cell decomposition analysis for binary-multiplication, matching the
Dyck-2-3/modular-arithmetic-simple template: for each of the 160 corrected-
trained cells (10 seeds x 4 architectures x 4 objectives) and each of the
160 standard-FLaRe-trained cells, re-evaluates on the corrected test set to
recover PER-EXAMPLE correctness, then reports positive/hard-negative/
overall accuracy plus three balanced-discrimination criteria:
  (a) balanced_any:    positive_accuracy > 0.5 AND hard_negative_accuracy > 0.5
  (b) balanced_strong: positive_accuracy > 0.7 AND hard_negative_accuracy > 0.7
  (c) exceeds_ceiling_balanced: overall_accuracy > THEORETICAL_CEILING
      (from fix_baseline_ceiling_binary_multiplication.json) AND balanced_any

IMPORTANT MAGNITUDE-SCALING NOTE (per explicit user instruction): hard-
negative bit-flip position tertiles carry an ASTRONOMICAL magnitude scale,
not a uniform one -- BMF1's audit found low-tertile flips average bit_idx
~20 (magnitude change ~2^20) while high-tertile flips average bit_idx ~99
(magnitude change ~2^99, at the longest test sequences up to bit_idx 409).
Position-sweep decomposition on the hard-negative subset is therefore NOT
merely a "where in the string" check the way it is for repeat-01/odds-
first/marked-copy -- a cell that only discriminates low-tertile hard
negatives is only catching SMALL-MAGNITUDE errors (a claimed product off by
a few units), not doing genuine bit-exact product verification. Only balanced
discrimination that holds ACROSS ALL THREE TERTILES (flat hard_negative_by_
position_tertile) is evidence of true value-level verification; a cell
with high low-tertile accuracy and near-floor high-tertile accuracy is
better read as coarse magnitude-sensitivity (matching the same confound
documented for binary-addition/compute-sqrt's Phase 3 causal-patching
pilot), not comprehensive correctness checking. This magnitude-scaling
relationship (bit_idx, magnitude=2**bit_idx) is reported per tertile below
(magnitude_scaling_by_tertile, a property of the TEST SET's hard-negative
population, not of any individual trained model).

PYTHONPATH=src:analysis python analysis/fix_full_sweep_analysis_binary_multiplication.py
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
FIXED_LANG_DIR = REPO_ROOT / "languages" / "binary-multiplication-fixed"
FIXED_TEST_DIR = FIXED_LANG_DIR / "datasets" / "test"
TASK = "binary-multiplication"
OBJECTIVES = ["rec", "rec+lm", "rec+ns", "rec+lm+ns"]
ARCHITECTURES = ["rnn", "lstm", "transformer", "mamba"]
TERTILES = ["low", "mid", "high"]


def load_theoretical_ceiling():
    d = json.loads((RESULTS / "fix_baseline_ceiling_binary_multiplication.json").read_text())
    return d["theoretical_shortcut_only_ceiling"]["theoretical_shortcut_only_ceiling"]


def load_fixed_test():
    seqs, labels = load_split(FIXED_TEST_DIR)
    kinds = (FIXED_TEST_DIR / "negative-kind.txt").read_text().splitlines()
    tertiles, bit_idxs = [], []
    with (FIXED_TEST_DIR / "hard-negative-bitflip-meta.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            tertiles.append(d.get("bit_position_tertile"))
            bit_idxs.append(d.get("bit_idx"))
    assert len(seqs) == len(labels) == len(kinds) == len(tertiles)
    return seqs, labels, kinds, tertiles, bit_idxs


def build_indices(labels, kinds, tertiles):
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    uniform_idx = [i for i, (l, k) in enumerate(zip(labels, kinds)) if l == 0 and k == "uniform_random"]
    hard_idx = [i for i, (l, k) in enumerate(zip(labels, kinds)) if l == 0 and k == "hard_negative"]
    tertile_idx = {t: [i for i in hard_idx if tertiles[i] == t] for t in TERTILES}
    n_no_tertile = sum(1 for i in hard_idx if tertiles[i] not in TERTILES)
    return pos_idx, uniform_idx, hard_idx, tertile_idx, n_no_tertile


def magnitude_scaling_by_tertile(hard_idx, tertile_idx, bit_idxs):
    out = {}
    for t in TERTILES:
        idxs = [bit_idxs[i] for i in tertile_idx[t] if bit_idxs[i] is not None]
        if not idxs:
            out[t] = None
            continue
        arr = np.array(idxs)
        out[t] = {
            "n": len(idxs), "bit_idx_min": int(arr.min()), "bit_idx_max": int(arr.max()),
            "bit_idx_mean": float(arr.mean()), "bit_idx_median": float(np.median(arr)),
            "magnitude_at_min_bit_idx": 2 ** int(arr.min()), "magnitude_at_mean_bit_idx_approx": 2 ** round(float(arr.mean())),
        }
    return out


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


def make_cell_record(population, arch, objective, seed_or_trial, ev, theoretical_ceiling):
    balanced_any = ev["positive_accuracy"] > 0.5 and ev["hard_negative_accuracy"] > 0.5
    balanced_strong = ev["positive_accuracy"] > 0.7 and ev["hard_negative_accuracy"] > 0.7
    exceeds_ceiling_balanced = ev["overall_accuracy"] > theoretical_ceiling and balanced_any
    tert = ev["hard_negative_by_position_tertile"]
    low_acc, high_acc = tert["low"]["accuracy"], tert["high"]["accuracy"]
    flat_across_tertiles = (
        not np.isnan(low_acc) and not np.isnan(high_acc) and abs(low_acc - high_acc) <= 0.2
    )
    return {
        "population": population, "arch": arch, "objective": objective, "seed": seed_or_trial,
        "overall_accuracy": ev["overall_accuracy"],
        "positive_accuracy": ev["positive_accuracy"],
        "hard_negative_accuracy": ev["hard_negative_accuracy"],
        "uniform_random_accuracy": ev["uniform_random_accuracy"],
        "hard_negative_by_position_tertile": ev["hard_negative_by_position_tertile"],
        "balanced_any": balanced_any, "balanced_strong": balanced_strong,
        "exceeds_ceiling_balanced": exceeds_ceiling_balanced,
        "flat_across_tertiles_within_0.2": flat_across_tertiles,
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
            "n_flat_across_tertiles": sum(1 for c in subset if c["flat_across_tertiles_within_0.2"]),
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
    theoretical_ceiling = load_theoretical_ceiling()
    vocab_tokens = torch.load(FIXED_LANG_DIR / "main.vocab", weights_only=False)["tokens"]
    seqs, labels, kinds, tertiles, bit_idxs = load_fixed_test()
    pos_idx, uniform_idx, hard_idx, tertile_idx, n_no_tertile = build_indices(labels, kinds, tertiles)
    mag_scaling = magnitude_scaling_by_tertile(hard_idx, tertile_idx, bit_idxs)
    print(f"corrected test set: n={len(seqs)} n_pos={len(pos_idx)} n_uniform={len(uniform_idx)} "
          f"n_hard={len(hard_idx)} n_no_tertile_among_hard={n_no_tertile} ceiling={theoretical_ceiling:.4f}", flush=True)
    for t in TERTILES:
        print(f"  tertile {t}: n={len(tertile_idx[t])} magnitude={mag_scaling[t]}", flush=True)

    out_path = RESULTS / "fix_full_sweep_analysis_binary_multiplication.json"
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
            "c_exceeds_ceiling_balanced": f"overall_accuracy > {theoretical_ceiling:.4f} AND balanced_any",
            "d_flat_across_tertiles_within_0.2": (
                "|low_tertile_hard_neg_acc - high_tertile_hard_neg_acc| <= 0.2 -- REQUIRED (not "
                "just balanced_any) to credit a cell with genuine bit-exact product verification, since "
                "low/high tertiles differ by up to ~2^56x in magnitude (see magnitude_scaling_by_"
                "tertile below); a cell strong on low but weak on high is catching small-magnitude "
                "errors only, not verifying the value exactly."
            ),
        }
        out = {
            "task": TASK,
            "experiment": (
                "Full-sweep per-cell decomposition analysis: does overall-accuracy improvement in "
                "corrected-trained models (vs standard-trained) reflect genuine balanced "
                "discrimination, and does it hold uniformly across the astronomically-scaled low/"
                "mid/high bit-flip-magnitude tertiles, or only at small magnitudes?"
            ),
            "magnitude_scaling_by_tertile": mag_scaling,
            "magnitude_scaling_note": (
                "Property of the TEST SET's hard-negative population (not model-dependent): LSB-"
                "first encoding means 'low' tertile bit flips change the claimed value by a small "
                "amount (bit_idx ~0-20ish), while 'high' tertile flips change it by an astronomically "
                "larger amount (bit_idx up to 344, i.e. magnitude changes up to 2^344) at the longest "
                "test sequences. Position-sweep decomposition on the hard-negative subset is critical "
                "for this task specifically: a cell that only discriminates low-tertile hard negatives "
                "is only catching small-magnitude errors, not doing bit-exact product verification."
            ),
            "theoretical_shortcut_only_ceiling": theoretical_ceiling,
            "criteria": crit,
            "position_tertiles": TERTILES,
            "corrected_trained": {
                "n_cells": len(corrected_cells), "cells": corrected_cells,
                "by_architecture_objective": aggregate_by_arch_objective(corrected_cells),
                "winning_objective_per_architecture": winning_objective_per_architecture(corrected_cells),
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
        run_file = RESULTS / f"fix_trained_models_binary_multiplication_{objective.replace('+', '_')}.json"
        if not run_file.exists():
            print(f"  {run_file} not found yet, skipping objective={objective}", flush=True)
            continue
        d = json.loads(run_file.read_text())
        for r in d["corrected_trained_runs"]:
            arch, trial_no = r["arch"], r["trial_no"]
            if (arch, objective, trial_no) in corrected_done:
                continue
            if r.get("train_failed") or r.get("eval_failed"):
                print(f"  skipping corrected {arch}/{objective}/trial{trial_no}: failed run", flush=True)
                continue
            print(f"=== corrected-trained {arch} {objective} trial {trial_no} ===", flush=True)
            ev = evaluate_cell(r["model_dir"], arch, seqs, labels, vocab_tokens, pos_idx, uniform_idx, hard_idx, tertile_idx)
            cell = make_cell_record("corrected_trained", arch, objective, trial_no, ev, theoretical_ceiling)
            print(f"  overall={cell['overall_accuracy']:.4f} pos={cell['positive_accuracy']:.4f} "
                  f"hard={cell['hard_negative_accuracy']:.4f} "
                  f"tertiles={{low:{ev['hard_negative_by_position_tertile']['low']['accuracy']:.3f}, "
                  f"mid:{ev['hard_negative_by_position_tertile']['mid']['accuracy']:.3f}, "
                  f"high:{ev['hard_negative_by_position_tertile']['high']['accuracy']:.3f}}} "
                  f"flat={cell['flat_across_tertiles_within_0.2']}", flush=True)
            corrected_cells.append(cell)
            save()

    # ------------------------------------------------------------------
    # standard-FLaRe-trained cells
    # ------------------------------------------------------------------
    std_file = RESULTS / "fix_standard_baseline_on_corrected_test_all_objectives_binary_multiplication.json"
    if std_file.exists():
        d = json.loads(std_file.read_text())
        for r in d["runs"]:
            if r.get("eval_failed"):
                continue
            arch, objective, trial = r["arch"], r["loss_terms"], r["trial_no"]
            if (arch, objective, trial) in standard_done:
                continue
            print(f"=== standard-trained {arch} {objective} trial{trial} ===", flush=True)
            ev = evaluate_cell(r["model_dir"], arch, seqs, labels, vocab_tokens, pos_idx, uniform_idx, hard_idx, tertile_idx)
            cell = make_cell_record("standard_trained", arch, objective, trial, ev, theoretical_ceiling)
            print(f"  overall={cell['overall_accuracy']:.4f} pos={cell['positive_accuracy']:.4f} "
                  f"hard={cell['hard_negative_accuracy']:.4f} flat={cell['flat_across_tertiles_within_0.2']}", flush=True)
            standard_cells.append(cell)
            save()
    else:
        print(f"  {std_file} not found yet -- run the standard-baseline evaluation first", flush=True)

    print(f"\nSaved {out_path}")
    n_bal_corrected = sum(1 for c in corrected_cells if c["balanced_any"])
    n_bal_standard = sum(1 for c in standard_cells if c["balanced_any"])
    n_flat_corrected = sum(1 for c in corrected_cells if c["flat_across_tertiles_within_0.2"])
    print(f"corrected_trained: {n_bal_corrected}/{len(corrected_cells)} balanced_any, "
          f"{n_flat_corrected}/{len(corrected_cells)} flat_across_tertiles")
    print(f"standard_trained: {n_bal_standard}/{len(standard_cells)} balanced_any")


if __name__ == "__main__":
    main()
