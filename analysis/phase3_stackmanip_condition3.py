"""Stack-manipulation Phase 3, condition 3 of 3: target-computation null
test (the load-bearing test) on the value_mismatch_only residual. clean/
corrupt pairs hold all six shortcut features fixed (same length, marker
count/position, well-formedness, arithmetic-consistent answer length,
bit-count availability) and differ ONLY in whether the answer's LIFO order
is correct, via an adjacent-differing-value swap. Swept early/mid/high
tertile by swap position, N=25 PER TERTILE per cell (75 pairs per cell, 8
cells).

BEFORE interpreting the causal (patching) results, this ALSO runs a direct-
accuracy check (bypassing patching entirely) on the exact 62-example
value_mismatch_only shortcut-blind population from the REAL test set
(phase3_stackmanip_task_audit.json's joint residual) for each of the 8
cells. Per the user's explicit three-way framing, each cell is classified
into exactly one of:
  READING 1 -- substantially-above-chance real-subset accuracy AND causal
    Condition 3 gap: genuine stack-simulation verification (surprising,
    worth investigating further).
  READING 2 -- near-constant real-subset accuracy (floor) AND near-zero
    Condition 3 gap: floor-effect pattern matching missing-duplicate-
    string -- the causal null reflects no non-degenerate decision to
    probe, NOT position-insensitivity per se.
  READING 3 -- near-chance real-subset accuracy AND non-trivial Condition 3
    gap: cells causally engage with position-specific features despite
    failing behaviorally -- a distinct, specifically worth-investigating
    pattern.
  UNCLASSIFIED -- doesn't cleanly fit any of the three (e.g. above-chance
    accuracy but near-zero gap, or floor accuracy but large gap) --
    reported explicitly, not forced into one of the three.

PRIMARY evidence for the causal test is the behavioral logit gap; full-
state patch is a SECONDARY wiring check, flagged UNRELIABLE in near-zero-
gap cells.

Saves analysis_outputs/final_results/phase3_stackmanip_condition3.json.

PYTHONPATH=src:analysis python analysis/phase3_stackmanip_condition3.py
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
sys.path.insert(0, "src")
from phase3_stackmanip_counterfactual_design import target_computation_pairs, audit_target_computation_pairs, TERTILES
from phase3_stackmanip_patching_common import CELLS, get_tok2idx, run_cell, summarize
from phase3_stackmanip_task_audit import classify_violation, load_split, SHORTCUT_FEATURES
from recognizers.neural_networks.data import add_data_arguments, load_vocabulary_data
from recognizers.neural_networks.model_interface import RecognitionModelInterface, ModelInput

RESULTS = Path("analysis_outputs/final_results")
TASK = "stack-manipulation"
TASK_DIR = Path("languages/stack-manipulation")
N_PAIRS_PER_TERTILE = 25
DEGENERATE_GAP_THRESHOLD = 0.05
FLOOR_THRESHOLD = 0.10
CHANCE_LOW, CHANCE_HIGH = 0.40, 0.60
ABOVE_CHANCE_THRESHOLD = 0.60

CHECKPOINT_DIRS = {
    ("rnn", 4): Path("data/models/stack-manipulation/rnn/rec+ns/validation-short/4"),
    ("rnn", 1): Path("data/models/stack-manipulation/rnn/rec+ns/validation-short/1"),
    ("lstm", 1): Path("data/models/stack-manipulation/lstm/rec+ns/validation-short/1"),
    ("lstm", 7): Path("data/models/stack-manipulation/lstm/rec+ns/validation-short/7"),
    ("transformer", 5): Path("data/models/stack-manipulation/transformer/rec+ns/validation-short/5"),
    ("transformer", 3): Path("data/models/stack-manipulation/transformer/rec+ns/validation-short/3"),
    ("mamba", 3): Path("models/stack-manipulation/mamba/rec+ns/validation-short/3"),
    ("mamba", 1): Path("models/stack-manipulation/mamba/rec+ns/validation-short/1"),
}


def get_genuinely_hard_population():
    seqs, labels = load_split("test")
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    def passes_all(seq):
        return all(fn(seq) for fn in SHORTCUT_FEATURES.values())

    hard_idx = [i for i in neg_idx if passes_all(seqs[i])]
    assert all(classify_violation(seqs[i]) == "value_mismatch_only" for i in hard_idx)
    return [(seqs[i], labels[i]) for i in hard_idx]


def load_model(arch, model_dir):
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    iface = RecognitionModelInterface()
    iface.add_arguments(parser)
    iface.add_forward_arguments(parser)
    tmp = tempfile.mkdtemp(prefix="stackmanip_condition3_")
    shutil.rmtree(tmp, ignore_errors=True)
    args = parser.parse_args([
        "--output", tmp, "--training-data", str(TASK_DIR), "--architecture", arch,
        "--load-model", str(model_dir), "--load-parameters", "main",
    ])
    vocab = load_vocabulary_data(args, parser)
    saver = iface.construct_saver(args, vocab)
    saver.model.eval()
    return iface, saver


def accuracy_on(iface, saver, tok2idx, pairs):
    if not pairs:
        return float("nan"), 0
    eos_index = saver.kwargs["eos_index"]
    device = next(saver.model.parameters()).device
    correct = []
    for s, l in pairs:
        idx = [tok2idx[t] for t in s]
        content = idx + [eos_index]
        x = torch.tensor([content], dtype=torch.long, device=device)
        last_index = torch.tensor([len(idx)], dtype=torch.long, device=device)
        positive_mask = torch.zeros(1, dtype=torch.bool, device=device)
        mi = ModelInput(x, last_index, positive_mask)
        with torch.no_grad():
            rec, _, _ = iface.get_logits(saver.model, mi)
        correct.append((rec.item() > 0) == bool(l))
    return float(np.mean(correct)), len(pairs)


def classify_reading(acc, gap):
    above_chance = acc >= ABOVE_CHANCE_THRESHOLD
    floor = acc <= FLOOR_THRESHOLD
    near_chance = CHANCE_LOW <= acc <= CHANCE_HIGH
    causal = gap >= DEGENERATE_GAP_THRESHOLD
    if above_chance and causal:
        return "READING_1_genuine_stack_simulation_verification"
    if floor and not causal:
        return "READING_2_floor_effect_no_nondegenerate_decision"
    if near_chance and causal:
        return "READING_3_causal_engagement_despite_behavioral_failure"
    return "UNCLASSIFIED"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    pairs_by_tertile = {}
    audits_by_tertile = {}
    for i, tertile in enumerate(TERTILES):
        rng = np.random.default_rng(3 + i)
        tcp = target_computation_pairs(N_PAIRS_PER_TERTILE, rng, n_stack=3, n_ops=20, fixed_tertile=tertile)
        audit = audit_target_computation_pairs(tcp)
        print(f"=== target_computation_pairs [{tertile}] audit ===", flush=True)
        print(json.dumps(audit, indent=2, default=str))
        assert audit["target_property_isolated"], f"target_computation_pairs[{tertile}] failed to isolate its target property"
        assert all(p["swap_position_tertile"] == tertile for p in tcp)
        pairs_by_tertile[tertile] = tcp
        audits_by_tertile[tertile] = audit

    tok2idx = get_tok2idx()

    # ------------------------------------------------------------------
    # causal patching, all 8 cells x 3 tertiles
    # ------------------------------------------------------------------
    cells = []
    for arch, seed, role in CELLS:
        print(f"\n--- {arch} seed{seed} ({role}) [causal patching] ---", flush=True)
        for tertile in TERTILES:
            pairs = pairs_by_tertile[tertile]
            results, ckpt = run_cell(arch, seed, pairs, tok2idx)
            cell = summarize(results, f"target_computation_{tertile}", arch, seed, role, ckpt)
            cells.append(cell)
            gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
            unreliable = cell["wiring_check_full_state_patch_SECONDARY"]["UNRELIABLE_near_zero_behavioral_gap"]
            print(f"  [{tertile}] PRIMARY behavioral gap={gap:.4f} (causally_effective={cell['causally_effective']})  |  "
                  f"wiring rf_mean={rf:.4f} n={cell['wiring_check_full_state_patch_SECONDARY']['n_nondegenerate_denom']} "
                  f"{'[UNRELIABLE]' if unreliable else ''}", flush=True)

    # ------------------------------------------------------------------
    # direct-accuracy check on the 62-example REAL value_mismatch_only
    # shortcut-blind population, per cell -- done BEFORE interpreting the
    # causal results above
    # ------------------------------------------------------------------
    hard_pop = get_genuinely_hard_population()
    print(f"\n=== direct-accuracy check on the {len(hard_pop)}-example real value_mismatch_only "
          f"shortcut-blind population ===", flush=True)
    hard_accuracy = {}
    for arch, seed, role in CELLS:
        model_dir = CHECKPOINT_DIRS[(arch, seed)]
        iface, saver = load_model(arch, model_dir)
        acc, n = accuracy_on(iface, saver, tok2idx, hard_pop)
        hard_accuracy[f"{arch}_seed{seed}_{role}"] = {"arch": arch, "seed": seed, "role": role, "accuracy": acc, "n": n}
        print(f"  {arch} seed{seed} ({role}): accuracy={acc:.3f} n={n}", flush=True)

    # ------------------------------------------------------------------
    # explicit three-reading classification per cell (using the mean
    # Condition 3 gap across all three tertiles for that cell)
    # ------------------------------------------------------------------
    readings = {}
    for arch, seed, role in CELLS:
        arch_cells = [c for c in cells if c["arch"] == arch and c["seed"] == seed]
        mean_gap = float(np.mean([c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"] for c in arch_cells]))
        max_gap = float(np.max([c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"] for c in arch_cells]))
        acc = hard_accuracy[f"{arch}_seed{seed}_{role}"]["accuracy"]
        reading = classify_reading(acc, mean_gap)
        readings[f"{arch}_seed{seed}_{role}"] = {
            "arch": arch, "seed": seed, "role": role,
            "genuinely_hard_subset_accuracy": acc,
            "condition3_mean_gap_across_tertiles": mean_gap,
            "condition3_max_gap_across_tertiles": max_gap,
            "reading": reading,
        }
    print("\n=== three-reading classification per cell ===", flush=True)
    print(json.dumps(readings, indent=2, default=str))

    reading_counts = {}
    for r in readings.values():
        reading_counts[r["reading"]] = reading_counts.get(r["reading"], 0) + 1
    print("\n=== reading distribution across all 8 cells ===", flush=True)
    print(json.dumps(reading_counts, indent=2))

    out = {
        "condition": "condition_3_target_computation_null_test",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only, flagged UNRELIABLE when the behavioral gap is "
                 "near zero (< 0.05).",
        "description": (
            "Tests genuine LIFO order-sensitivity: clean/corrupt pairs hold all six shortcut "
            "features fixed (length, marker count/position, well-formedness, arithmetic-consistent "
            "answer length, bit-count availability), differing ONLY via an adjacent-differing-value "
            "swap in the answer. Swept early/mid/high tertile by swap position, N=25 per tertile per "
            "cell (75 pairs per cell, 8 cells). ALSO runs a direct-accuracy check (bypassing patching) "
            "on the exact 62-example real value_mismatch_only shortcut-blind population, per cell, "
            "BEFORE interpreting the causal results -- distinguishing genuine verification from a "
            "floor effect from causal engagement without behavioral success."
        ),
        "n_pairs_per_tertile": N_PAIRS_PER_TERTILE,
        "pair_construction_audits_by_tertile": audits_by_tertile,
        "sample_pairs_by_tertile": {t: pairs_by_tertile[t][:2] for t in TERTILES},
        "cells": cells,
        "genuinely_hard_subset_n": len(hard_pop),
        "genuinely_hard_subset_accuracy_per_cell": hard_accuracy,
        "three_reading_classification": readings,
        "reading_distribution": reading_counts,
    }
    out_path = RESULTS / "phase3_stackmanip_condition3.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
