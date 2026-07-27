"""Repeat-01 Phase 3, condition 3 of 3: target-computation null test (the
load-bearing test) on the 115-example genuinely-shortcut-blind population
(NOT viol_repeat_01's 151-example "hard" set -- the task audit's key
correction). clean/corrupt pairs hold all four shortcut features fixed
(length parity, first/last symbol, count balance -- structurally
guaranteed since an adjacent swap can't change length, boundary values
when kept away from position 0/n-1, or the content multiset) and differ
ONLY via an adjacent swap breaking alternation at two positions. Swept
EARLY/MID/LATE tertile by swap position, N=25 per tertile per cell (75
pairs per cell, 8 cells).

INTERPRETATION DISCIPLINE (per the user, matching the task audit): position
has NO magnitude interpretation for this task (unlike the LSB-first
arithmetic tasks). An early>late gap here would match the audit's own
early-concentrated natural violation distribution (58/115 in the first
quintile), NOT magnitude sensitivity. Uniform-large gaps across all three
tertiles would indicate comprehensive verification. A floor-effect (near-
zero accuracy on the 115-example subset AND near-zero Condition 3 gaps)
would join the confounded-shortcut cluster (missing-duplicate-string/
stack-manipulation/compute-sqrt-RNN-LSTM).

BEFORE interpreting the causal (patching) results, this ALSO runs a direct-
accuracy check (bypassing patching entirely) on the exact 115-example
value_mismatch_only shortcut-blind population from the REAL test set, for
each of the 8 cells. Given RNN/LSTM/Mamba scored 100% on BOTH the
(mostly-trivial) 151-example viol-hard bucket AND the 300-example
scrambled bucket (per cell selection), this checks whether that perfect
performance EXTENDS to the genuinely-hard 115 -- which would be the first
pilot task where three non-Transformer architectures show comprehensive
target-computation success under standard training. Transformer is
predicted floor-effect. Any cell scoring unusually high (>=0.90) gets the
long-positive control check before being accepted as genuine verification.

PRIMARY evidence for the causal test is the behavioral logit gap; full-
state patch is a SECONDARY wiring check, flagged UNRELIABLE in near-zero-
gap cells.

Saves analysis_outputs/final_results/phase3_repeat01_condition3.json.

PYTHONPATH=src:analysis python analysis/phase3_repeat01_condition3.py
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
from phase3_repeat01_counterfactual_design import target_computation_pairs, audit_target_computation_pairs, TERTILES, SHORTCUT_FEATURES
from phase3_repeat01_patching_common import CELLS, get_tok2idx, run_cell, summarize
from phase3_repeat01_task_audit import classify_violation, load_split
from recognizers.neural_networks.data import add_data_arguments, load_vocabulary_data
from recognizers.neural_networks.model_interface import RecognitionModelInterface, ModelInput

RESULTS = Path("analysis_outputs/final_results")
TASK = "repeat-01"
TASK_DIR = Path("languages/repeat-01")
N_PAIRS_PER_TERTILE = 25
DEGENERATE_GAP_THRESHOLD = 0.05
FLOOR_THRESHOLD = 0.10
ABOVE_CHANCE_THRESHOLD = 0.60
MAGNITUDE_SCALING_RATIO = 2.0

CHECKPOINT_DIRS = {
    ("rnn", 3): Path("data/models/repeat-01/rnn/rec+ns/validation-short/3"),
    ("rnn", 4): Path("data/models/repeat-01/rnn/rec+ns/validation-short/4"),
    ("lstm", 1): Path("data/models/repeat-01/lstm/rec+ns/validation-short/1"),
    ("lstm", 2): Path("data/models/repeat-01/lstm/rec+ns/validation-short/2"),
    ("transformer", 4): Path("data/models/repeat-01/transformer/rec+ns/validation-short/4"),
    ("transformer", 3): Path("data/models/repeat-01/transformer/rec+ns/validation-short/3"),
    ("mamba", 1): Path("models/repeat-01/mamba/rec+ns/validation-short/1"),
    ("mamba", 3): Path("models/repeat-01/mamba/rec+ns/validation-short/3"),
}


def get_genuinely_hard_population():
    seqs, labels = load_split("test")
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    def passes_all(seq):
        return all(fn(seq) for fn in SHORTCUT_FEATURES.values())

    hard_idx = [i for i in neg_idx if passes_all(seqs[i])]
    assert all(classify_violation(seqs[i]) == "alternation_mismatch" for i in hard_idx)
    return [(seqs[i], labels[i]) for i in hard_idx]


def get_long_positive_control():
    seqs, labels = load_split("test")
    lens = sorted(len(s) for s, l in zip(seqs, labels) if l == 1)
    threshold = lens[len(lens) * 3 // 4] if lens else 30  # 75th percentile of positive lengths
    return [(s, l) for s, l in zip(seqs, labels) if l == 1 and len(s) >= threshold], threshold


def load_model(arch, model_dir):
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    iface = RecognitionModelInterface()
    iface.add_arguments(parser)
    iface.add_forward_arguments(parser)
    tmp = tempfile.mkdtemp(prefix="repeat01_condition3_")
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


def classify_reading(acc, gaps_by_tertile):
    low, mid, high = gaps_by_tertile["low"], gaps_by_tertile["mid"], gaps_by_tertile["high"]
    above_chance = acc >= ABOVE_CHANCE_THRESHOLD
    floor = acc <= FLOOR_THRESHOLD
    all_causal = all(g >= DEGENERATE_GAP_THRESHOLD for g in (low, mid, high))
    any_causal = any(g >= DEGENERATE_GAP_THRESHOLD for g in (low, mid, high))
    uniform_large = all_causal and (max(low, mid, high) <= MAGNITUDE_SCALING_RATIO * max(min(low, mid, high), 1e-9))
    early_gt_late = any_causal and low >= MAGNITUDE_SCALING_RATIO * max(high, 1e-9)
    if above_chance and uniform_large:
        return "READING_1_comprehensive_per_position_verification"
    if above_chance and any_causal:
        return "READING_2_partial_verification_position_dependent"
    if floor and not any_causal:
        return "READING_3_floor_effect_confounded_shortcut_cluster"
    return "UNCLASSIFIED"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    pairs_by_tertile = {}
    audits_by_tertile = {}
    for i, tertile in enumerate(TERTILES):
        rng = np.random.default_rng(3 + i)
        tcp = target_computation_pairs(N_PAIRS_PER_TERTILE, rng, fixed_tertile=tertile)
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
    # direct-accuracy check on the 115-example REAL genuinely-shortcut-
    # blind population, per cell
    # ------------------------------------------------------------------
    hard_pop = get_genuinely_hard_population()
    print(f"\n=== direct-accuracy check on the {len(hard_pop)}-example real genuinely-shortcut-blind "
          f"population ===", flush=True)
    hard_accuracy = {}
    loaded_models = {}
    for arch, seed, role in CELLS:
        model_dir = CHECKPOINT_DIRS[(arch, seed)]
        iface, saver = load_model(arch, model_dir)
        loaded_models[(arch, seed)] = (iface, saver)
        acc, n = accuracy_on(iface, saver, tok2idx, hard_pop)
        hard_accuracy[f"{arch}_seed{seed}_{role}"] = {"arch": arch, "seed": seed, "role": role, "accuracy": acc, "n": n}
        print(f"  {arch} seed{seed} ({role}): accuracy={acc:.3f} n={n}", flush=True)

    # ------------------------------------------------------------------
    # long-positive control check for any cell with unusually high
    # (>=0.90) accuracy on the all-negative hard subset
    # ------------------------------------------------------------------
    long_pos_control, long_threshold = get_long_positive_control()
    print(f"\n=== long-positive control check (threshold len>={long_threshold}, n={len(long_pos_control)}) "
          f"for any cell with >=0.90 accuracy on the genuinely-hard subset ===", flush=True)
    control_checks = {}
    for key, info in hard_accuracy.items():
        if info["accuracy"] >= 0.90:
            arch, seed = info["arch"], info["seed"]
            iface, saver = loaded_models[(arch, seed)]
            acc_pos, n_pos = accuracy_on(iface, saver, tok2idx, long_pos_control)
            confounded = acc_pos < 0.50
            control_checks[key] = {
                "genuinely_hard_subset_accuracy": info["accuracy"],
                "long_positive_control_accuracy": acc_pos, "n_long_positive": n_pos,
                "blanket_rejection_confound": confounded,
                "verdict": (
                    "CONFOUNDED -- rejects most long positives regardless of content; high hard-subset "
                    "accuracy is a blanket-rejection artifact, NOT genuine verification."
                    if confounded else
                    "NOT CONFOUNDED -- accepts long positives at a reasonable rate; high hard-subset "
                    "accuracy is not explained by blanket long-sequence rejection."
                ),
            }
            print(f"  {key}: hard_acc={info['accuracy']:.3f} long_positive_acc={acc_pos:.3f} "
                  f"{'[CONFOUNDED]' if confounded else '[not confounded]'}", flush=True)
    if not control_checks:
        print("  no cell scored >=0.90 on the genuinely-hard subset -- no control check needed", flush=True)

    # ------------------------------------------------------------------
    # explicit reading classification per cell
    # ------------------------------------------------------------------
    readings = {}
    for arch, seed, role in CELLS:
        arch_cells = {c["pair_type"].rsplit("_", 1)[-1]: c for c in cells if c["arch"] == arch and c["seed"] == seed}
        gaps_by_tertile = {t: arch_cells[t]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"] for t in TERTILES}
        acc = hard_accuracy[f"{arch}_seed{seed}_{role}"]["accuracy"]
        key = f"{arch}_seed{seed}_{role}"
        reading = classify_reading(acc, gaps_by_tertile)
        if key in control_checks and control_checks[key]["blanket_rejection_confound"]:
            reading = "READING_CONFOUNDED_blanket_rejection_not_genuine_verification"
        readings[key] = {
            "arch": arch, "seed": seed, "role": role,
            "genuinely_hard_subset_accuracy": acc,
            "gaps_by_tertile": gaps_by_tertile,
            "reading": reading,
        }
    print("\n=== reading classification per cell ===", flush=True)
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
            "Tests genuine per-position alternation verification: clean/corrupt pairs hold all four "
            "shortcut features fixed (length parity, first/last symbol, count balance), differing "
            "ONLY via an adjacent swap breaking alternation. Swept EARLY/MID/LATE tertile by swap "
            "position, N=25 per tertile per cell. Position has NO magnitude interpretation for this "
            "task -- early>late reflects the natural early-concentrated violation distribution "
            "(58/115 in the first quintile), not magnitude sensitivity. ALSO runs a direct-accuracy "
            "check on the 115-example real genuinely-shortcut-blind population (NOT viol_repeat_01's "
            "151-example hard set), and a long-positive control for any cell scoring >=0.90 on that "
            "all-negative subset."
        ),
        "n_pairs_per_tertile": N_PAIRS_PER_TERTILE,
        "pair_construction_audits_by_tertile": audits_by_tertile,
        "sample_pairs_by_tertile": {t: pairs_by_tertile[t][:2] for t in TERTILES},
        "cells": cells,
        "genuinely_hard_subset_n": len(hard_pop),
        "genuinely_hard_subset_accuracy_per_cell": hard_accuracy,
        "long_positive_control_threshold_length": long_threshold,
        "long_positive_control_n": len(long_pos_control),
        "long_positive_control_checks": control_checks,
        "reading_classification": readings,
        "reading_distribution": reading_counts,
    }
    out_path = RESULTS / "phase3_repeat01_condition3.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
