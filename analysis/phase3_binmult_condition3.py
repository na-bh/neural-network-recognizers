"""Binary-multiplication Phase 3, condition 3 of 3: target-computation null
test (the load-bearing test) on the value_mismatch_only residual. clean/
corrupt pairs hold all five shortcut features fixed (operator/equals
counts, well-formedness, length-sufficiency, lsb-consistency -- structurally
guaranteed since only one u_z bit is touched, excluding index 0 to keep
lsb_consistent fixed) and differ ONLY in one flipped bit of u_z, producing a
wrong product. Swept LOW/MID/HIGH tertile by flipped-bit index within u_z --
LSB-first: LOW = low-order/small-magnitude, HIGH = high-order/large-
magnitude. N=25 PER TERTILE per cell (75 pairs per cell, 8 cells).

BEFORE interpreting the causal (patching) results, this ALSO runs a direct-
accuracy check (bypassing patching entirely) on the exact 323-example
value_mismatch_only shortcut-blind population from the REAL test set
(phase3_binmult_task_audit.json's joint residual) for each of the 8 cells.
Per the user's explicit three-way framing:
  READING 1 -- above-chance real-subset accuracy AND POSITION-INDEPENDENT
    Condition 3 gaps: bit-exact position-general verification. Would be
    the FIRST such finding in this pilot.
  READING 2 -- position-scaling gaps (late > early, matching binary-
    addition/compute-sqrt Mamba-secondary): magnitude sensitivity,
    extending that finding to a THIRD arithmetic task.
  READING 3 -- floor-effect (near-zero real-subset accuracy AND near-zero
    Condition 3 gaps across all tertiles): confounded-shortcut cluster
    pattern, matching missing-duplicate-string/stack-manipulation.
Any cell scoring unusually high (>=0.90) on the all-negative 323-example
subset is checked against a REAL LONG-POSITIVE control before being
accepted as genuine verification (matching the stack-manipulation/compute-
sqrt Transformer catches).

PRIMARY evidence for the causal test is the behavioral logit gap; full-
state patch is a SECONDARY wiring check, flagged UNRELIABLE in near-zero-
gap cells.

Saves analysis_outputs/final_results/phase3_binmult_condition3.json.

PYTHONPATH=src:analysis python analysis/phase3_binmult_condition3.py
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
from phase3_binmult_counterfactual_design import target_computation_pairs, audit_target_computation_pairs, TERTILES, SHORTCUT_FEATURES
from phase3_binmult_patching_common import CELLS, get_tok2idx, run_cell, summarize
from phase3_binmult_task_audit import classify_violation, load_split
from recognizers.neural_networks.data import add_data_arguments, load_vocabulary_data
from recognizers.neural_networks.model_interface import RecognitionModelInterface, ModelInput

RESULTS = Path("analysis_outputs/final_results")
TASK = "binary-multiplication"
TASK_DIR = Path("languages/binary-multiplication")
N_PAIRS_PER_TERTILE = 25
DEGENERATE_GAP_THRESHOLD = 0.05
FLOOR_THRESHOLD = 0.10
ABOVE_CHANCE_THRESHOLD = 0.60
MAGNITUDE_SCALING_RATIO = 2.0

CHECKPOINT_DIRS = {
    ("rnn", 4): Path("data/models/binary-multiplication/rnn/rec+ns/validation-short/4"),
    ("rnn", 2): Path("data/models/binary-multiplication/rnn/rec+ns/validation-short/2"),
    ("lstm", 8): Path("data/models/binary-multiplication/lstm/rec+ns/validation-short/8"),
    ("lstm", 2): Path("data/models/binary-multiplication/lstm/rec+ns/validation-short/2"),
    ("transformer", 9): Path("data/models/binary-multiplication/transformer/rec+ns/validation-short/9"),
    ("transformer", 4): Path("data/models/binary-multiplication/transformer/rec+ns/validation-short/4"),
    ("mamba", 4): Path("models/binary-multiplication/mamba/rec+ns/validation-short/4"),
    ("mamba", 8): Path("models/binary-multiplication/mamba/rec+ns/validation-short/8"),
}


def get_genuinely_hard_population():
    seqs, labels = load_split("test")
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    def passes_all(seq):
        return all(fn(seq) for fn in SHORTCUT_FEATURES.values())

    hard_idx = [i for i in neg_idx if passes_all(seqs[i])]
    assert all(classify_violation(seqs[i]) == "value_mismatch_only" for i in hard_idx)
    return [(seqs[i], labels[i]) for i in hard_idx]


def get_long_positive_control():
    seqs, labels = load_split("test")
    hard_pop_lens = sorted(len(s) for s, l in zip(seqs, labels) if l == 0 and
                            all(fn(s) for fn in SHORTCUT_FEATURES.values()))
    threshold = hard_pop_lens[len(hard_pop_lens) // 2] if hard_pop_lens else 30  # median length of the hard subset
    return [(s, l) for s, l in zip(seqs, labels) if l == 1 and len(s) >= threshold], threshold


def load_model(arch, model_dir):
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    iface = RecognitionModelInterface()
    iface.add_arguments(parser)
    iface.add_forward_arguments(parser)
    tmp = tempfile.mkdtemp(prefix="binmult_condition3_")
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
    position_independent = all_causal and (max(low, mid, high) <= MAGNITUDE_SCALING_RATIO * max(min(low, mid, high), 1e-9))
    magnitude_scaling = any_causal and high >= MAGNITUDE_SCALING_RATIO * max(low, 1e-9)
    if above_chance and position_independent:
        return "READING_1_bit_exact_position_general_verification"
    if magnitude_scaling:
        return "READING_2_magnitude_scaling_third_arithmetic_task"
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
        assert all(p["flip_position_tertile"] == tertile for p in tcp)
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
    # direct-accuracy check on the 323-example REAL value_mismatch_only
    # shortcut-blind population, per cell
    # ------------------------------------------------------------------
    hard_pop = get_genuinely_hard_population()
    print(f"\n=== direct-accuracy check on the {len(hard_pop)}-example real value_mismatch_only "
          f"shortcut-blind population ===", flush=True)
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
            "Tests genuine multiplication-value-sensitivity: clean/corrupt pairs hold all five "
            "shortcut features fixed (operator/equals counts, well-formedness, length-sufficiency, "
            "lsb-consistency), differing ONLY via a single flipped bit in u_z (excluding index 0, "
            "the LSB, to preserve lsb_consistent). Swept LOW/MID/HIGH tertile by bit index, N=25 "
            "per tertile per cell. ALSO runs a direct-accuracy check on the 323-example real "
            "value_mismatch_only shortcut-blind population, and a long-positive control for any "
            "cell scoring >=0.90 on that all-negative subset."
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
    out_path = RESULTS / "phase3_binmult_condition3.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
