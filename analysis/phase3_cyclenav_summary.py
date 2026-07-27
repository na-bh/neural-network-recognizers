"""Cycle-navigation Phase 3 final deliverable: consolidates conditions 1-2
into a per-cell causal summary, closing the pilot's second (and final)
causal-patching layer.

PYTHONPATH=src:analysis python analysis/phase3_cyclenav_summary.py
"""

import json
from pathlib import Path

RESULTS = Path("analysis_outputs/final_results")


def load(name):
    return json.loads((RESULTS / name).read_text())


def main():
    cond1 = load("phase3_cyclenav.json")
    cond2 = load("phase3_cyclenav_condition2.json")

    c1_by_cell = {(c["arch"], c["seed"]): c for c in cond1["cells"]}
    c2_by_cell = {(c["arch"], c["seed"]): c for c in cond2["cells"]}

    cells_out = []
    for arch, seeds in {"rnn": [1, 4], "lstm": [2, 3], "transformer": [6, 9], "mamba": [0, 7]}.items():
        for seed in seeds:
            c1 = c1_by_cell[(arch, seed)]
            c2 = c2_by_cell[(arch, seed)]
            is_mamba7_exception = (arch == "mamba" and seed == 7)

            if is_mamba7_exception:
                mechanism = ("PARTIAL GENUINE POSITION-TRACKING, not pure shortcut -- long_seq_"
                             "accuracy 0.8284 (below the 0.95 solving threshold and the 0.9231 "
                             "shortcut ceiling every other cell hits), 84% hard-negative accuracy on "
                             "single-move-flip adversarial pairs (vs 0% for every other cell), and a "
                             "statistically real (if small, ratio=0.011 to condition1 reference) "
                             "causal effect from the true_position probe-direction subspace patch. "
                             "This cell does not fit the 'uniform shortcut' story and should not be "
                             "treated as directly comparable to the other 7 cells.")
            else:
                mechanism = ("symbol_counts shortcut, causally confirmed (behavioral gap "
                             f"{c1['PRIMARY_behavioral_logit_gap']['mean_abs_clean_corrupt_logit_gap']:.2f}, "
                             "layer-specific rf=1.0000); true_position decodable in Phase 2 but NOT "
                             "causally used at readout (subspace-patch effect ratio to condition1 "
                             f"reference = {c2['subspace_effect_ratio_to_condition1_reference']:.4f}, "
                             "consistent with null)")

            cells_out.append({
                "task": "cycle-navigation", "arch": arch, "seed": seed,
                "identified_mechanism": mechanism,
                "condition1_symbol_counts_behavioral_gap": c1["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"],
                "condition1_layer_specific_rf": c1["layer_specific_patch_at_canonical_layer"]["layer_specific_restored_fraction_mean"],
                "condition2_true_position_subspace_abs_delta": c2["true_position_subspace_patch_PRIMARY_METRIC"]["mean_abs_delta_logit_subspace_patch"],
                "condition2_ratio_to_condition1_reference": c2["subspace_effect_ratio_to_condition1_reference"],
                "condition2_surprising": c2["SURPRISING_above_noise_true_position_causal_effect"],
                "is_exception_to_uniform_shortcut_story": is_mamba7_exception,
            })

    summary_narrative = {
        "condition1_result": (
            "symbol_counts is decisively causal for all 8 cells: large behavioral gaps (3.55-9.42) and "
            "PERFECT layer-specific restoration (rf=1.0000) at each architecture's established Phase 2 "
            "site -- including, for rnn/lstm, confirming that layer 0 ALONE (not the full 5-layer "
            "state) is causally sufficient. No exceptions, no ambiguity."
        ),
        "condition2_result": (
            "true_position is NOT causally used at readout for 7 of 8 cells (subspace-patch effect "
            "ratio to condition1's reference scale: 0.0000-0.0006) -- the residual Phase 2 "
            "representational signal (0.037-0.072 selectivity) is a decodable-but-causally-inert "
            "byproduct for these cells, cleanly closing the shortcut-vs-target story with a NEW "
            "subtype of the paper's decodable-not-causal finding (here: the feature is genuinely "
            "represented, MLP-confirmed, yet plays no role in the readout decision -- the model "
            "computes and uses a DIFFERENT feature instead)."
        ),
        "mamba_seed7_exception": (
            "mamba seed7 is a genuine, investigated exception: NOT a null-test violation of the "
            "'decodable but unused' pattern, but direct evidence of a qualitatively different "
            "solution -- partial genuine position-tracking, evidenced independently by hard-negative "
            "accuracy (84% vs 0-0.9% everywhere else), long-sequence accuracy (0.8284, below the 0.95 "
            "solving threshold used elsewhere in this pilot), and a statistically real causal "
            "subspace-patch effect. This mirrors marked-reversal's seed-dependent-mechanism story in a "
            "task previously believed to be a clean, uniform, exception-free shortcut case -- the "
            "paper's 'uniform cycle-navigation' claim needs a seed-level caveat, and seed7's inclusion "
            "in cross-cell comparisons (Phase 2 P2, the original P4) should be flagged since it does "
            "not meet the architecture's own solving-seed threshold."
        ),
    }

    out = {
        "task": "cycle-navigation",
        "description": "Phase 3 final deliverable: per-cell causal mechanism identification for all 8 "
                       "architecture-seed cells, consolidating condition 1 (symbol_counts) and "
                       "condition 2 (true_position null test, resolved via probe-direction subspace "
                       "patching due to a mathematically-necessary construction confound).",
        "cells": cells_out,
        "condition1_prediction_confirmed": cond1["prediction_confirmed"],
        "condition2_prediction_confirmed": cond2["prediction_confirmed"],
        "condition2_exception_cells": cond2["any_surprising_cells"],
        "summary_narrative": summary_narrative,
        "source_files": [
            "phase3_cyclenav.json (condition 1: symbol_counts)",
            "phase3_cyclenav_condition2.json (condition 2: true_position null test + mamba seed7 investigation)",
        ],
    }
    out_path = RESULTS / "phase3_cyclenav_summary.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved {out_path}")

    print("\n=== per-cell final mechanism ===")
    for c in cells_out:
        flag = "  [EXCEPTION]" if c["is_exception_to_uniform_shortcut_story"] else ""
        print(f"  {c['arch']} seed{c['seed']}: {c['identified_mechanism'][:90]}...{flag}")


if __name__ == "__main__":
    main()
