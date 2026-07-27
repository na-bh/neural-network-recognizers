"""Phase 3 final deliverable: consolidates conditions 1-3 plus the
disambiguation step into a single per-cell mechanism summary with causal
evidence, closing the marked-reversal pilot's causal-patching layer.

PYTHONPATH=src:analysis python analysis/phase3_causal_patching_summary.py
"""

import json
from pathlib import Path

RESULTS = Path("analysis_outputs/final_results")


def load(name):
    return json.loads((RESULTS / name).read_text())


def main():
    cond1 = load("phase3_causal_patching.json")
    cond2 = load("phase3_causal_patching_condition2.json")
    disamb = load("phase3_disambiguation.json")
    cond3 = load("phase3_condition3_balanced_content_match.json")
    design = load("phase3_counterfactual_design.json")

    mc_gaps = {(c["arch"], c["seed"]): c["mean_abs_clean_corrupt_logit_gap"]
               for c in cond1["cells"] if c["pair_type"] == "marker_count"}
    mp_gaps = {(c["arch"], c["seed"]): c["mean_abs_clean_corrupt_logit_gap"]
               for c in cond1["cells"] if c["pair_type"] == "marker_position"}
    lp_cells = {(c["arch"], c["seed"]): c for c in cond2["cells"]}
    bal_cells = {(c["arch"], c["seed"]): c for c in disamb["cells"]}
    bcm_cells = {(c["arch"], c["seed"]): c for c in cond3["cells"]}
    mechanism_attr = disamb["mechanism_attribution_by_cell"]

    rnn_design_gap = design["marked_reversal_audit"]["behavioral_gap_check"]

    cells_out = []
    for arch, seeds in {"rnn": [3, 7], "lstm": [10, 1], "transformer": [9, 1], "mamba": [4, 6]}.items():
        for seed in seeds:
            key = f"{arch}_seed{seed}"
            attr = mechanism_attr[key]
            bcm = bcm_cells[(arch, seed)]
            lp = lp_cells.get((arch, seed))
            bal = bal_cells.get((arch, seed))

            evidence = {
                "condition1_marker_count_gap": mc_gaps.get((arch, seed)),
                "condition1_marker_position_gap": mp_gaps.get((arch, seed)),
                "condition2_length_parity_gap": lp["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"] if lp else None,
                "condition2_length_parity_layer_specific_rf": (
                    lp["layer_specific_patch_at_best_layer"]["layer_specific_restored_fraction_mean"]
                    if lp and lp["layer_specific_rf_is_meaningful"] else None
                ),
                "disambiguation_balance_pairs_gap": (
                    bal["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"] if bal
                    else (rnn_design_gap["balance_pairs_mean_abs_logit_gap"] if arch == "rnn" and seed == 3 else None)
                ),
                "condition3_balanced_content_match_gap": bcm["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"],
                "condition3_content_sensitivity_surprising": bcm["SURPRISING_above_noise_content_sensitivity"],
                "condition3_layer_specific_rf_if_meaningful": (
                    bcm["layer_specific_patch_at_best_layer"]["layer_specific_restored_fraction_mean"]
                    if bcm["layer_specific_rf_is_meaningful"] else None
                ),
            }

            cells_out.append({
                "task": "marked-reversal", "arch": arch, "seed": seed,
                "identified_mechanism": attr["identified_mechanism"],
                "causal_evidence": evidence,
                "verifies_mirror_content_as_primary_mechanism": False,
                "faint_genuine_content_trace_detected": bcm["SURPRISING_above_noise_content_sensitivity"],
            })

    surprising_cells = [c for c in cells_out if c["faint_genuine_content_trace_detected"]]

    summary_narrative = {
        "primary_result": (
            "Condition 3 (balanced_content_match, the corrected/confound-free content-verification "
            "target) shows near-zero behavioral effect for 7 of 8 cells, closing the marked-reversal "
            "pilot's causal-patching layer: no architecture-seed pair uses genuine mirror-content "
            "verification as its PRIMARY decision mechanism. Every cell's accept/reject decision is "
            "explained by some combination of marker_count, marker_position, is_balanced (marker-"
            "centering), and -- for rnn and transformer seed1 specifically -- sequence-length parity. "
            "This is now established with three converging lines of evidence per architecture: Phase 1 "
            "behavioral hard-negative accuracy, Phase 2 corrected representational probes (structural-"
            "decomposition-audited), and Phase 3 causal patching (behavioral counterfactual gap + "
            "layer-specific restoration)."
        ),
        "exception_not_smoothed_over": (
            "lstm seed1 is the ONE cell with a statistically real (CI95 excludes zero), reproducible "
            "(consistent across all 25 pairs, layer-specific rf=1.0012 +/- 0.0035) but SMALL "
            "balanced_content_match effect (gap=0.112, vs. its own marker_count/position/balance gaps "
            "of 4.2-5.7 -- roughly 2 orders of magnitude smaller). This is fully localized to layer 4. "
            "Interpretation: lstm seed1 carries a faint but genuine mirror-content-verification trace "
            "on top of its dominant structural mechanism (marker_count + marker_position + is_balanced) "
            "-- consistent with the paper's carrier account already documented for cycle-navigation's "
            "position-residual finding and the (pre-correction) Mamba full_reversal_match uptick: "
            "non-Mamba, non-canonical-shortcut architectures can retain a small trace of the 'genuine' "
            "computation even when the primary decision path is a shortcut. This does NOT change lstm "
            "seed1's identified PRIMARY mechanism (still structural), but it is the most trustworthy "
            "'faint genuine signal' finding in this investigation, since it survived the full "
            "structural-decomposition confound audit that invalidated the ORIGINAL (pre-correction) "
            "full_reversal_match-based claims about rnn."
        ),
        "lstm_seed10_marker_count_only_final_status": (
            "CONFIRMED across FOUR independent null counterfactuals (marker_position 0.0065, "
            "length_parity 0.0014, balance 0.0021, balanced_content_match 0.0000) against one large "
            "positive (marker_count 8.59). This is the simplest identified mechanism in the entire "
            "pilot (both tasks, all architectures)."
        ),
        "cross_cell_mechanism_diversity": (
            "8 cells resolve into 5 distinct mechanism classes: (1) marker_count only [lstm seed10]; "
            "(2) marker_count + marker_position, no refinement [transformer seed9]; (3) marker_count + "
            "length_parity [rnn both seeds, transformer seed1]; (4) marker_count + marker_position + "
            "is_balanced/marker-centering [lstm seed1, mamba both seeds] (lstm seed1 additionally shows "
            "the faint content trace above). No cell uses marker_position alone without marker_count "
            "(marker_count is universal -- every architecture/seed's largest or tied-largest gap). This "
            "is a substantially richer picture than the corrected P4's architecture-level summary "
            "(\"rnn: length parity; others: marker count/position\") -- length-parity sensitivity is "
            "present in 3/8 cells spanning 2 architectures (rnn, transformer), and marker-centering "
            "sensitivity is present in 3/8 cells spanning 2 architectures (lstm, mamba) -- both cut "
            "across the architecture boundary at the seed level, not the architecture level."
        ),
    }

    out = {
        "task": "marked-reversal",
        "description": "Phase 3 final deliverable: per-cell causal mechanism identification for all "
                       "8 architecture-seed cells, consolidating condition 1 (marker features), "
                       "condition 2 (length parity), the balance_pairs disambiguation step, and "
                       "condition 3 (balanced_content_match null test).",
        "cells": cells_out,
        "condition3_all_cells_null": cond3["all_cells_null"],
        "condition3_surprising_cells": [f"{c['arch']}_seed{c['seed']}" for c in surprising_cells],
        "summary_narrative": summary_narrative,
        "source_files": [
            "phase3_causal_patching.json (condition 1: marker features)",
            "phase3_causal_patching_condition2.json (condition 2: length parity)",
            "phase3_disambiguation.json (balance_pairs disambiguation)",
            "phase3_condition3_balanced_content_match.json (condition 3: null test)",
            "phase3_counterfactual_design.json (design step, rnn seed3 balance_pairs baseline)",
        ],
    }
    out_path = RESULTS / "phase3_causal_patching_summary.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved {out_path}")

    print("\n=== per-cell final mechanism ===")
    for c in cells_out:
        flag = "  [+faint content trace]" if c["faint_genuine_content_trace_detected"] else ""
        print(f"  {c['arch']} seed{c['seed']}: {c['identified_mechanism']}{flag}")


if __name__ == "__main__":
    main()
