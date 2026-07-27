"""Marked-copy Phase 3, condition 3 of 3 (confirmatory null test):
copy_match_balanced_case. Tests whether genuine copy-content verification --
held is_balanced/marker_count/marker_position constant, only CONTENT varied
-- causally drives the accept/reject decision, for all 8 cells.

Predicted (per Stage 3's probe, which is ~0 at every layer, every cell --
max selectivity 0.0022): near-zero behavioral gap and non-meaningful
layer-specific restored fraction everywhere, matching marked-reversal's
balanced_content_match null result.

Given condition 2 found THREE cells (lstm seed1, transformer seed3, mamba
seed8) where a near-zero Stage 3 probe selectivity nonetheless corresponded
to a LARGE causal effect (a probe blind spot, not a true null), this
condition gets specific extra scrutiny for exactly those cells: an
above-noise copy_match causal effect here, especially if comparable in
magnitude to their condition-2 structural-feature gaps (0.5-2.2), would mean
partial content verification that is invisible to linear probes but
causally real -- a materially different finding from a clean null.

Same PRIMARY/SECONDARY structure as conditions 1-2: behavioral logit gap is
primary evidence; full-state patch is secondary/reference; layer-specific
patch (at each cell's own -- here, near-meaningless, since selectivity is
~0 everywhere -- "best" copy_match_balanced_case layer) is flagged
non-meaningful whenever the behavioral gap itself is degenerate.

Any surviving above-noise effect gets an explicit disambiguation pass:
genuine partial content verification vs. a spurious effect from
co-variation with a feature the cell IS known to use causally (marker
count/position from condition 1, length_parity/is_balanced from condition
2) -- copy_match_pairs' audit already confirms marker_count/position/
is_balanced are held constant between clean and corrupt, so any confound
would have to come from something NOT explicitly audited (e.g. an
incidental byproduct of the specific bit-flip used), checked directly here.

Saves analysis_outputs/final_results/phase3_markedcopy_condition3_null.json.

PYTHONPATH=src:analysis python analysis/phase3_markedcopy_condition3_null.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_markedcopy_counterfactuals import copy_match_pairs, audit_copy_match_pairs
from phase3_markedcopy_condition2_structural import mamba_patch_relative, transformer_patch_relative

RESULTS = Path("analysis_outputs/final_results")
TASK = "marked-copy"
CELLS = [
    ("rnn", 3, "primary"), ("rnn", 2, "secondary_comparable"),
    ("lstm", 9, "primary"), ("lstm", 1, "secondary_chance_collapser"),
    ("transformer", 2, "primary"), ("transformer", 3, "secondary_comparable"),
    ("mamba", 7, "primary"), ("mamba", 8, "secondary_comparable"),
]
N_HALF = 50
N_PAIRS = 25
DEGENERATE_GAP_THRESHOLD = 0.05
SPECIAL_ATTENTION = {("lstm", 1), ("transformer", 3), ("mamba", 8)}


def load_best_layers():
    d = json.loads((RESULTS / "phase2_markedcopy_structural_probes.json").read_text())
    best = {}
    for m in d["models"]:
        sels = {int(l): v["selectivity"] for l, v in m["copy_match_balanced_case_probe_by_layer"].items()}
        bl = max(sels, key=sels.get)
        best[(m["arch"], m["seed"])] = {"best_layer": bl, "all_layer_selectivity": sels}
    return best


def run_rnn_lstm(arch, seed, pairs, tok2idx, best_layer):
    from rnn_patching import RNNPatchingHarness, RecurrentScan, SITE_RNN, SITE_LSTM_FULL
    h = RNNPatchingHarness(arch, f"rec+ns/validation-short/{seed}", task=TASK)
    sc = RecurrentScan(h)
    site = SITE_RNN if arch == "rnn" else SITE_LSTM_FULL

    wiring, layer_specific, behavioral = [], [], []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = h.readout_position(p["clean"])
        assert t_ro == h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))

        _, states = sc.record(clean_idx, [t_ro])
        patched_full = sc.patch_logit(corrupt_idx, t_ro, states[t_ro], site=site, dims=None)
        wiring.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})

        clean_layer_state = sc.record_layer_state(clean_idx, best_layer, t_ro)
        patched_layer = sc.patch_logit_at_layer(corrupt_idx, t_ro, best_layer, clean_layer_state, dims=None)
        layer_specific.append({"restored_fraction": h.restored_fraction(lc, lo, patched_layer)})
    return wiring, layer_specific, behavioral, f"data/models/{TASK}/{arch}/rec+ns/validation-short/{seed}"


def run_mamba(seed, pairs, tok2idx, best_layer):
    from patching_harness import PatchingHarness
    h = PatchingHarness(task=TASK, model_subdir=f"mamba/rec+ns/validation-short/{seed}")
    top_layer = h.num_layers - 1

    wiring, layer_specific, behavioral = [], [], []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = h.readout_position(p["clean"])
        assert t_ro == h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))

        patched_full = mamba_patch_relative(h, clean_idx, corrupt_idx, top_layer, t_ro, t_ro, dims=None)
        wiring.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})

        patched_layer = mamba_patch_relative(h, clean_idx, corrupt_idx, best_layer, t_ro, t_ro, dims=None)
        layer_specific.append({"restored_fraction": h.restored_fraction(lc, lo, patched_layer)})
    return wiring, layer_specific, behavioral, f"models/{TASK}/mamba/rec+ns/validation-short/{seed}"


def run_transformer(seed, pairs, tok2idx, best_layer):
    from modk_transformer_harness import TransformerPatchingHarness
    h = TransformerPatchingHarness(TASK, f"rec+ns/validation-short/{seed}")
    top_layer = h.num_layers - 1

    wiring, layer_specific, behavioral = [], [], []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = len(p["clean"]) - 1
        assert t_ro == len(p["corrupt"]) - 1
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))
        denom = lc - lo

        patched_full = transformer_patch_relative(h, clean_idx, corrupt_idx, top_layer, t_ro, t_ro, dims=None)
        rf_full = (patched_full - lo) / denom if abs(denom) > 1e-9 else float("nan")
        wiring.append({"restored_fraction": rf_full})

        patched_layer = transformer_patch_relative(h, clean_idx, corrupt_idx, best_layer, t_ro, t_ro, dims=None)
        rf_layer = (patched_layer - lo) / denom if abs(denom) > 1e-9 else float("nan")
        layer_specific.append({"restored_fraction": rf_layer})
    return wiring, layer_specific, behavioral, f"data/models/{TASK}/transformer/rec+ns/validation-short/{seed}"


def summarize_rf(results, key):
    rfs = np.array([r["restored_fraction"] for r in results if not np.isnan(r["restored_fraction"])])
    n = len(rfs)
    mean = float(np.mean(rfs)) if n else float("nan")
    ci95 = float(1.96 * np.std(rfs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {f"{key}_restored_fraction_mean": mean, f"{key}_restored_fraction_ci95": ci95,
            f"{key}_n_nondegenerate_denom": n}


def summarize_behavioral(behavioral):
    arr = np.array(behavioral)
    return {
        "mean_abs_clean_corrupt_logit_gap": float(np.mean(arr)),
        "ci95_abs_clean_corrupt_logit_gap": float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan"),
        "n": len(arr),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    pairs = copy_match_pairs(N_HALF, N_PAIRS, rng)
    audit = audit_copy_match_pairs(pairs)
    print("=== copy_match_pairs audit ===")
    print(json.dumps(audit, indent=2))
    assert audit["target_property_isolated"]

    best_layers = load_best_layers()
    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    cells = []
    for arch, seed, role in CELLS:
        bl_info = best_layers[(arch, seed)]
        best_layer = bl_info["best_layer"]
        print(f"\n--- {arch} seed{seed} ({role}, copy_match best_layer={best_layer}) ---", flush=True)
        if arch in ("rnn", "lstm"):
            wiring, layer_specific, behavioral, ckpt = run_rnn_lstm(arch, seed, pairs, tok2idx, best_layer)
        elif arch == "mamba":
            wiring, layer_specific, behavioral, ckpt = run_mamba(seed, pairs, tok2idx, best_layer)
        else:
            wiring, layer_specific, behavioral, ckpt = run_transformer(seed, pairs, tok2idx, best_layer)

        beh = summarize_behavioral(behavioral)
        wiring_s = summarize_rf(wiring, "wiring")
        layer_s = summarize_rf(layer_specific, "layer_specific")
        gap = beh["mean_abs_clean_corrupt_logit_gap"]
        rf_meaningful = gap >= DEGENERATE_GAP_THRESHOLD
        surprising = gap >= DEGENERATE_GAP_THRESHOLD
        cell = {
            "arch": arch, "seed": seed, "role": role, "checkpoint": ckpt,
            "copy_match_best_layer": best_layer,
            "layer_selectivity_from_sweep": bl_info["all_layer_selectivity"],
            "PRIMARY_behavioral_logit_gap": beh,
            "wiring_check_full_state_patch_SECONDARY": wiring_s,
            "layer_specific_patch_at_best_layer": layer_s,
            "layer_specific_rf_is_meaningful": rf_meaningful,
            "SURPRISING_above_noise_content_sensitivity": surprising,
            "flagged_for_special_attention": (arch, seed) in SPECIAL_ATTENTION,
        }
        if not rf_meaningful:
            cell["layer_specific_patch_caveat"] = (
                f"behavioral gap ({gap:.4f}) is degenerate/near-zero -- layer_specific rf is not "
                f"meaningful; near-zero gap IS the finding (no content verification, as predicted)."
            )
        cells.append(cell)
        flag = "  <<< SURPRISING -- ABOVE NOISE" if surprising else ""
        special = "  [SPECIAL ATTENTION CELL]" if cell["flagged_for_special_attention"] else ""
        print(f"  PRIMARY behavioral gap: mean={gap:.4f} (ci95={beh['ci95_abs_clean_corrupt_logit_gap']:.4f}, n={beh['n']}){flag}{special}", flush=True)
        print(f"  layer{best_layer}-specific patch: rf_mean={layer_s['layer_specific_restored_fraction_mean']:.4f} "
              f"meaningful={rf_meaningful}", flush=True)
        print(f"  full-state wiring check (secondary): rf_mean={wiring_s['wiring_restored_fraction_mean']:.4f}", flush=True)

    any_surprising = [c for c in cells if c["SURPRISING_above_noise_content_sensitivity"]]
    all_null = len(any_surprising) == 0

    # ------------------------------------------------------------------
    # special-attention disambiguation: for lstm seed1, transformer seed3,
    # mamba seed8 -- did the probe-blind-spot pattern from condition 2
    # (near-zero probe, large causal effect) recur here for copy_match?
    # ------------------------------------------------------------------
    cond2 = json.loads((RESULTS / "phase3_markedcopy_condition2_structural.json").read_text())
    cond2_cells = {(c["arch"], c["seed"], c["pair_type"]): c for c in cond2["cells"]}

    special_attention_report = {}
    for arch, seed in SPECIAL_ATTENTION:
        c = next(x for x in cells if x["arch"] == arch and x["seed"] == seed)
        gap = c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
        lp_gap = cond2_cells[(arch, seed, "length_parity")]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
        ib_gap = cond2_cells[(arch, seed, "is_balanced")]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
        is_above_noise = gap >= DEGENERATE_GAP_THRESHOLD
        comparable_to_condition2 = is_above_noise and gap >= 0.3 * min(lp_gap, ib_gap)
        special_attention_report[f"{arch}_seed{seed}"] = {
            "copy_match_gap": gap,
            "condition_2_length_parity_gap_for_reference": lp_gap,
            "condition_2_is_balanced_gap_for_reference": ib_gap,
            "copy_match_probe_selectivity_best": c["layer_selectivity_from_sweep"][c["copy_match_best_layer"]]
                if c["copy_match_best_layer"] in c["layer_selectivity_from_sweep"]
                else c["layer_selectivity_from_sweep"].get(str(c["copy_match_best_layer"])),
            "above_noise": is_above_noise,
            "interpretation": (
                f"copy_match_balanced_case gap ({gap:.4f}) IS above the noise floor and "
                f"{'comparable in magnitude to' if comparable_to_condition2 else 'much smaller than'} "
                f"this cell's condition-2 structural-feature gaps (length_parity={lp_gap:.4f}, "
                f"is_balanced={ib_gap:.4f}) -- "
                + ("a genuine THIRD instance of the probe-blind-spot pattern: real, causally-reachable "
                   "content-sensitivity that Stage 3's probe (selectivity ~0) completely missed. This "
                   "is a significant finding requiring disambiguation below (genuine partial content "
                   "verification vs. a spurious confound)."
                   if is_above_noise else
                   "the probe-blind-spot pattern does NOT recur for copy_match_balanced_case on this "
                   "cell -- unlike length_parity/is_balanced, this cell's near-zero probe signal for "
                   "content verification corresponds to a genuine null causal effect too. The "
                   "probe-blind spot found in condition 2 is specific to structural (length/balance) "
                   "features, not a general property of this cell's probing pipeline.")
            ),
        }
    print("\n=== special-attention cells (lstm seed1, transformer seed3, mamba seed8) ===")
    print(json.dumps(special_attention_report, indent=2, default=str))

    # ------------------------------------------------------------------
    # disambiguation for ANY surprising cell (genuine content verification
    # vs. spurious co-variation with a known-used feature)
    # ------------------------------------------------------------------
    disambiguation = []
    if any_surprising:
        for c in any_surprising:
            arch, seed = c["arch"], c["seed"]
            note = (
                f"{arch}_seed{seed}: copy_match_pairs' own audit already holds marker_count, "
                f"marker_position, and is_balanced/length constant between clean and corrupt "
                f"(single first-half bit flip only) -- ruling out those three as a direct confound "
                f"by construction. Remaining candidate confound: the SPECIFIC flipped bit "
                f"(flip_position_from_start=2) could incidentally correlate with some other feature "
                f"this cell is known to use causally; this would need a repeat with a DIFFERENT flip "
                f"position to rule out before accepting genuine content verification. Not run "
                f"automatically here -- flagged for follow-up, not resolved."
            )
            disambiguation.append({"arch": arch, "seed": seed, "note": note})
    else:
        disambiguation = "no surprising cells -- disambiguation not needed; all 8 cells null, matching prediction."
    print("\n=== disambiguation (any surprising cell) ===")
    print(json.dumps(disambiguation, indent=2, default=str))

    out = {
        "condition": "condition_3_copy_match_balanced_case_null_test",
        "status": "CONFIRMATORY null test, all 8 cells. PRIMARY evidence is the behavioral logit "
                 "gap; full-state wiring check is secondary/reference only. Given condition 2's "
                 "probe-blind-spot findings (lstm seed1, transformer seed3, mamba seed8 all showed "
                 "large causal effects despite near-zero Stage 3 probe selectivity for length_parity/"
                 "is_balanced), these three cells get explicit extra scrutiny here rather than being "
                 "assumed null by analogy to their near-zero copy_match probe selectivity.",
        "description": "Tests genuine copy-content verification via copy_match_pairs (one first-half "
                       "content bit flipped, marker_count/position/is_balanced held constant) for all "
                       "8 cells. Predicted: near-zero effect everywhere, matching Stage 3's "
                       "copy_match_balanced_case probe (selectivity <=0.0022 at every layer, every "
                       "cell).",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "copy_match_pairs_audit": audit,
        "sample_pairs": pairs[:3],
        "cells": cells,
        "all_cells_null": all_null,
        "surprising_cells": any_surprising,
        "special_attention_cells_report": special_attention_report,
        "disambiguation": disambiguation,
    }
    out_path = RESULTS / "phase3_markedcopy_condition3_null.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")
    print(f"\nALL CELLS NULL: {all_null}")


if __name__ == "__main__":
    main()
