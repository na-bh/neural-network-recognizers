"""Bucket-sort Phase 3, condition 3 follow-up: swap-POSITION sweep for
target_computation_multiset_matched_case. Condition 3 (phase3_bucketsort_
condition3.py) found large causal effects at ONE fixed swap position
(indices 76/77, the middle of the n_half=50 sorted half) for 7/8 cells --
this follow-up tests whether that generalizes to early/mid/late positions
within the sorted half, or was specific to the position tested.

Same construction as condition 3 (analysis/phase3_bucketsort_
counterfactuals.py's target_computation_pairs): clean = w#sorted(w);
corrupt = same length/marker-count/marker-position/is_balanced/multiset,
two ADJACENT sorted-half positions swapped -- only swap_idx varies here.
4 positions swept: early (5), early-mid (17), mid-late (33), late (44) --
all satisfy 0 < swap_idx < n_half-2=48, so none touch the very first or
very last sorted position (avoiding the max-check/min-check contamination
already ruled out for swap_idx=25 in condition 3's own audit; re-audited
per-position below, not assumed to transfer).

If every non-lstm-seed2 cell shows a large effect at EVERY position, that
supports genuine (comprehensive) sort-order verification. If the effect is
concentrated at one position (e.g. only the original 76/77 or only middle
positions), that indicates position-specific sensitivity, not comprehensive
verification -- a materially different, weaker claim.

Saves analysis_outputs/final_results/phase3_bucketsort_condition3_position_sweep.json.

PYTHONPATH=src:analysis python analysis/phase3_bucketsort_condition3_position_sweep.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_bucketsort_counterfactuals import target_computation_pairs, audit_target_computation_pairs

RESULTS = Path("analysis_outputs/final_results")
TASK = "bucket-sort"
CELLS = [
    ("rnn", 4, "primary"), ("rnn", 10, "secondary_comparable"),
    ("lstm", 2, "primary"), ("lstm", 3, "secondary_near_chance_partial"),
    ("transformer", 7, "primary"), ("transformer", 3, "secondary_comparable"),
    ("mamba", 7, "primary"), ("mamba", 3, "secondary_comparable"),
]
N_HALF = 50
N_PAIRS = 25
DEGENERATE_GAP_THRESHOLD = 0.05
SWAP_POSITIONS = {"early": 5, "early_mid": 17, "mid_late": 33, "late": 44}


def run_rnn_lstm(arch, seed, pairs, tok2idx):
    from rnn_patching import RNNPatchingHarness, RecurrentScan, SITE_RNN, SITE_LSTM_FULL
    h = RNNPatchingHarness(arch, f"rec+ns/validation-short/{seed}", task=TASK)
    sc = RecurrentScan(h)
    site = SITE_RNN if arch == "rnn" else SITE_LSTM_FULL

    wiring, behavioral = [], []
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
    return wiring, behavioral, f"data/models/{TASK}/{arch}/rec+ns/validation-short/{seed}"


def run_mamba(seed, pairs, tok2idx):
    from patching_harness import PatchingHarness
    h = PatchingHarness(task=TASK, model_subdir=f"mamba/rec+ns/validation-short/{seed}")
    top_layer = h.num_layers - 1

    wiring, behavioral = [], []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = h.readout_position(p["clean"])
        assert t_ro == h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))

        _, clean_cache = h.run_with_cache(clean_idx, layers=[top_layer])
        patched_full = h.patch_run(clean_idx, corrupt_idx, layer=top_layer, position=t_ro, dims=None,
                                   clean_cache=clean_cache)
        wiring.append({"restored_fraction": h.restored_fraction(lc, lo, patched_full)})
    return wiring, behavioral, f"models/{TASK}/mamba/rec+ns/validation-short/{seed}"


def run_transformer(seed, pairs, tok2idx):
    from modk_transformer_harness import TransformerPatchingHarness
    h = TransformerPatchingHarness(TASK, f"rec+ns/validation-short/{seed}")
    layer = h.num_layers - 1

    wiring, behavioral = [], []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = len(p["clean"]) - 1
        assert t_ro == len(p["corrupt"]) - 1
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        behavioral.append(abs(lc - lo))

        _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
        clean_row = clean_cache[layer][t_ro + h.bos_off]
        patched_full = h.patch_logit(corrupt_idx, layer, t_ro, clean_row, dims=None)
        rf = (patched_full - lo) / (lc - lo) if abs(lc - lo) > 1e-9 else float("nan")
        wiring.append({"restored_fraction": rf})
    return wiring, behavioral, f"data/models/{TASK}/transformer/rec+ns/validation-short/{seed}"


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

    position_pairs = {}
    position_audits = {}
    for label, swap_idx in SWAP_POSITIONS.items():
        rng = np.random.default_rng(hash(label) % (2**31))
        pairs = target_computation_pairs(N_HALF, N_PAIRS, rng, swap_idx=swap_idx)
        audit = audit_target_computation_pairs(pairs)
        position_pairs[label] = pairs
        position_audits[label] = audit
        print(f"=== swap_idx={swap_idx} ({label}) audit ===")
        print(json.dumps(audit, indent=2))
        assert audit["target_property_isolated"], f"{label} (swap_idx={swap_idx}) failed to isolate the target property"

    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    results_by_position = {}
    for label, swap_idx in SWAP_POSITIONS.items():
        pairs = position_pairs[label]
        cells = []
        for arch, seed, role in CELLS:
            print(f"\n--- [{label}, swap_idx={swap_idx}] {arch} seed{seed} ({role}) ---", flush=True)
            if arch in ("rnn", "lstm"):
                wiring, behavioral, ckpt = run_rnn_lstm(arch, seed, pairs, tok2idx)
            elif arch == "mamba":
                wiring, behavioral, ckpt = run_mamba(seed, pairs, tok2idx)
            else:
                wiring, behavioral, ckpt = run_transformer(seed, pairs, tok2idx)

            beh = summarize_behavioral(behavioral)
            wiring_s = summarize_rf(wiring, "wiring")
            gap = beh["mean_abs_clean_corrupt_logit_gap"]
            causal = gap >= DEGENERATE_GAP_THRESHOLD
            cell = {
                "arch": arch, "seed": seed, "role": role, "checkpoint": ckpt,
                "swap_idx": swap_idx, "swap_position_label": label,
                "PRIMARY_behavioral_logit_gap": beh,
                "wiring_check_full_state_patch_SECONDARY": wiring_s,
                "causally_effective": causal,
            }
            cells.append(cell)
            print(f"  PRIMARY behavioral gap: mean={gap:.4f} (ci95={beh['ci95_abs_clean_corrupt_logit_gap']:.4f}, n={beh['n']}) causal={causal}", flush=True)
            print(f"  full-state wiring check (secondary): rf_mean={wiring_s['wiring_restored_fraction_mean']:.4f}", flush=True)
        results_by_position[label] = cells

    # ------------------------------------------------------------------
    # cross-position summary per cell: comprehensive verification (large
    # effect at EVERY position) vs. position-specific sensitivity
    # ------------------------------------------------------------------
    per_cell_summary = {}
    for arch, seed, role in CELLS:
        key = f"{arch}_seed{seed}"
        gaps_by_position = {}
        causal_by_position = {}
        for label in SWAP_POSITIONS:
            c = next(x for x in results_by_position[label] if x["arch"] == arch and x["seed"] == seed)
            gaps_by_position[label] = c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            causal_by_position[label] = c["causally_effective"]
        all_causal = all(causal_by_position.values())
        none_causal = not any(causal_by_position.values())
        per_cell_summary[key] = {
            "arch": arch, "seed": seed, "role": role,
            "gap_by_position": gaps_by_position, "causal_by_position": causal_by_position,
            "verdict": (
                "NULL AT EVERY POSITION -- consistent with the marker-only characterization "
                "(lstm seed2) or a genuine absence of order-sensitivity."
                if none_causal else
                "COMPREHENSIVE SORT-ORDER VERIFICATION -- large causal effect at EVERY tested "
                "position (early, early-mid, mid-late, late), not concentrated at the single "
                "position condition 3 originally tested. This is genuine, position-general "
                "sensitivity to sort-order correctness, not an artifact of one specific location."
                if all_causal else
                "POSITION-SPECIFIC SENSITIVITY, NOT COMPREHENSIVE VERIFICATION -- causal effect "
                "present at some positions but not others. This cell's sort-order sensitivity is "
                "NOT uniform across the sequence -- report which positions specifically, do not "
                "generalize to 'verifies sort order' without qualification."
            ),
        }
        print(f"{key}: " + " ".join(f"{l}={gaps_by_position[l]:.3f}" for l in SWAP_POSITIONS))

    n_comprehensive = sum(1 for v in per_cell_summary.values() if "COMPREHENSIVE" in v["verdict"])
    n_null = sum(1 for v in per_cell_summary.values() if "NULL AT EVERY" in v["verdict"])
    n_position_specific = len(CELLS) - n_comprehensive - n_null

    overall_conclusion = {
        "n_cells_comprehensive_verification": n_comprehensive,
        "n_cells_null_everywhere": n_null,
        "n_cells_position_specific_only": n_position_specific,
        "interpretation": (
            f"{n_comprehensive}/{len(CELLS)} cells show large causal effects at ALL 4 swept "
            f"positions -- genuine, position-general sort-order verification, not an artifact of "
            f"condition 3's single tested position. {n_null}/{len(CELLS)} show null effects "
            f"everywhere (expected: lstm seed2). "
            + (f"{n_position_specific}/{len(CELLS)} show effects at SOME but not all positions -- "
               f"position-specific sensitivity, a weaker and more qualified claim than comprehensive "
               f"verification, reported per-cell above rather than smoothed into the aggregate story."
               if n_position_specific else
               "No cell shows a mixed (position-specific-only) pattern -- the comprehensive-vs-null "
               "split is completely clean.")
        ),
    }
    print("\n=== overall conclusion ===")
    print(json.dumps(overall_conclusion, indent=2, default=str))

    out = {
        "condition": "condition_3_position_sweep_followup",
        "description": "Follow-up to condition 3: sweeps the target_computation_multiset_matched_"
                       "case swap position across early/early-mid/mid-late/late locations within "
                       "the sorted half, all 8 cells, to test whether condition 3's large causal "
                       "effects (found at ONE fixed position, swap_idx=25) reflect comprehensive "
                       "sort-order verification or position-specific sensitivity.",
        "swap_positions_tested": SWAP_POSITIONS,
        "n_half": N_HALF, "n_pairs_per_position": N_PAIRS,
        "position_construction_audits": position_audits,
        "results_by_position": results_by_position,
        "per_cell_summary": per_cell_summary,
        "overall_conclusion": overall_conclusion,
    }
    out_path = RESULTS / "phase3_bucketsort_condition3_position_sweep.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
