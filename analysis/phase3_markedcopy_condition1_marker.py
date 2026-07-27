"""Marked-copy Phase 3, condition 1 of 3: marker features (marker_count,
marker_position). Tests whether the marker shortcut features -- represented
at 0.22-0.38 selectivity by ALL 8 cells in Stage 2 (phase2_markedcopy_
marker_probes.json) -- causally drive the accept/reject decision.

Per the approved protocol: PRIMARY evidence is the behavioral logit gap
(|clean_logit - corrupt_logit|, unpatched); the full-state patch (all
layers/channels, canonical readout site per architecture: RNN/LSTM full
recurrent state via SITE_RNN/SITE_LSTM_FULL, Mamba/Transformer last layer)
is reported as a SECONDARY wiring check only. Marker features are broadly
represented (not a subtle single-layer signal per Stage 2), so no layer
sweep is needed for condition 1 specifically -- flagged explicitly for LSTM
seed1, whose Stage 2 probe selectivity (0.34/0.30) exceeded even its own
primary seed9 (0.28/0.26) despite chance behavioral accuracy: if patching
still produces a large behavioral gap / high rf here, the representation is
causally reachable but seed1's forward pass isn't routing to it in its
normal (unpatched) processing of its OWN inputs; if patching produces near-
zero effect, the representation is decoupled from the readout entirely.

Saves analysis_outputs/final_results/phase3_markedcopy_condition1_marker.json.
STOP after this condition for review before condition 2 (structural features).

PYTHONPATH=src:analysis python analysis/phase3_markedcopy_condition1_marker.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
from phase3_markedcopy_counterfactuals import (
    marker_count_pairs, marker_position_pairs,
    audit_marker_count_pairs, audit_marker_position_pairs,
)

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


def run_rnn_lstm(arch, seed, pairs, tok2idx):
    from rnn_patching import RNNPatchingHarness, RecurrentScan, SITE_RNN, SITE_LSTM_FULL
    h = RNNPatchingHarness(arch, f"rec+ns/validation-short/{seed}", task=TASK)
    sc = RecurrentScan(h)
    site = SITE_RNN if arch == "rnn" else SITE_LSTM_FULL

    results = []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = h.readout_position(p["clean"])
        assert t_ro == h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, states = sc.record(clean_idx, [t_ro])
        patched = sc.patch_logit(corrupt_idx, t_ro, states[t_ro], site=site, dims=None)
        rf = h.restored_fraction(lc, lo, patched)
        results.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched,
                         "restored_fraction": rf, "behavioral_gap": abs(lc - lo)})
    return results, f"data/models/{TASK}/{arch}/rec+ns/validation-short/{seed}"


def run_mamba(seed, pairs, tok2idx):
    from patching_harness import PatchingHarness
    h = PatchingHarness(task=TASK, model_subdir=f"mamba/rec+ns/validation-short/{seed}")
    layer = h.num_layers - 1

    results = []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = h.readout_position(p["clean"])
        assert t_ro == h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
        patched = h.patch_run(clean_idx, corrupt_idx, layer=layer, position=t_ro, dims=None,
                              clean_cache=clean_cache)
        rf = h.restored_fraction(lc, lo, patched)
        results.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched,
                         "restored_fraction": rf, "behavioral_gap": abs(lc - lo)})
    return results, f"models/{TASK}/mamba/rec+ns/validation-short/{seed}"


def run_transformer(seed, pairs, tok2idx):
    from modk_transformer_harness import TransformerPatchingHarness
    h = TransformerPatchingHarness(TASK, f"rec+ns/validation-short/{seed}")
    layer = h.num_layers - 1

    results = []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = len(p["clean"]) - 1
        assert t_ro == len(p["corrupt"]) - 1
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
        clean_row = clean_cache[layer][t_ro + h.bos_off]
        patched = h.patch_logit(corrupt_idx, layer, t_ro, clean_row, dims=None)
        rf = (patched - lo) / (lc - lo) if abs(lc - lo) > 1e-9 else float("nan")
        results.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched,
                         "restored_fraction": rf, "behavioral_gap": abs(lc - lo)})
    return results, f"data/models/{TASK}/transformer/rec+ns/validation-short/{seed}"


def summarize(results, pair_type, arch, seed, role, checkpoint):
    rfs = np.array([r["restored_fraction"] for r in results if not np.isnan(r["restored_fraction"])])
    n = len(rfs)
    mean = float(np.mean(rfs)) if n else float("nan")
    ci95 = float(1.96 * np.std(rfs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    gaps = np.array([r["behavioral_gap"] for r in results])
    gap_mean = float(np.mean(gaps))
    gap_ci95 = float(1.96 * np.std(gaps, ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else float("nan")
    return {
        "arch": arch, "seed": seed, "role": role, "pair_type": pair_type, "checkpoint": checkpoint,
        "PRIMARY_behavioral_logit_gap": {
            "mean_abs_clean_corrupt_logit_gap": gap_mean, "ci95": gap_ci95, "n": len(gaps),
        },
        "wiring_check_full_state_patch_SECONDARY": {
            "restored_fraction_mean": mean, "restored_fraction_ci95": ci95,
            "n_nondegenerate_denom": n,
        },
        "raw_restored_fractions": [float(x) for x in rfs],
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    mc_pairs = marker_count_pairs(N_HALF, N_PAIRS, rng)
    mp_pairs = marker_position_pairs(N_HALF, N_PAIRS, rng)
    mc_audit = audit_marker_count_pairs(mc_pairs)
    mp_audit = audit_marker_position_pairs(mp_pairs)
    print("=== marker_count_pairs audit ===")
    print(json.dumps(mc_audit, indent=2))
    print("=== marker_position_pairs audit ===")
    print(json.dumps(mp_audit, indent=2))
    assert mc_audit["target_property_isolated"] and mp_audit["target_property_isolated"]

    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    cells = []
    for arch, seed, role in CELLS:
        print(f"\n--- {arch} seed{seed} ({role}) ---", flush=True)
        for pair_type, pairs in [("marker_count", mc_pairs), ("marker_position", mp_pairs)]:
            if arch in ("rnn", "lstm"):
                results, ckpt = run_rnn_lstm(arch, seed, pairs, tok2idx)
            elif arch == "mamba":
                results, ckpt = run_mamba(seed, pairs, tok2idx)
            else:
                results, ckpt = run_transformer(seed, pairs, tok2idx)
            cell = summarize(results, pair_type, arch, seed, role, ckpt)
            cells.append(cell)
            gap = cell["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            rf = cell["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
            print(f"  [{pair_type}] PRIMARY behavioral gap={gap:.4f}  |  "
                  f"wiring rf_mean={rf:.4f} n={cell['wiring_check_full_state_patch_SECONDARY']['n_nondegenerate_denom']}",
                  flush=True)

    # ------------------------------------------------------------------
    # representation-not-causal-use check, LSTM seed1 specifically flagged
    # ------------------------------------------------------------------
    DEGENERATE_GAP_THRESHOLD = 0.05
    findings = []
    for arch, seed, role in CELLS:
        cs = [c for c in cells if c["arch"] == arch and c["seed"] == seed]
        for c in cs:
            gap = c["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
            rf = c["wiring_check_full_state_patch_SECONDARY"]["restored_fraction_mean"]
            causally_effective = gap >= DEGENERATE_GAP_THRESHOLD and not np.isnan(rf) and rf >= 0.5
            findings.append({
                "arch": arch, "seed": seed, "role": role, "pair_type": c["pair_type"],
                "behavioral_gap": gap, "wiring_rf": rf, "causally_effective": causally_effective,
            })

    lstm1_marker_count = next(f for f in findings if f["arch"] == "lstm" and f["seed"] == 1 and f["pair_type"] == "marker_count")
    lstm1_marker_position = next(f for f in findings if f["arch"] == "lstm" and f["seed"] == 1 and f["pair_type"] == "marker_position")
    lstm1_reachable = lstm1_marker_count["causally_effective"] or lstm1_marker_position["causally_effective"]
    lstm1_mechanism_finding = {
        "cell": "lstm_seed1 (chance-performing collapser, 0.4933 behavioral accuracy)",
        "marker_count": {"gap": lstm1_marker_count["behavioral_gap"], "wiring_rf": lstm1_marker_count["wiring_rf"]},
        "marker_position": {"gap": lstm1_marker_position["behavioral_gap"], "wiring_rf": lstm1_marker_position["wiring_rf"]},
        "interpretation": (
            "REPRESENTATION IS CAUSALLY REACHABLE: patching produces a large behavioral gap / high "
            "restored fraction despite seed1's own forward pass performing at chance on its OWN "
            "inputs -- the marker-feature information Stage 2 decoded IS wired to the readout; "
            "seed1's failure must be in how it computes/routes to that representation on real "
            "inputs, not in whether the readout can use it once present."
            if lstm1_reachable else
            "REPRESENTATION IS DECOUPLED FROM READOUT: despite Stage 2's probe showing strong "
            "marker-feature selectivity (0.34 marker_count, 0.30 marker_position, both HIGHER than "
            "seed9's), patching produces no meaningful behavioral effect -- the information is "
            "linearly decodable from the activation but is not causally connected to the accept/"
            "reject decision. This is the sharper of the two possible 'representation without "
            "causal use' findings: not a routing failure on real inputs, but a genuine "
            "representation/readout disconnection."
        ),
    }
    print("\n=== LSTM seed1 mechanism finding ===")
    print(json.dumps(lstm1_mechanism_finding, indent=2, default=str))

    disconnected_despite_signal = [
        f for f in findings if not f["causally_effective"]
    ]
    print("\n=== cells/pair-types with near-zero causal effect (any) ===")
    print(json.dumps(disconnected_despite_signal, indent=2, default=str))

    out = {
        "condition": "condition_1_marker_features",
        "status": "PRIMARY evidence is the behavioral logit gap (clean vs corrupt, unpatched); "
                 "full-state patch (all layers/channels, canonical readout site per architecture) "
                 "is SECONDARY, a wiring check only. Predicted: large behavioral gap and high "
                 "restored fraction across all 8 cells, since marker features are represented "
                 "broadly (0.22-0.38 selectivity, Stage 2) by every cell.",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "marker_count_pairs_audit": mc_audit,
        "marker_position_pairs_audit": mp_audit,
        "sample_pairs": {"marker_count": mc_pairs[:3], "marker_position": mp_pairs[:3]},
        "cells": cells,
        "representation_vs_causal_use_findings": findings,
        "lstm_seed1_mechanism_finding": lstm1_mechanism_finding,
    }
    out_path = RESULTS / "phase3_markedcopy_condition1_marker.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
