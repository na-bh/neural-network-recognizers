"""Shared causal-patching runner functions for compute-sqrt's bounded
Phase 3 pilot -- task-parameterized versions of the RNN/LSTM/Mamba/
Transformer runner boilerplate already established and reused across every
prior task in this pilot.

Uses separate clean/corrupt readout positions throughout (Condition 2's
answer_length_insufficient pairs are NOT same-length).

Checkpoints: rec+ns/validation-short/<seed>, matching every other task's
Phase 3 patching in this pilot.

PYTHONPATH=src:analysis python analysis/phase3_computesqrt_patching_common.py
"""

import json
from pathlib import Path

import numpy as np
import torch

TASK = "compute-sqrt"
RESULTS = Path("analysis_outputs/final_results")
DEGENERATE_GAP_THRESHOLD = 0.05
CELLS = [
    ("rnn", 1, "primary"), ("rnn", 8, "secondary"),
    ("lstm", 6, "primary"), ("lstm", 7, "secondary"),
    ("transformer", 7, "primary"), ("transformer", 2, "secondary"),
    ("mamba", 0, "primary"), ("mamba", 5, "secondary"),
]


def get_tok2idx():
    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    return {t: i for i, t in enumerate(vd["tokens"])}


def run_rnn_lstm(arch, seed, pairs, tok2idx):
    from rnn_patching import RNNPatchingHarness, RecurrentScan, SITE_RNN, SITE_LSTM_FULL
    h = RNNPatchingHarness(arch, f"rec+ns/validation-short/{seed}", task=TASK)
    sc = RecurrentScan(h)
    site = SITE_RNN if arch == "rnn" else SITE_LSTM_FULL

    results = []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro_clean = h.readout_position(p["clean"])
        t_ro_corrupt = h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, states = sc.record(clean_idx, [t_ro_clean])
        patched = sc.patch_logit(corrupt_idx, t_ro_corrupt, states[t_ro_clean], site=site, dims=None)
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
        t_ro_clean = h.readout_position(p["clean"])
        t_ro_corrupt = h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
        cv = clean_cache[layer].to(h.device)

        def hook(module, inputs, output, cv=cv, t_ro_corrupt=t_ro_corrupt, t_ro_clean=t_ro_clean):
            out = output.clone()
            out[0, t_ro_corrupt, :] = cv[t_ro_clean, :]
            return out

        patched = h._forward(corrupt_idx, hooks=[(layer, hook)])
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
        t_ro_clean = len(p["clean"]) - 1
        t_ro_corrupt = len(p["corrupt"]) - 1
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
        clean_row = clean_cache[layer][t_ro_clean + h.bos_off]
        patched = h.patch_logit(corrupt_idx, layer, t_ro_corrupt, clean_row, dims=None)
        rf = (patched - lo) / (lc - lo) if abs(lc - lo) > 1e-9 else float("nan")
        results.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched,
                        "restored_fraction": rf, "behavioral_gap": abs(lc - lo)})
    return results, f"data/models/{TASK}/transformer/rec+ns/validation-short/{seed}"


def run_cell(arch, seed, pairs, tok2idx):
    if arch in ("rnn", "lstm"):
        return run_rnn_lstm(arch, seed, pairs, tok2idx)
    elif arch == "mamba":
        return run_mamba(seed, pairs, tok2idx)
    else:
        return run_transformer(seed, pairs, tok2idx)


def summarize(results, pair_type, arch, seed, role, checkpoint):
    gaps = np.array([r["behavioral_gap"] for r in results])
    gap_mean = float(np.mean(gaps))
    gap_ci95 = float(1.96 * np.std(gaps, ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else float("nan")
    degenerate = gap_mean < DEGENERATE_GAP_THRESHOLD
    rfs = np.array([r["restored_fraction"] for r in results if not np.isnan(r["restored_fraction"])])
    n = len(rfs)
    rf_mean = float(np.mean(rfs)) if n else float("nan")
    rf_ci95 = float(1.96 * np.std(rfs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "arch": arch, "seed": seed, "role": role, "pair_type": pair_type, "checkpoint": checkpoint,
        "PRIMARY_behavioral_logit_gap": {
            "mean_abs_clean_corrupt_logit_gap": gap_mean, "ci95": gap_ci95, "n": len(gaps),
        },
        "causally_effective": bool(gap_mean >= DEGENERATE_GAP_THRESHOLD),
        "wiring_check_full_state_patch_SECONDARY": {
            "restored_fraction_mean": rf_mean, "restored_fraction_ci95": rf_ci95,
            "n_nondegenerate_denom": n,
            "UNRELIABLE_near_zero_behavioral_gap": degenerate,
        },
        "raw_restored_fractions": [float(x) for x in rfs],
    }


def load_cell_selection_context():
    d = json.loads((RESULTS / "phase3_computesqrt_cell_selection.json").read_text())
    return d["cell_selection"]


if __name__ == "__main__":
    tok2idx = get_tok2idx()
    print("tok2idx:", tok2idx)
    print("cells:", CELLS)
