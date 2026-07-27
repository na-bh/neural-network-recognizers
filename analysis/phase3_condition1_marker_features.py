"""Phase 3, patch condition 1 of 3: marker features (marker_count,
marker_position). Tests whether the directly-observable marker shortcut
features -- represented at ~0.24-0.32 selectivity by ALL FOUR architectures
in Phase 2 (phase2_p3_marked_reversal.json) -- causally drive the accept/
reject decision. This is the "obvious, large-effect" test, analogous to
cycle-navigation's grammar_pairs: a FULL-STATE patch (all layers/channels,
readout position, established canonical site per architecture) is used as
the primary wiring-style check, since marker features are not a subtle
signal requiring a layer sweep to locate (unlike length_parity/is_balanced).

Two pair types:
  - marker_count_pairs: clean = genuine positive (1 marker). corrupt = SAME
    LENGTH, one content character (just before the marker) replaced by a
    second '#' -- marker_count_class flips 1 -> 2. Differs at exactly 1
    token position.
  - marker_position_pairs: clean = genuine positive (marker centered).
    corrupt = SAME LENGTH, SAME marker count (1), independently-generated
    content with the marker placed FAR off-center (relative deviation 0.38,
    well past the 0.05 near-center threshold) -- marker_position_bin flips
    1 (exact-center) -> 3 (far).

Saves analysis_outputs/final_results/phase3_causal_patching.json with key
"condition_1_marker_features". Per the user's instruction, this is the FIRST
of three patch conditions -- STOP after this one for review before running
condition 2 (length parity).

PYTHONPATH=src:analysis python analysis/phase3_condition1_marker_features.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
import phase2_targets as T

RESULTS = Path("analysis_outputs/final_results")
TASK = "marked-reversal"
SEED_PAIRS = {"rnn": [3, 7], "lstm": [10, 1], "transformer": [9, 1], "mamba": [4, 6]}
N_HALF = 50
N_PAIRS = 25


# ---------------------------------------------------------------------------
# pair construction
# ---------------------------------------------------------------------------

def marker_count_pairs(n_half, n_pairs, rng):
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + list(reversed(w))
        corrupt_tokens = clean_tokens[:]
        corrupt_tokens[n_half - 1] = "#"  # extra marker just before the real one
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "clean_marker_count": T.markedrev_marker_count_class(clean_tokens),
            "corrupt_marker_count": T.markedrev_marker_count_class(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} marker_count pairs")
    return pairs


def marker_position_pairs(n_half, n_pairs, rng):
    L = 2 * n_half + 1
    far_marker_idx = n_half // 4
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + list(reversed(w))
        corrupt_content = [str(int(rng.integers(0, 2))) for _ in range(L - 1)]
        corrupt_tokens = corrupt_content[:far_marker_idx] + ["#"] + corrupt_content[far_marker_idx:]
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "clean_marker_position_bin": T.markedrev_marker_position_bin(clean_tokens),
            "corrupt_marker_position_bin": T.markedrev_marker_position_bin(corrupt_tokens),
            "clean_marker_count": T.markedrev_marker_count_class(clean_tokens),
            "corrupt_marker_count": T.markedrev_marker_count_class(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} marker_position pairs")
    return pairs


def audit_marker_count_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for a, b in zip(p["clean"], p["corrupt"]) if a != b) for p in pairs]
    audit["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    audit["clean_all_count_1"] = all(p["clean_marker_count"] == 1 for p in pairs)
    audit["corrupt_all_count_2"] = all(p["corrupt_marker_count"] == 2 for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["differs_at_exactly_one_position"] and
        audit["clean_all_count_1"] and audit["corrupt_all_count_2"]
    )
    return audit


def audit_marker_position_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    audit["same_marker_count"] = all(
        p["clean_marker_count"] == 1 and p["corrupt_marker_count"] == 1 for p in pairs)
    audit["clean_all_centered"] = all(p["clean_marker_position_bin"] == 1 for p in pairs)
    audit["corrupt_all_far"] = all(p["corrupt_marker_position_bin"] == 3 for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["same_marker_count"] and
        audit["clean_all_centered"] and audit["corrupt_all_far"]
    )
    return audit


# ---------------------------------------------------------------------------
# per-architecture patch runners (full-state / canonical-last-layer patch)
# ---------------------------------------------------------------------------

def run_rnn_lstm(arch, seed, pairs, tok2idx):
    from rnn_patching import RNNPatchingHarness, RecurrentScan, SITE_RNN, SITE_LSTM_FULL
    h = RNNPatchingHarness(arch, f"rec+ns/validation-short/{seed}", task=TASK)
    sc = RecurrentScan(h)
    site = SITE_RNN if arch == "rnn" else SITE_LSTM_FULL

    results = []
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = h.readout_position(p["clean"])  # same length -> same readout position
        assert t_ro == h.readout_position(p["corrupt"])
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, states = sc.record(clean_idx, [t_ro])
        patched = sc.patch_logit(corrupt_idx, t_ro, states[t_ro], site=site, dims=None)
        rf = h.restored_fraction(lc, lo, patched)
        results.append({"clean_logit": lc, "corrupt_logit": lo, "patched_logit": patched,
                         "restored_fraction": rf})
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
                         "restored_fraction": rf})
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
                         "restored_fraction": rf})
    return results, f"data/models/{TASK}/transformer/rec+ns/validation-short/{seed}"


def summarize(results, pair_type, arch, seed, checkpoint):
    rfs = np.array([r["restored_fraction"] for r in results if not np.isnan(r["restored_fraction"])])
    n = len(rfs)
    mean = float(np.mean(rfs)) if n else float("nan")
    ci95 = float(1.96 * np.std(rfs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    logit_gaps = [abs(r["clean_logit"] - r["corrupt_logit"]) for r in results]
    return {
        "arch": arch, "seed": seed, "pair_type": pair_type, "checkpoint": checkpoint,
        "n_pairs_total": len(results), "n_pairs_nondegenerate_denom": n,
        "restored_fraction_mean": mean, "restored_fraction_ci95": ci95,
        "mean_abs_clean_corrupt_logit_gap": float(np.mean(logit_gaps)),
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
    for arch, seeds in SEED_PAIRS.items():
        for seed in seeds:
            print(f"\n--- {arch} seed{seed} ---", flush=True)
            for pair_type, pairs in [("marker_count", mc_pairs), ("marker_position", mp_pairs)]:
                if arch in ("rnn", "lstm"):
                    results, ckpt = run_rnn_lstm(arch, seed, pairs, tok2idx)
                elif arch == "mamba":
                    results, ckpt = run_mamba(seed, pairs, tok2idx)
                else:
                    results, ckpt = run_transformer(seed, pairs, tok2idx)
                cell = summarize(results, pair_type, arch, seed, ckpt)
                cells.append(cell)
                print(f"  [{pair_type}] rf_mean={cell['restored_fraction_mean']:.4f} "
                      f"(ci95={cell['restored_fraction_ci95']:.4f}) "
                      f"n={cell['n_pairs_nondegenerate_denom']} "
                      f"gap={cell['mean_abs_clean_corrupt_logit_gap']:.4f}", flush=True)

    out = {
        "condition": "condition_1_marker_features",
        "status": "PRIMARY wiring-style test: full-state patch (all layers/channels) at the "
                 "readout position, canonical established site per architecture (matches how "
                 "cycle-navigation's grammar_pairs were tested). No layer sweep needed since "
                 "marker features are represented broadly (~0.24-0.32 selectivity) across all "
                 "layers/architectures per Phase 2, not a subtle late-layer signal.",
        "description": "Tests whether marker_count and marker_position -- the directly-observable "
                       "structural shortcut every architecture represents -- causally drive the "
                       "accept/reject decision. Predicted: high restored fraction (~1.0) for all "
                       "4 architectures, both pair types, both seeds.",
        "n_half": N_HALF, "n_pairs_generated": N_PAIRS,
        "marker_count_pairs_audit": mc_audit,
        "marker_position_pairs_audit": mp_audit,
        "sample_pairs": {"marker_count": mc_pairs[:3], "marker_position": mp_pairs[:3]},
        "cells": cells,
    }
    out_path = RESULTS / "phase3_causal_patching.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
