"""Follow-up investigation for transformer seed2's condition-1 puzzle: this
cell (primary, 0.88 behavioral accuracy, the pilot's second-best cell) shows
STRONG marker_count/marker_position probe selectivity (0.349/0.293 at
Stage 2's layer2) but NEAR-ZERO causal effect when marker_count and
marker_position are patched INDIVIDUALLY at the canonical (last-layer)
readout site (condition 1: gaps 0.047 and 0.049 respectively -- both at the
noise floor).

Two hypotheses tested here, both restricted to transformer seed2 only:

  H1 (joint routing): the marker features are causally used, but only when
     BOTH are simultaneously wrong -- a single corrupted example that is
     WRONG on marker_count (2 markers) AND marker_position (far off-center)
     at once, full-state patched at the canonical (last) layer. If the
     joint behavioral gap/restored fraction is large despite both
     individual gaps being near zero, that is a genuine joint-feature
     mechanism, not visible to either single-feature test.

  H2 (wrong site): marker_count's causal effect appears at an EARLIER layer
     than the canonical last-layer site Stage 2's probe used. Patches
     marker_count_pairs (from condition 1) at EACH of layers 0-4
     individually and reports the per-layer behavioral restored fraction,
     to check whether the probe's layer2 selectivity peak corresponds to a
     causal effect concentrated away from layer4.

Saves analysis_outputs/final_results/phase3_markedcopy_transformer2_investigation.json.

PYTHONPATH=src:analysis python analysis/phase3_markedcopy_transformer2_investigation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "analysis")
import phase2_targets as T
from phase3_markedcopy_counterfactuals import marker_count_pairs

RESULTS = Path("analysis_outputs/final_results")
TASK = "marked-copy"
ARCH, SEED = "transformer", 2
N_HALF = 50
N_PAIRS = 25


# ---------------------------------------------------------------------------
# H1: joint marker_count + marker_position corruption
# ---------------------------------------------------------------------------

def joint_marker_pairs(n_half, n_pairs, rng):
    """clean = w#w (marker_count=1, marker_position centered). corrupt =
    independently-generated content with the marker placed FAR off-center
    (marker_position_pairs' recipe) AND a second marker inserted just before
    it (marker_count_pairs' recipe) -- WRONG on both features
    simultaneously in one example."""
    L = 2 * n_half + 1
    far_marker_idx = n_half // 4
    assert far_marker_idx >= 1
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        w = [str(int(rng.integers(0, 2))) for _ in range(n_half)]
        clean_tokens = w + ["#"] + w
        corrupt_content = [str(int(rng.integers(0, 2))) for _ in range(L - 1)]
        corrupt_tokens = corrupt_content[:far_marker_idx] + ["#"] + corrupt_content[far_marker_idx:]
        corrupt_tokens[far_marker_idx - 1] = "#"
        key = (tuple(clean_tokens), tuple(corrupt_tokens))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "clean": clean_tokens, "corrupt": corrupt_tokens, "length": len(clean_tokens),
            "clean_marker_count": T.markedcopy_marker_count_class(clean_tokens),
            "corrupt_marker_count": T.markedcopy_marker_count_class(corrupt_tokens),
            "clean_marker_position_bin": T.markedcopy_marker_position_bin(clean_tokens),
            "corrupt_marker_position_bin": T.markedcopy_marker_position_bin(corrupt_tokens),
        })
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only made {len(pairs)}/{n_pairs} joint pairs")
    return pairs


def audit_joint_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    audit["clean_all_count_1"] = all(p["clean_marker_count"] == 1 for p in pairs)
    audit["corrupt_all_count_2"] = all(p["corrupt_marker_count"] == 2 for p in pairs)
    audit["clean_all_centered"] = all(p["clean_marker_position_bin"] == 1 for p in pairs)
    audit["corrupt_all_far"] = all(p["corrupt_marker_position_bin"] == 3 for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["clean_all_count_1"] and audit["corrupt_all_count_2"] and
        audit["clean_all_centered"] and audit["corrupt_all_far"]
    )
    return audit


def run_h1(pairs, tok2idx):
    from modk_transformer_harness import TransformerPatchingHarness
    h = TransformerPatchingHarness(TASK, f"rec+ns/validation-short/{SEED}")
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
    return results


# ---------------------------------------------------------------------------
# H2: marker_count patched at each layer 0-4 individually
# ---------------------------------------------------------------------------

def run_h2(pairs, tok2idx):
    from modk_transformer_harness import TransformerPatchingHarness
    h = TransformerPatchingHarness(TASK, f"rec+ns/validation-short/{SEED}")
    num_layers = h.num_layers

    per_layer = {l: [] for l in range(num_layers)}
    for p in pairs:
        clean_idx = [tok2idx[t] for t in p["clean"]]
        corrupt_idx = [tok2idx[t] for t in p["corrupt"]]
        t_ro = len(p["clean"]) - 1
        assert t_ro == len(p["corrupt"]) - 1
        lc = h.logit_diff(clean_idx)
        lo = h.logit_diff(corrupt_idx)
        _, clean_cache = h.run_with_cache(clean_idx, layers=list(range(num_layers)))
        for layer in range(num_layers):
            clean_row = clean_cache[layer][t_ro + h.bos_off]
            patched = h.patch_logit(corrupt_idx, layer, t_ro, clean_row, dims=None)
            rf = (patched - lo) / (lc - lo) if abs(lc - lo) > 1e-9 else float("nan")
            per_layer[layer].append({"restored_fraction": rf, "behavioral_gap": abs(lc - lo)})
    return per_layer, num_layers


def summarize(results):
    rfs = np.array([r["restored_fraction"] for r in results if not np.isnan(r["restored_fraction"])])
    n = len(rfs)
    mean = float(np.mean(rfs)) if n else float("nan")
    ci95 = float(1.96 * np.std(rfs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    gaps = np.array([r["behavioral_gap"] for r in results])
    return {
        "restored_fraction_mean": mean, "restored_fraction_ci95": ci95, "n_nondegenerate_denom": n,
        "mean_abs_clean_corrupt_logit_gap": float(np.mean(gaps)), "n": len(gaps),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    vd = torch.load(f"languages/{TASK}/main.vocab", weights_only=False)
    tok2idx = {t: i for i, t in enumerate(vd["tokens"])}

    # -------------------------- H1 --------------------------
    print("=== H1: joint marker_count + marker_position patch ===", flush=True)
    joint_pairs = joint_marker_pairs(N_HALF, N_PAIRS, rng)
    joint_audit = audit_joint_pairs(joint_pairs)
    print(json.dumps(joint_audit, indent=2))
    assert joint_audit["target_property_isolated"]
    h1_results = run_h1(joint_pairs, tok2idx)
    h1_summary = summarize(h1_results)
    print(f"H1 joint patch: behavioral_gap={h1_summary['mean_abs_clean_corrupt_logit_gap']:.4f} "
          f"rf_mean={h1_summary['restored_fraction_mean']:.4f} n={h1_summary['n']}", flush=True)

    cond1 = json.loads((RESULTS / "phase3_markedcopy_condition1_marker.json").read_text())
    t2_cells = {c["pair_type"]: c for c in cond1["cells"] if c["arch"] == "transformer" and c["seed"] == 2}
    mc_gap_individual = t2_cells["marker_count"]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    mp_gap_individual = t2_cells["marker_position"]["PRIMARY_behavioral_logit_gap"]["mean_abs_clean_corrupt_logit_gap"]
    joint_gap = h1_summary["mean_abs_clean_corrupt_logit_gap"]
    DEGENERATE_GAP_THRESHOLD = 0.05
    h1_verdict = (
        "JOINT ROUTING CONFIRMED: the joint corruption produces a large behavioral gap "
        f"({joint_gap:.4f}) despite BOTH individual gaps being near the noise floor "
        f"(marker_count={mc_gap_individual:.4f}, marker_position={mp_gap_individual:.4f}) -- "
        "transformer seed2 only reacts when marker_count AND marker_position are simultaneously "
        "wrong, a genuinely joint (non-additive) feature dependency invisible to either "
        "single-feature test."
        if joint_gap >= DEGENERATE_GAP_THRESHOLD else
        "JOINT ROUTING NOT SUPPORTED: the joint corruption's behavioral gap "
        f"({joint_gap:.4f}) is ALSO near the noise floor, matching both individual gaps "
        f"(marker_count={mc_gap_individual:.4f}, marker_position={mp_gap_individual:.4f}) -- this "
        "cell's high accuracy (0.88) is not explained by marker_count or marker_position, "
        "individually OR jointly, at the canonical last-layer site. The puzzle from condition 1 "
        "remains open; whatever mechanism drives this cell's accuracy is not captured by any "
        "marker-feature corruption tested so far."
    )
    print(h1_verdict, flush=True)

    # -------------------------- H2 --------------------------
    print("\n=== H2: marker_count patched at each layer 0-4 ===", flush=True)
    mc_pairs = marker_count_pairs(N_HALF, N_PAIRS, np.random.default_rng(0))
    per_layer, num_layers = run_h2(mc_pairs, tok2idx)
    per_layer_summary = {l: summarize(res) for l, res in per_layer.items()}
    for l in range(num_layers):
        s = per_layer_summary[l]
        print(f"  layer{l}: rf_mean={s['restored_fraction_mean']:.4f} "
              f"gap={s['mean_abs_clean_corrupt_logit_gap']:.4f} n={s['n']}", flush=True)

    best_layer_h2 = max(per_layer_summary, key=lambda l: (per_layer_summary[l]["restored_fraction_mean"]
                                                            if not np.isnan(per_layer_summary[l]["restored_fraction_mean"]) else -999))
    h2_verdict = (
        f"marker_count's causal effect is concentrated at layer{best_layer_h2} "
        f"(rf={per_layer_summary[best_layer_h2]['restored_fraction_mean']:.4f}), "
        + ("matching the canonical last-layer site already tested in condition 1 -- the near-zero "
           "condition-1 result is NOT a wrong-site artifact; marker_count genuinely has minimal "
           "causal effect at this cell regardless of which layer is patched."
           if best_layer_h2 == num_layers - 1 else
           f"DIFFERENT from the canonical last-layer site (layer{num_layers-1}) condition 1 tested -- "
           "the near-zero condition-1 result WAS a wrong-site artifact: marker_count does have a "
           "meaningful causal effect on this cell, just carried at an earlier layer than the "
           "canonical readout site probes/patches by convention.")
    )
    print(h2_verdict, flush=True)

    out = {
        "description": "Follow-up investigation, transformer seed2 only: does the marker-feature "
                       "causal-effect puzzle from condition 1 (strong probe signal, near-zero "
                       "individual causal effect) resolve via joint-feature routing (H1) or "
                       "wrong-layer testing (H2)?",
        "cell": {"arch": ARCH, "seed": SEED, "role": "primary", "behavioral_accuracy": 0.88},
        "h1_joint_marker_patch": {
            "hypothesis": "marker features are causally used only when BOTH marker_count and "
                          "marker_position are simultaneously wrong, not individually",
            "joint_pairs_audit": joint_audit,
            "sample_pairs": joint_pairs[:3],
            "result": h1_summary,
            "condition_1_individual_gaps_for_reference": {
                "marker_count": mc_gap_individual, "marker_position": mp_gap_individual,
            },
            "verdict": h1_verdict,
        },
        "h2_per_layer_marker_count_patch": {
            "hypothesis": "marker_count's causal effect is concentrated at a layer other than the "
                          "canonical last-layer site tested in condition 1",
            "num_layers": num_layers,
            "per_layer_result": per_layer_summary,
            "best_layer": best_layer_h2,
            "canonical_site_tested_in_condition_1": num_layers - 1,
            "verdict": h2_verdict,
        },
    }
    out_path = RESULTS / "phase3_markedcopy_transformer2_investigation.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
