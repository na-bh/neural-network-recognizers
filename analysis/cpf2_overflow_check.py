"""CPF2 overflow-rate characterization: quantifies how often the pair-merge
step in ContrastiveRecognitionTrainingLoop pushes a batch over
max_tokens_per_batch, across the ACTUAL [128, 4096] budget range FLaRe's
own random_sample.py samples from (log-uniform), and across multiple
shuffle seeds to characterize variance. Pure batch-construction analysis --
no model, no training, no GPU.

PYTHONPATH=src:analysis python analysis/cpf2_overflow_check.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import scipy.stats

sys.path.insert(0, "src")
from recognizers.neural_networks.data import load_prepared_data_from_directory
from recognizers.neural_networks.batching import group_into_batches
from recognizers.neural_networks.contrastive_batching import load_pair_ids, build_pair_partner_map

RESULTS = Path("analysis_outputs/final_results")
LANG_DIR = Path("languages/marked-reversal-fixed-contrastive")
BUDGETS = [128, 256, 512, 1024, 2048, 4096]
SHUFFLE_SEEDS = [1001, 1002, 1003, 1004, 1005]


class _StubModelInterface:
    use_next_symbols_head = False


def instrumented_merge(examples, max_tokens, pair_partner_map):
    """Standalone re-implementation of the SAME merge logic used in
    contrastive_batching.group_into_batches_pair_aware, but returning full
    measurement detail (fraction requiring merge, fraction of merges
    overflowing, and the exact overflow percentage per overflowed batch)
    instead of just logging summary counts."""
    is_small_enough = lambda b, n: b * n <= max_tokens
    batches = [list(b) for b in group_into_batches(examples, is_small_enough)]
    owner_batch = {}
    id_to_example = {}
    for bi, batch in enumerate(batches):
        for ex in batch:
            owner_batch[id(ex)] = bi
            id_to_example[id(ex)] = ex

    seen_pairs = set()
    n_pairs_present = 0
    n_merged = 0
    overflow_pcts = []  # percent over budget, for every merge that overflowed
    for a_id, b_id in pair_partner_map.items():
        pair_key = frozenset((a_id, b_id))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        if a_id not in owner_batch or b_id not in owner_batch:
            continue
        n_pairs_present += 1
        bi_a, bi_b = owner_batch[a_id], owner_batch[b_id]
        if bi_a == bi_b:
            continue
        n_merged += 1
        example_b = id_to_example[b_id]
        target_batch = batches[bi_a]
        source_batch = batches[bi_b]
        for idx, x in enumerate(source_batch):
            if x is example_b:
                del source_batch[idx]
                break
        target_batch.append(example_b)
        owner_batch[b_id] = bi_a
        max_len_target = max(len(x[0]) for x in target_batch)
        actual_tokens = len(target_batch) * max_len_target
        if actual_tokens > max_tokens:
            overflow_pcts.append(100.0 * (actual_tokens - max_tokens) / max_tokens)

    return {
        "n_pairs_present": n_pairs_present,
        "n_merged": n_merged,
        "n_overflowed": len(overflow_pcts),
        "overflow_pcts": overflow_pcts,
    }


def percentiles(values, ps=(50, 90, 99, 100)):
    if not values:
        return {f"p{p}": None for p in ps}
    arr = np.array(values)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    training_data = load_prepared_data_from_directory(LANG_DIR, _StubModelInterface())
    pair_ids = load_pair_ids(LANG_DIR / "pair-id.txt", len(training_data))
    pair_partner_map = build_pair_partner_map(training_data, pair_ids)
    print(f"loaded {len(training_data)} training examples, "
          f"{len(pair_partner_map) // 2} pairs", flush=True)

    results_by_budget = {}
    for budget in BUDGETS:
        seed_results = []
        for seed in SHUFFLE_SEEDS:
            examples = list(training_data)
            random.Random(seed).shuffle(examples)
            r = instrumented_merge(examples, budget, pair_partner_map)
            seed_results.append(r)
            print(f"  budget={budget} seed={seed}: "
                  f"n_pairs_present={r['n_pairs_present']} n_merged={r['n_merged']} "
                  f"n_overflowed={r['n_overflowed']}", flush=True)

        all_overflow_pcts = [p for r in seed_results for p in r["overflow_pcts"]]
        frac_requiring_merge = [r["n_merged"] / r["n_pairs_present"] for r in seed_results if r["n_pairs_present"]]
        frac_merge_overflowed = [r["n_overflowed"] / r["n_merged"] if r["n_merged"] else 0.0 for r in seed_results]

        results_by_budget[str(budget)] = {
            "per_seed": seed_results,
            "mean_fraction_pairs_requiring_merge": float(np.mean(frac_requiring_merge)),
            "mean_fraction_merges_that_overflowed": float(np.mean(frac_merge_overflowed)),
            "overflow_percent_over_budget_distribution": percentiles(all_overflow_pcts),
            "n_overflow_events_pooled_across_seeds": len(all_overflow_pcts),
        }
        print(f"budget={budget}: mean_frac_requiring_merge="
              f"{results_by_budget[str(budget)]['mean_fraction_pairs_requiring_merge']:.3f} "
              f"mean_frac_merge_overflowed="
              f"{results_by_budget[str(budget)]['mean_fraction_merges_that_overflowed']:.3f} "
              f"overflow_pct_p50={results_by_budget[str(budget)]['overflow_percent_over_budget_distribution']['p50']}",
              flush=True)

    # ------------------------------------------------------------------
    # empirical distribution of max_tokens_per_batch as actually sampled by
    # FLaRe's own random_sample.py (log-uniform over [128,4096], rounded to
    # int) -- to know how much probability mass falls in the small-budget
    # (worse-overflow) region.
    # ------------------------------------------------------------------
    n_draws = 20000
    draws = scipy.stats.loguniform(128, 4096).rvs(size=n_draws, random_state=42)
    draws_int = np.round(draws).astype(int)
    octave_edges = [128, 256, 512, 1024, 2048, 4096]
    octave_counts = {}
    for lo, hi in zip(octave_edges[:-1], octave_edges[1:]):
        frac = float(np.mean((draws_int >= lo) & (draws_int < hi)))
        octave_counts[f"[{lo},{hi})"] = frac
    octave_counts[f"[{octave_edges[-1]},{octave_edges[-1]}]"] = float(np.mean(draws_int == octave_edges[-1]))

    sampling_distribution = {
        "n_draws": n_draws,
        "distribution": "scipy.stats.loguniform(128, 4096), rounded to nearest int -- EXACTLY "
                        "matching train_one()'s random_sample(128, 4096, log=True, as_int=True) "
                        "call used for every fix-experiment training trial",
        "fraction_per_octave": octave_counts,
        "note": (
            "Log-uniform means each OCTAVE (doubling of the range) gets roughly EQUAL "
            "probability mass, not that small values are rare -- confirmed empirically: "
            "~20% of all 160 training trials are expected to land with max_tokens_per_batch "
            "in [128,256), the worst-overflow regime measured above, not a rare edge case."
        ),
        "percentiles": percentiles(draws_int.tolist(), ps=(10, 25, 50, 75, 90)),
    }
    print("\n=== max_tokens_per_batch sampling distribution ===")
    print(json.dumps(sampling_distribution, indent=2))

    # weighted overflow estimate: combine the per-budget overflow rates with
    # the actual sampling-distribution weights (using each octave's nearest
    # tested budget as a proxy)
    budget_to_octave_weight = {
        128: octave_counts["[128,256)"],
        256: octave_counts["[256,512)"],
        512: octave_counts["[512,1024)"],
        1024: octave_counts["[1024,2048)"],
        2048: octave_counts["[2048,4096)"],
        4096: octave_counts["[4096,4096]"],
    }
    weighted_frac_merges_overflowed = sum(
        results_by_budget[str(b)]["mean_fraction_merges_that_overflowed"] * w
        for b, w in budget_to_octave_weight.items()
    )
    weighted_frac_pairs_requiring_merge = sum(
        results_by_budget[str(b)]["mean_fraction_pairs_requiring_merge"] * w
        for b, w in budget_to_octave_weight.items()
    )

    out = {
        "experiment": "CPF2 overflow-rate characterization",
        "description": (
            "Quantifies how often ContrastiveRecognitionTrainingLoop's pair-merge step "
            "pushes a batch over max_tokens_per_batch, across the actual [128,4096] budget "
            "range and 5 shuffle seeds per budget, weighted by the REAL log-uniform sampling "
            "distribution random_sample.py uses for every fix-experiment training trial."
        ),
        "dataset": str(LANG_DIR),
        "budgets_tested": BUDGETS,
        "shuffle_seeds": SHUFFLE_SEEDS,
        "results_by_budget": results_by_budget,
        "max_tokens_per_batch_sampling_distribution": sampling_distribution,
        "queue_weighted_estimate": {
            "mean_fraction_pairs_requiring_cross_batch_merge": weighted_frac_pairs_requiring_merge,
            "mean_fraction_of_merges_that_overflow_budget": weighted_frac_merges_overflowed,
            "note": (
                "Weighted average of the per-budget overflow rates using each octave's actual "
                "probability mass under random_sample.py's log-uniform [128,4096] sampling -- "
                "this approximates the overflow rate expected across the real 160-run queue, "
                "using each octave's tested boundary budget as a representative proxy."
            ),
        },
    }
    out_path = RESULTS / "cpf2_overflow_check.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")
    print(f"\nQueue-weighted estimate: "
          f"{weighted_frac_pairs_requiring_merge:.1%} of pairs will require a cross-batch merge, "
          f"{weighted_frac_merges_overflowed:.1%} of those merges will overflow max_tokens_per_batch")


if __name__ == "__main__":
    main()
