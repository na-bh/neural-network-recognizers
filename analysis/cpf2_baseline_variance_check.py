"""Baseline comparison for CPF2's overflow finding: characterizes the
EFFECTIVE-BATCH-SIZE variance standard (non-paired) FLaRe training already
produces on the ORIGINAL corrected marked-reversal dataset, across the same
budget range and shuffle-seed set used in cpf2_overflow_check.py. Standard
group_into_batches NEVER exceeds max_tokens_per_batch by construction (a
batch that would overflow simply isn't grown further), so this reports how
far BELOW budget batches typically land (as a fraction of max_tokens_per_
batch) -- establishing whether standard training already tolerates
substantial batch-size variance, or runs consistently near-full, which
would make CPF2's overflow-above-budget a categorically new kind of
variance rather than a variation on an existing one.

PYTHONPATH=src:analysis python analysis/cpf2_baseline_variance_check.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from recognizers.neural_networks.data import load_prepared_data_from_directory
from recognizers.neural_networks.batching import group_into_batches

RESULTS = Path("analysis_outputs/final_results")
LANG_DIR = Path("languages/marked-reversal-fixed")
BUDGETS = [128, 256, 512, 1024, 2048, 4096]
SHUFFLE_SEEDS = [1001, 1002, 1003, 1004, 1005]


class _StubModelInterface:
    use_next_symbols_head = False


def percentiles(values, ps=(5, 50, 95)):
    arr = np.array(values)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    training_data = load_prepared_data_from_directory(LANG_DIR, _StubModelInterface())
    print(f"loaded {len(training_data)} training examples (standard, non-paired)", flush=True)

    results_by_budget = {}
    for budget in BUDGETS:
        is_small_enough = lambda b, n: b * n <= budget
        all_ratios = []
        all_ratios_by_seed = {}
        for seed in SHUFFLE_SEEDS:
            examples = list(training_data)
            random.Random(seed).shuffle(examples)
            batches = list(group_into_batches(examples, is_small_enough))
            ratios = []
            for batch in batches:
                max_len = max(len(x[0]) for x in batch)
                actual_tokens = len(batch) * max_len
                ratios.append(actual_tokens / budget)
            all_ratios_by_seed[seed] = {
                "n_batches": len(batches),
                "mean_ratio": float(np.mean(ratios)),
                "std_ratio": float(np.std(ratios)),
            }
            all_ratios.extend(ratios)
            assert max(ratios) <= 1.0 + 1e-9, "standard batching must never exceed budget"

        results_by_budget[str(budget)] = {
            "per_seed": all_ratios_by_seed,
            "pooled_mean_fraction_of_budget_used": float(np.mean(all_ratios)),
            "pooled_std_fraction_of_budget_used": float(np.std(all_ratios)),
            "pooled_percentiles_fraction_of_budget_used": percentiles(all_ratios),
            "n_batches_pooled": len(all_ratios),
        }
        r = results_by_budget[str(budget)]
        print(f"budget={budget}: mean_frac_of_budget_used={r['pooled_mean_fraction_of_budget_used']:.3f} "
              f"std={r['pooled_std_fraction_of_budget_used']:.3f} "
              f"p5={r['pooled_percentiles_fraction_of_budget_used']['p5']:.3f} "
              f"p50={r['pooled_percentiles_fraction_of_budget_used']['p50']:.3f} "
              f"p95={r['pooled_percentiles_fraction_of_budget_used']['p95']:.3f}", flush=True)

    out = {
        "experiment": "CPF2 baseline variance check -- standard (non-paired) batch-size variance",
        "description": (
            "Characterizes how much batch-size variance standard FLaRe training ALREADY "
            "produces (as a fraction of max_tokens_per_batch, always <=1.0 by construction), "
            "on the same marked-reversal task and the same budget/seed grid as cpf2_overflow_"
            "check.json's contrastive-pairing overflow measurements, to give the CPF2 overflow "
            "numbers a baseline for comparison."
        ),
        "dataset": str(LANG_DIR),
        "budgets_tested": BUDGETS,
        "shuffle_seeds": SHUFFLE_SEEDS,
        "results_by_budget": results_by_budget,
    }
    out_path = RESULTS / "cpf2_baseline_variance_check.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
