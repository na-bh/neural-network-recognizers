"""A3: consolidated verdict table (task x architecture) + mechanism
characterization for shortcut-vulnerable tasks + marker-placement
decomposition for the 4 marker-based tasks.

Verdict thresholds (pre-registered in the A2 brief):
  shortcut-vulnerable        : logit diff < 0.1  AND hard-neg acc < 20%
  arithmetic/answer-sensitive: logit diff > 1.0  AND hard-neg acc > 70%
  partial-shortcut-with-ceiling: grammar-only ceiling is a real, specific
                                 accuracy value the architecture's test
                                 accuracy sits near (within 3pts), while
                                 logit-diff/hard-neg-acc are intermediate
  else                       : mixed / does not cleanly fit any bucket
                                 (reported as "mixed", not forced)
  test-unresolvable          : counterfactual test errored or no checkpoint

Uses long_L400 for the logit-diff figure (the harder, more diagnostic
regime) throughout.

PYTHONPATH=src:analysis python analysis/flare_a3_verdict.py
"""

import json
from pathlib import Path

RESULTS = Path("analysis_outputs/final_results")
LANG = Path("languages")

ALL_TASKS = [
    "parity", "marked-copy", "marked-reversal", "odds-first", "bucket-sort",
    "binary-addition", "binary-multiplication", "compute-sqrt", "dyck-2-3",
    "repeat-01", "stack-manipulation", "missing-duplicate-string",
    "unmarked-reversal",
]
MARKER_TASKS = ["marked-copy", "marked-reversal", "odds-first", "bucket-sort"]
ARCHS = ["rnn", "lstm", "transformer", "mamba"]


def load_a2(task):
    return json.loads((RESULTS / f"flare_a2_{task.replace('-', '_')}.json").read_text())


def classify_cell(logit_diff, hard_acc, test_acc, ceiling):
    if logit_diff is None or hard_acc is None or hard_acc != hard_acc:
        return "test-unresolvable"
    if logit_diff < 0.1 and hard_acc < 0.20:
        return "shortcut-vulnerable"
    if logit_diff > 1.0 and hard_acc > 0.70:
        return "arithmetic/answer-sensitive"
    if ceiling == ceiling and test_acc == test_acc and abs(test_acc - ceiling) <= 0.03:
        return "partial-shortcut-with-ceiling"
    return "mixed"


# ---------------------------------------------------------------- marker decomposition
def _marker_reason(seq, marker, transform, expected_pos_fn):
    n = len(seq)
    marker_positions = [i for i, t in enumerate(seq) if t == marker]
    if len(marker_positions) != 1:
        return "marker_count_wrong"
    i = marker_positions[0]
    if i != expected_pos_fn(n):
        return "marker_position_wrong"
    prefix, suffix = seq[:i], seq[i + 1:]
    expected_suffix = transform(prefix)
    if len(suffix) != len(expected_suffix):
        return "other"
    if suffix != expected_suffix:
        return "content_mismatch"
    return "other"  # shouldn't happen for a genuine negative


MARKER_SPECS = {
    "marked-copy": (lambda w: list(w), lambda n: (n - 1) // 2),
    "marked-reversal": (lambda w: list(reversed(w)), lambda n: (n - 1) // 2),
    "odds-first": (lambda w: w[::2] + w[1::2], lambda n: (n - 1) // 2),
    "bucket-sort": (lambda w: sorted(w), lambda n: (n - 1) // 2),
}


def decompose_marker_task(task, sample_cap=300):
    transform, expected_pos_fn = MARKER_SPECS[task]
    test_dir = LANG / task / "datasets" / "test"
    tok_lines = test_dir.joinpath("main.tok").read_text().splitlines()
    labels = [int(x) for x in test_dir.joinpath("labels.txt").read_text().splitlines()]

    from flare_a1_task_audit import TASKS as A1_TASKS
    viol_fn = A1_TASKS[task]["viol_fn"]

    scrambled = []
    for line, label in zip(tok_lines, labels):
        if label == 1:
            continue
        seq = line.split()
        if len(seq) == 0:
            continue
        if viol_fn(seq) < 0.8:  # matches A1's "scrambled" classification exactly
            scrambled.append(seq)
    import random
    rng = random.Random(0)
    sample = rng.sample(scrambled, min(sample_cap, len(scrambled)))

    reasons = {"marker_count_wrong": 0, "marker_position_wrong": 0,
              "content_mismatch": 0, "other": 0}
    for seq in sample:
        r = _marker_reason(seq, "#", transform, expected_pos_fn)
        reasons[r] += 1
    n = len(sample)
    fracs = {k: v / n for k, v in reasons.items()}
    return {
        "n_scrambled_total": len(scrambled), "n_sampled": n,
        "counts": reasons, "fractions": fracs,
        "marker_placement_frac": fracs["marker_count_wrong"] + fracs["marker_position_wrong"],
        "mechanism_confirmed_80pct": (fracs["marker_count_wrong"] + fracs["marker_position_wrong"]) >= 0.80,
    }


def main():
    print("=== A3: consolidated verdict table ===\n")
    a1 = json.loads((RESULTS / "flare_a1_task_audit.json").read_text())

    table = {}
    for task in ALL_TASKS:
        d = load_a2(task)
        ceiling = d.get("grammar_only_ceiling")
        row = {}
        for arch in ARCHS:
            if arch not in d.get("selected_seeds", {}):
                row[arch] = {"verdict": "test-unresolvable", "reason": "no checkpoint found"}
                continue
            logit_row = d["counterfactual_logit_difference"].get(arch, {}).get("long_L400", {})
            if "error" in logit_row:
                row[arch] = {"verdict": "test-unresolvable", "reason": logit_row["error"]}
                continue
            logit_diff = logit_row.get("mean_abs_delta_logit")
            hs = d["hard_vs_scrambled_negative_accuracy"].get(arch, {})
            hard_acc = hs.get("hard_negative_accuracy")
            test_acc = d["accuracy_table"].get(arch, {}).get(str(d["selected_seeds"][arch]))
            verdict = classify_cell(logit_diff, hard_acc, test_acc, ceiling)
            row[arch] = {
                "verdict": verdict, "logit_diff_long": logit_diff, "hard_neg_acc": hard_acc,
                "n_hard": hs.get("n_hard"), "scrambled_neg_acc": hs.get("scrambled_negative_accuracy"),
                "n_scrambled": hs.get("n_scrambled"), "long_seq_test_acc": test_acc,
                "grammar_only_ceiling": ceiling,
            }
        table[task] = row
        print(f"{task:28s} " + "  ".join(f"{a}={row[a]['verdict']}" for a in ARCHS))

    print("\n=== marker-placement mechanism decomposition ===")
    marker_decomp = {}
    for task in MARKER_TASKS:
        res = decompose_marker_task(task)
        marker_decomp[task] = res
        print(f"  {task:16s} n_sampled={res['n_sampled']:>4}  "
              f"marker_count_wrong={res['fractions']['marker_count_wrong']:.2f}  "
              f"marker_position_wrong={res['fractions']['marker_position_wrong']:.2f}  "
              f"content_mismatch={res['fractions']['content_mismatch']:.2f}  "
              f"other={res['fractions']['other']:.2f}  "
              f"-> marker_placement_total={res['marker_placement_frac']:.2f}  "
              f"CONFIRMED(>=80%)={res['mechanism_confirmed_80pct']}")

    all_marker_placement = [marker_decomp[t]["marker_placement_frac"] for t in MARKER_TASKS]
    overall_mechanism_confirmed = all(marker_decomp[t]["mechanism_confirmed_80pct"] for t in MARKER_TASKS)

    # ---------------------------------------------------------- special notes
    unmarked_rev = load_a2("unmarked-reversal")
    unmarked_note = {
        "finding": (
            "RNN and LSTM show EXACT ZERO logit-sensitivity (mean|delta_logit|=0.0, "
            "frac_corrupt_accepted=1.0) to a flip of the OUTERMOST mirror pair (last "
            "token vs. first token) at BOTH short (L20) and long (L400) lengths, despite "
            "60.7% accuracy on the broader real hard-negative population (which spans "
            "flips at many positions, not just the outermost pair). Transformer and Mamba "
            "show strong sensitivity to the same flip (delta_logit 6-8, 0% accepted)."
        ),
        "interpretation": (
            "This is exactly the 'shortcut A1 didn't anticipate' scenario: a narrow, "
            "position-specific blind spot (the single outermost mirror pair) in two "
            "architectures, invisible to A1's structural audit because A1 only measures "
            "structural properties of the string, and A2's population-level hard-negative "
            "accuracy averages over many flip positions (staying moderate at 60.7%), "
            "diluting a failure that is total at exactly one position."
        ),
        "separate_puzzle_worth_flagging": (
            "scrambled_negative_accuracy is ALSO surprisingly low for RNN/LSTM/Mamba "
            "(0%, 0%, 3.2%) -- i.e. these architectures accept most scrambled (obviously "
            "malformed) negatives as valid, the opposite of the usual pattern where "
            "scrambled negatives are the easy category. Only transformer (33.8%) partially "
            "escapes this. Not further diagnosed here (out of A2/A3 scope) -- flagged as "
            "an anomaly rather than averaged away, per the discipline for this experiment."
        ),
        "rnn_long_L400": unmarked_rev["counterfactual_logit_difference"]["rnn"]["long_L400"],
        "lstm_long_L400": unmarked_rev["counterfactual_logit_difference"]["lstm"]["long_L400"],
        "transformer_long_L400": unmarked_rev["counterfactual_logit_difference"]["transformer"]["long_L400"],
        "mamba_long_L400": unmarked_rev["counterfactual_logit_difference"]["mamba"]["long_L400"],
        "population_hard_neg_acc": unmarked_rev["hard_vs_scrambled_negative_accuracy"],
    }

    out = {
        "description": "A3: consolidated task x architecture shortcut verdict table for the 13 audited FLaRe tasks.",
        "verdict_thresholds": {
            "shortcut-vulnerable": "logit_diff_long < 0.1 AND hard_neg_acc < 0.20",
            "arithmetic/answer-sensitive": "logit_diff_long > 1.0 AND hard_neg_acc > 0.70",
            "partial-shortcut-with-ceiling": "test accuracy within 3pts of the grammar-only ceiling",
            "mixed": "does not cleanly fit any of the above",
            "test-unresolvable": "no checkpoint, or counterfactual test errored",
        },
        "table": table,
        "marker_placement_decomposition": marker_decomp,
        "marker_placement_mechanism_confirmed_across_all_4_tasks": overall_mechanism_confirmed,
        "unmarked_reversal_special_note": unmarked_note,
    }
    out_path = RESULTS / "flare_a3_verdict.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")
    print("A3 done.")


if __name__ == "__main__":
    main()
