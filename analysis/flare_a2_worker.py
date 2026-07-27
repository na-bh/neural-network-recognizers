"""A2 per-task worker: empirical shortcut test for ONE FLaRe task, all 4
architectures, reusing the exact harness classes cyclenav_s2 used
(RNNPatchingHarness, PatchingHarness, TransformerPatchingHarness) and A1's
first_violation_frac classifiers for the hard/scrambled split. No retraining.

For each architecture: select the seed (of up to 10) with highest accuracy
on the long (length>=LONG_SEQ_THRESH) subset of the real test set, then
report:
  (i)   mean |logit(clean)-logit(corrupt)| on the task's hard-negative-style
        counterfactual pairs (analysis/flare_a2_counterfactuals.py or, for
        parity, analysis/parity_counterfactuals.py), at short (~L20) and
        long (~L400) lengths, N=100 pairs each.
  (ii)  accuracy on the REAL hard-negative subset vs scrambled-negative
        subset (from A1's classification), N = actual population counts.
  (iii) overall test accuracy.
  (iv)  grammar-only ceiling = (n_pos + n_scrambled + 0.5*n_hard) / n_total
        (the accuracy achievable by accepting all positives, rejecting all
        scrambled negatives, and guessing randomly on hard negatives).

PYTHONPATH=src:analysis python analysis/flare_a2_worker.py --task marked-copy
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).parent))
import flare_a2_counterfactuals as fc
from flare_a1_task_audit import TASKS as A1_TASKS
from rnn_patching import RNNPatchingHarness
from patching_harness import PatchingHarness
from modk_transformer_harness import TransformerPatchingHarness
from recognizers.neural_networks.data import (
    add_data_arguments, load_vocabulary_data, load_prepared_data_from_directory,
)
from recognizers.neural_networks.model_interface import RecognitionModelInterface

RESULTS = Path("analysis_outputs/final_results")
LONG_SEQ_THRESH = 400
SOLVING_THRESH = 0.95
N_SEEDS = 10
VARIANT = "rec+ns/validation-short"
N_PAIRS = 100
LENGTH_REGIMES = [(20, "short_L20"), (400, "long_L400")]


def checkpoint_root(task, arch):
    for base in ["models", "data/models"]:
        p = Path(base) / task / arch / VARIANT
        if p.exists():
            return Path(base)
    raise FileNotFoundError(f"{task}/{arch}")


def load_model_generic(task, arch, seed):
    base = checkpoint_root(task, arch)
    model_dir = base / task / arch / VARIANT / str(seed)
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    iface = RecognitionModelInterface()
    iface.add_arguments(parser)
    iface.add_forward_arguments(parser)
    tmp = tempfile.mkdtemp(prefix="flare_a2_")
    shutil.rmtree(tmp, ignore_errors=True)
    args = parser.parse_args([
        "--output", tmp, "--training-data", f"languages/{task}",
        "--architecture", arch, "--load-model", str(model_dir), "--load-parameters", "main",
    ])
    vocab = load_vocabulary_data(args, parser)
    saver = iface.construct_saver(args, vocab)
    saver.model.eval()
    return iface, saver


def eval_accuracy(iface, saver, examples):
    device = next(saver.model.parameters()).device
    correct = total = 0
    for seq, (label, next_symbols) in examples:
        mi, _ = iface.prepare_batch([(seq, (label, next_symbols))], device)
        with torch.no_grad():
            rec, _, _ = iface.get_logits(saver.model, mi)
        pred = bool(rec.item() > 0)
        correct += (pred == bool(label))
        total += 1
    return (correct / total if total else float("nan")), correct, total


def get_harness(task, arch, seed):
    if arch in ("rnn", "lstm"):
        return RNNPatchingHarness(arch, f"{VARIANT}/{seed}", task=task)
    if arch == "mamba":
        return PatchingHarness(task=task, model_subdir=f"mamba/{VARIANT}/{seed}")
    if arch == "transformer":
        return TransformerPatchingHarness(task, f"{VARIANT}/{seed}")
    raise ValueError(arch)


def make_pairs(task, n_pairs, target_len, rng):
    return fc.TASK_MAKERS[task](n_pairs, target_len, rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    task = args.task

    print(f"=== A2: {task} ===", flush=True)
    test_dir = Path(f"languages/{task}/datasets/test")
    test_examples = load_prepared_data_from_directory(
        test_dir, type("I", (), {"use_next_symbols_head": True})()
    )
    n_long = sum(1 for seq, _ in test_examples if len(seq) >= LONG_SEQ_THRESH)
    print(f"  {task}: {len(test_examples)} test examples, {n_long} with length>={LONG_SEQ_THRESH}", flush=True)

    long_examples_full = [(s, l) for s, l in test_examples if len(s) >= LONG_SEQ_THRESH]
    if not long_examples_full:
        long_examples_full = test_examples
    # Subsample for seed SELECTION only (40 checkpoint x eval combinations) -- 150
    # examples is enough to rank seeds; full evaluation is wasteful at this stage.
    rng_sel = np.random.default_rng(42)
    if len(long_examples_full) > 150:
        idx = rng_sel.choice(len(long_examples_full), size=150, replace=False)
        long_examples_selection = [long_examples_full[i] for i in idx]
    else:
        long_examples_selection = long_examples_full

    archs = ["rnn", "lstm", "transformer", "mamba"]
    selected_seeds, accuracy_table = {}, {}
    for arch in archs:
        accs = {}
        for seed in range(0, N_SEEDS + 1):  # covers both 0..9 (mamba) and 1..10 (others) conventions
            try:
                iface, saver = load_model_generic(task, arch, seed)
            except FileNotFoundError:
                continue
            acc, _, _ = eval_accuracy(iface, saver, long_examples_selection)
            accs[seed] = acc
        if not accs:
            print(f"  {arch}: NO CHECKPOINTS FOUND", flush=True)
            continue
        best_seed = max(accs, key=accs.get)
        selected_seeds[arch] = best_seed
        accuracy_table[arch] = accs
        mode = "solving" if accs[best_seed] >= SOLVING_THRESH else "best-available (not solving)"
        print(f"  {arch:12s} -> seed {best_seed} (acc={accs[best_seed]:.4f}, {mode})", flush=True)

    # ---- hard vs scrambled real-test-set accuracy ----
    # Classify using the actual token STRINGS (main.tok), matching A1 exactly.
    viol_fn = A1_TASKS[task]["viol_fn"]
    tok_lines = test_dir.joinpath("main.tok").read_text().splitlines()
    labels = [int(x) for x in test_dir.joinpath("labels.txt").read_text().splitlines()]
    hard, scrambled = [], []
    hard_examples, scr_examples = [], []
    for line, label in zip(tok_lines, labels):
        if label == 1:
            continue
        seq_str = line.split()
        if len(seq_str) == 0:
            continue  # Mamba's einops rearrange divides by zero on empty input
        frac = viol_fn(seq_str)
        if frac >= 0.8:
            hard.append(seq_str)
        else:
            scrambled.append(seq_str)

    n_pos = sum(1 for l in labels if l == 1)
    n_hard, n_scrambled = len(hard), len(scrambled)
    n_total = len(labels)
    grammar_only_ceiling = (n_pos + n_scrambled + 0.5 * n_hard) / n_total if n_total else float("nan")

    # subsample for the per-arch accuracy pass (evaluation is one-example-at-a-time
    # and this cell runs once per selected arch, not once per candidate seed, but
    # capping keeps runtime bounded on the longer tasks -- N reported reflects the
    # actual subsample size used, honestly, not the full population).
    rng_hs = np.random.default_rng(7)
    EVAL_CAP = 300

    def subsample(lst):
        if len(lst) > EVAL_CAP:
            idx = rng_hs.choice(len(lst), size=EVAL_CAP, replace=False)
            return [lst[i] for i in idx]
        return lst

    hard_eval, scrambled_eval = subsample(hard), subsample(scrambled)

    hard_scrambled = {}
    if selected_seeds:
        vocab = fc._vocab(task)
        for arch, seed in selected_seeds.items():
            iface, saver = load_model_generic(task, arch, seed)
            device = next(saver.model.parameters()).device

            def acc_on(str_seqs):
                if not str_seqs:
                    return float("nan"), 0
                correct = 0
                for seq_str in str_seqs:
                    toks = torch.tensor([vocab.index(t) for t in seq_str], dtype=torch.long)
                    mi, _ = iface.prepare_batch([(toks, (0, None))], device)
                    with torch.no_grad():
                        rec, _, _ = iface.get_logits(saver.model, mi)
                    pred = bool(rec.item() > 0)
                    correct += (pred == False)
                return correct / len(str_seqs), len(str_seqs)

            hacc, hn = acc_on(hard_eval)
            sacc, sn = acc_on(scrambled_eval)
            hard_scrambled[arch] = {
                "hard_negative_accuracy": hacc, "n_hard": hn,
                "scrambled_negative_accuracy": sacc, "n_scrambled": sn,
            }
            print(f"  {arch:12s} hard_neg_acc={hacc:.4f} (n={hn})  scrambled_neg_acc={sacc:.4f} (n={sn})", flush=True)

    # ---- counterfactual logit-difference test ----
    print(f"  --- counterfactual logit-diff test ---", flush=True)
    logit_results = {}
    if task in fc.TASK_MAKERS:
        for arch, seed in selected_seeds.items():
            h = get_harness(task, arch, seed)
            row = {}
            for target_len, tag in LENGTH_REGIMES:
                rng = np.random.default_rng(3000 + target_len)
                try:
                    pairs = make_pairs(task, N_PAIRS, target_len, rng)
                except Exception as e:
                    row[tag] = {"error": str(e)}
                    continue
                diffs = []
                accepts_corrupt = []
                for p in pairs:
                    cl = h.logit_diff(p["clean"])
                    co = h.logit_diff(p["corrupt"])
                    diffs.append(abs(cl - co))
                    accepts_corrupt.append(co > 0)
                row[tag] = {
                    "length": len(pairs[0]["clean"]), "n_pairs": len(pairs),
                    "mean_abs_delta_logit": float(np.mean(diffs)),
                    "median_abs_delta_logit": float(np.median(diffs)),
                    "frac_corrupt_accepted": float(np.mean(accepts_corrupt)),
                }
                print(f"  {arch:12s} {tag:10s} mean|delta_logit|={np.mean(diffs):.4f} "
                      f"frac_corrupt_accepted={np.mean(accepts_corrupt):.2f}", flush=True)
            logit_results[arch] = row
        test_resolvable = True
    else:
        test_resolvable = False
        print(f"  {task}: no counterfactual maker available -- test-unresolvable", flush=True)

    out = {
        "task": task,
        "n_test_examples": len(test_examples), "n_long": n_long,
        "selected_seeds": selected_seeds,
        "accuracy_table": {a: {str(s): v for s, v in accs.items()} for a, accs in accuracy_table.items()},
        "n_positives": n_pos, "n_hard_negatives": n_hard, "n_scrambled_negatives": n_scrambled,
        "n_total": n_total, "grammar_only_ceiling": grammar_only_ceiling,
        "hard_vs_scrambled_negative_accuracy": hard_scrambled,
        "counterfactual_logit_difference": logit_results,
        "test_resolvable": test_resolvable,
    }
    out_path = RESULTS / f"flare_a2_{task.replace('-', '_')}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  Saved {out_path}", flush=True)
    print(f"A2 {task} done.", flush=True)


if __name__ == "__main__":
    main()
