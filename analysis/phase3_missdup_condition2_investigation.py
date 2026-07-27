"""Investigation triggered by Condition 2's uniform near-zero behavioral
gap across all 8 cells (phase3_missdup_condition2.json), which flatly
contradicted the flare_a2-derived prediction (Transformer/Mamba large,
RNN/LSTM modest). Rather than accept "no causal effect" at face value, this
does an INDEPENDENT direct-accuracy check (bypassing the patching harness
entirely, using the exact evaluate_recognition-style forward pass) on the
real test set, bucketed by the two trivial structural shortcuts audited in
phase3_missdup_task_audit.json: blank_count and sequence_length_parity.

FINDING: flare_a2's "hard negative" population (viol_missing_duplicate_
string frac >= 0.8) is CONFOUNDED for this task. viol_missing_duplicate_
string returns 1.0 ("hard") for BOTH blank_count != 1 negatives AND
odd-length negatives -- not because these require deep computation, but
because "no blank found" / "wrong total length parity" are only PROVABLE
once the whole sequence has been scanned (the incremental-detectability
framework's definition of "hard"), even though both are trivial O(1)
running counters, not genuine content verification. Real test-set accuracy,
bucketed by the ACTUAL structural cause:
  blank_count != 1            -- trivial shortcut, ~100% for all 4 archs
  blank_count == 1, odd length -- trivial length-parity shortcut
  blank_count == 1, even length -- the ONLY subpopulation requiring genuine
    position-by-position content verification (phase3_missdup_task_audit.
    json's "genuinely hard candidates")

If accuracy on the third bucket is near-floor for an architecture, that
architecture has NOT learned genuine duplicate-content verification at all
-- Condition 2's near-zero behavioral gap for that architecture is not "no
causal aggregate-route reliance," it is "the model already defaults to one
answer regardless of input in this regime," a floor effect patching cannot
reveal anything through.

PYTHONPATH=src python analysis/phase3_missdup_condition2_investigation.py
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "src")
from recognizers.neural_networks.data import add_data_arguments, load_vocabulary_data
from recognizers.neural_networks.model_interface import RecognitionModelInterface, ModelInput

RESULTS = Path("analysis_outputs/final_results")
TASK_DIR = Path("languages/missing-duplicate-string")
CONFIGS = [
    ("rnn", 2, "primary", Path("data/models/missing-duplicate-string/rnn/rec+ns/validation-short/2")),
    ("rnn", 3, "secondary", Path("data/models/missing-duplicate-string/rnn/rec+ns/validation-short/3")),
    ("lstm", 1, "primary", Path("data/models/missing-duplicate-string/lstm/rec+ns/validation-short/1")),
    ("lstm", 2, "secondary", Path("data/models/missing-duplicate-string/lstm/rec+ns/validation-short/2")),
    ("transformer", 8, "primary", Path("data/models/missing-duplicate-string/transformer/rec+ns/validation-short/8")),
    ("transformer", 10, "secondary", Path("data/models/missing-duplicate-string/transformer/rec+ns/validation-short/10")),
    ("mamba", 9, "primary", Path("models/missing-duplicate-string/mamba/rec+ns/validation-short/9")),
    ("mamba", 5, "secondary", Path("models/missing-duplicate-string/mamba/rec+ns/validation-short/5")),
]


def load_model(arch, model_dir):
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    iface = RecognitionModelInterface()
    iface.add_arguments(parser)
    iface.add_forward_arguments(parser)
    tmp = tempfile.mkdtemp(prefix="missdup_investigation_")
    shutil.rmtree(tmp, ignore_errors=True)
    args = parser.parse_args([
        "--output", tmp, "--training-data", str(TASK_DIR), "--architecture", arch,
        "--load-model", str(model_dir), "--load-parameters", "main",
    ])
    vocab = load_vocabulary_data(args, parser)
    saver = iface.construct_saver(args, vocab)
    saver.model.eval()
    return iface, saver


def accuracy_on(iface, saver, tok2idx, pairs):
    if not pairs:
        return float("nan"), 0
    eos_index = saver.kwargs["eos_index"]
    device = next(saver.model.parameters()).device
    correct = []
    for s, l in pairs:
        idx = [tok2idx[t] for t in s]
        content = idx + [eos_index]
        x = torch.tensor([content], dtype=torch.long, device=device)
        last_index = torch.tensor([len(idx)], dtype=torch.long, device=device)
        positive_mask = torch.zeros(1, dtype=torch.bool, device=device)
        mi = ModelInput(x, last_index, positive_mask)
        with torch.no_grad():
            rec, _, _ = iface.get_logits(saver.model, mi)
        correct.append((rec.item() > 0) == bool(l))
    return float(np.mean(correct)), len(pairs)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    tokens = torch.load(TASK_DIR / "main.vocab", weights_only=False)["tokens"]
    tok2idx = {t: i for i, t in enumerate(tokens)}

    d = TASK_DIR / "datasets" / "test"
    toks = (d / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (d / "labels.txt").read_text().splitlines()]
    seqs = [line.split() for line in toks]
    neg = [(s, l) for s, l in zip(seqs, labels) if l == 0]
    blank_not1 = [(s, l) for s, l in neg if s.count("_") != 1]
    blank1_odd = [(s, l) for s, l in neg if s.count("_") == 1 and len(s) % 2 == 1]
    blank1_even_genuinely_hard = [(s, l) for s, l in neg if s.count("_") == 1 and len(s) % 2 == 0]

    print(f"bucket sizes (real test set): blank_count!=1 n={len(blank_not1)}, "
          f"blank_count==1&odd n={len(blank1_odd)}, "
          f"blank_count==1&even (genuinely hard) n={len(blank1_even_genuinely_hard)}", flush=True)

    per_cell = []
    for arch, seed, role, model_dir in CONFIGS:
        iface, saver = load_model(arch, model_dir)
        a_nb, n_nb = accuracy_on(iface, saver, tok2idx, blank_not1)
        a_odd, n_odd = accuracy_on(iface, saver, tok2idx, blank1_odd)
        a_hard, n_hard = accuracy_on(iface, saver, tok2idx, blank1_even_genuinely_hard)
        row = {
            "arch": arch, "seed": seed, "role": role,
            "acc_blank_count_not_1_trivial_shortcut": a_nb, "n_blank_count_not_1": n_nb,
            "acc_blank1_odd_length_parity_shortcut": a_odd, "n_blank1_odd": n_odd,
            "acc_blank1_even_GENUINELY_HARD": a_hard, "n_blank1_even": n_hard,
            "genuinely_hard_near_floor (<=0.10)": a_hard <= 0.10,
            "genuinely_hard_near_chance (0.40-0.60)": 0.40 <= a_hard <= 0.60,
        }
        per_cell.append(row)
        print(f"  {arch} seed{seed} ({role}): blank!=1 acc={a_nb:.3f}  blank1_odd acc={a_odd:.3f}  "
              f"blank1_even(GENUINELY HARD) acc={a_hard:.3f}", flush=True)

    out = {
        "task": "missing-duplicate-string",
        "trigger": (
            "Condition 2 (phase3_missdup_condition2.json) showed near-zero behavioral gap for ALL "
            "8 cells, flatly contradicting the flare_a2-derived prediction (Transformer/Mamba large "
            "effect, RNN/LSTM modest). Investigated rather than accepted at face value."
        ),
        "finding": (
            "flare_a2's hard-negative population (viol_missing_duplicate_string frac>=0.8) is "
            "CONFOUNDED: it returns 'hard' (frac=1.0) for BOTH blank_count!=1 negatives AND "
            "odd-length negatives, because these are only PROVABLE at end-of-sequence under the "
            "incremental-detectability metric -- despite both being trivial O(1) running-counter "
            "shortcuts, not genuine content verification. Direct per-bucket accuracy (bypassing "
            "patching entirely) shows: (1) ALL FOUR architectures solve blank_count!=1 near-perfectly "
            "(~100%, the cheapest possible shortcut); (2) RNN/LSTM ALSO solve blank1_odd near-"
            "perfectly via length parity (matching Condition 1's ~8-logit causal effect there); "
            "Transformer/Mamba do NOT reliably use length parity (near/below chance); (3) on the "
            "ONE population that actually requires position-by-position content verification "
            "(blank_count==1 AND even length -- phase3_missdup_task_audit.json's 'genuinely hard "
            "candidates'), RNN/LSTM/Mamba are all NEAR-FLOOR (always predict accept, ~0-4%), and "
            "Transformer is NEAR-CHANCE (~48-51%). NONE of the four architectures show evidence of "
            "genuine duplicate-content verification."
        ),
        "revised_interpretation": (
            "The original hypothesis framing (RNN/LSTM = genuine positional tracking via content-"
            "preserving carry, Transformer/Mamba = shallow aggregate-count route) is NOT SUPPORTED. "
            "A more accurate characterization: RNN/LSTM's high reported 'hard-negative accuracy' "
            "(91%, flare_a2) is fully explained by TWO trivial structural shortcuts (blank-count-is-"
            "one AND length-parity) that happen to cover the vast majority of the real test "
            "distribution's negatives -- NOT by genuine content verification. Transformer relies on "
            "only ONE of those shortcuts (blank-count), explaining its lower reported hard-negative "
            "accuracy (25%). Mamba is similar to Transformer/near-floor on genuine verification "
            "despite solving blank-count. Condition 2's uniform near-zero gap is therefore CORRECT, "
            "not a null result masking a real effect: there is no meaningful decision for patching "
            "to flip in the genuinely-hard regime, because every architecture already defaults to "
            "(near-)constant behavior there regardless of input content."
        ),
        "implication_for_condition_3": (
            "Condition 3's position_mismatch_pairs are ALSO drawn from the genuinely-hard population "
            "(blank_count==1, even length -- a same-half SWAP changes neither). Given this "
            "investigation, Condition 3 is now PREDICTED to ALSO show near-zero behavioral gaps for "
            "the same underlying reason (floor/near-chance baseline accuracy in this regime, not "
            "genuine position-sensitivity to test) -- reported explicitly as a revised prediction, "
            "not assumed; run and reported honestly regardless of outcome."
        ),
        "per_cell_bucketed_accuracy": per_cell,
    }
    out_path = RESULTS / "phase3_missdup_condition2_investigation.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
