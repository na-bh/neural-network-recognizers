"""A1: specification audit of the 16 not-yet-checked FLaRe tasks, mirroring
the hard-vs-scrambled-negative classification used for modular-arithmetic-simple
and cycle-navigation (analysis/cyclenav_s2_shortcut_check.py's is_grammatical).

Generalized concept: for each task, define first_violation_frac(seq) -- the
position (as a fraction of sequence length) of the EARLIEST point at which a
left-to-right incremental check of the task's own well-formedness rule can
prove the sequence is not accepted, ignoring whether the "answer" itself
(the specific target value being computed) happens to be right. This exactly
generalizes cyclenav's is_grammatical (there, the check is "are all body
tokens move-type and the last token digit-type" -- a violation is only
located at the digit position when the *type* is wrong, and otherwise the
mismatch is a wrong VALUE, not a wrong type, so cyclenav's grammatical set is
larger than the always-correct set).

A negative is classified HARD if first_violation_frac >= 0.8 (the deviation
only becomes provable very late -- structurally plausible, requires tracking
the actual computation to catch), SCRAMBLED otherwise (provably wrong early,
catchable via a shallow/local check). Some tasks have a genuinely GLOBAL
answer (parity, majority) where no incremental check can locate a violation
before the end at all -- first_violation_frac is always 1.0 for these by
construction, and they are reported as structurally shortcut-free rather than
empirically 80%+ hard by coincidence.

Reads real generated test data directly (languages/<task>/datasets/test/),
computes exact per-task hard/scrambled fractions -- no model inference, no
retraining (per the A1 brief).

PYTHONPATH=src python analysis/flare_a1_task_audit.py
"""

import json
import math
from pathlib import Path

import torch

RESULTS = Path("analysis_outputs/final_results")
LANG = Path("languages")
HARD_THRESHOLD = 0.8


def load_test(task):
    d = LANG / task / "datasets" / "test"
    toks = (d / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (d / "labels.txt").read_text().splitlines()]
    return toks, labels


def load_vocab(task):
    vd = torch.load(LANG / task / "main.vocab", weights_only=False)
    return vd["tokens"]


def idx(vocab, tok):
    return vocab.index(tok)


# ----------------------------------------------------------------------
# per-task first_violation_frac(seq: list[str]) -> float in [0,1]
# ----------------------------------------------------------------------

def viol_even_pairs(seq):
    # first == last is a purely global property; unprovable before the end.
    return 1.0


def viol_parity(seq):
    return 1.0  # cumulative XOR; unprovable before the end.


def viol_repeat_01(seq):
    n = len(seq)
    for i, tok in enumerate(seq):
        expected = "0" if i % 2 == 0 else "1"
        if tok != expected:
            return i / n
    # alternates perfectly throughout; only provable non-accepting if length is odd
    return 1.0 if n % 2 == 1 else 0.0  # 0.0 unreachable: perfectly-alternating even-length is positive


def viol_first(seq):
    # single-bit rule -- violation (if any) is always at position 0.
    return 0.0 if seq and seq[0] != "1" else 1.0


def viol_majority(seq):
    return 1.0  # global count comparison; unprovable before the end.


def viol_unmarked_reversal(seq):
    n = len(seq)
    if n % 2 == 1:
        return 1.0  # odd length only provable wrong at EOS
    half = n // 2
    for i in range(half):
        if seq[i] != seq[n - 1 - i]:
            return (n - 1 - i) / n  # provable only once its mirror position is seen
    return 1.0


def _marker_transform_violation(seq, marker, transform):
    n = len(seq)
    marker_positions = [i for i, t in enumerate(seq) if t == marker]
    if len(marker_positions) == 0:
        return 1.0  # only provable "missing marker" at EOS
    if len(marker_positions) >= 2:
        return marker_positions[1] / n  # provably too many markers as soon as the 2nd appears
    i = marker_positions[0]
    prefix, suffix = seq[:i], seq[i + 1:]
    expected_suffix = transform(prefix)
    m = min(len(suffix), len(expected_suffix))
    for j in range(m):
        if suffix[j] != expected_suffix[j]:
            return (i + 1 + j) / n
    if len(suffix) != len(expected_suffix):
        return (i + 1 + m) / n if m < max(len(suffix), len(expected_suffix)) else 1.0
    return 1.0


def viol_marked_reversal(seq):
    return _marker_transform_violation(seq, "#", lambda w: list(reversed(w)))


def viol_marked_copy(seq):
    return _marker_transform_violation(seq, "#", lambda w: list(w))


def viol_odds_first(seq):
    return _marker_transform_violation(seq, "#", lambda w: w[::2] + w[1::2])


def viol_bucket_sort(seq):
    return _marker_transform_violation(seq, "#", lambda w: sorted(w))


def viol_missing_duplicate_string(seq):
    n = len(seq)
    missing_positions = [i for i, t in enumerate(seq) if t == "_"]
    if len(missing_positions) == 0:
        return 1.0
    if len(missing_positions) >= 2:
        return missing_positions[1] / n
    if n % 2 == 1:
        return 1.0
    half = n // 2
    i = missing_positions[0]
    ss = list(seq)
    ss[i] = "1"
    for j in range(half):
        if ss[j] != ss[half + j]:
            # provable once the LATER of the mirrored pair (j, half+j) is seen,
            # unless the missing slot itself is one of the pair (then only
            # provable once the *other* member of the pair is seen)
            other = half + j if j < half else j - half
            return max(j, half + j) / n
    return 1.0


def viol_stack_manipulation(seq):
    PUSH, POP, MARKER = "PUSH", "POP", "#"
    n = len(seq)
    i = 0
    stack = []
    while i < n and seq[i] in ("0", "1"):
        stack.append(seq[i])
        i += 1
    while i < n and seq[i] in (PUSH, POP):
        if seq[i] == PUSH:
            i += 1
            if i < n and seq[i] in ("0", "1"):
                stack.append(seq[i])
                i += 1
            else:
                return i / n  # PUSH not followed by a bit -- provable right there
        else:  # POP
            if not stack:
                return i / n  # provable right here: pop on empty stack
            stack.pop()
            i += 1
    if i < n and seq[i] == MARKER:
        i += 1
    else:
        return 1.0 if i >= n else i / n  # missing marker where ops end
    expected = list(reversed(stack))
    suffix = seq[i:]
    m = min(len(suffix), len(expected))
    for j in range(m):
        if suffix[j] != expected[j]:
            return (i + j) / n
    if len(suffix) != len(expected):
        return 1.0
    return 1.0


def viol_dyck(seq):
    n = len(seq)
    stack = []  # holds bracket "type" strings, e.g. '0','1'
    for i, tok in enumerate(seq):
        if tok.startswith("("):
            stack.append(tok[1:])
        elif tok.startswith(")"):
            t = tok[1:]
            if not stack:
                return i / n  # closing with nothing open -- provable right here
            top = stack.pop()
            if top != t:
                return i / n  # mismatched closing type -- provable right here
        else:
            return i / n
    return 1.0 if stack else 0.0  # unclosed brackets at EOS: only provable at the end


def viol_binary_arith(seq):
    # x_bits OP y_bits = z_bits  (LSB-first); OP in {'+','x'}
    n = len(seq)
    op_positions = [i for i, t in enumerate(seq) if t in ("+", "×")]
    eq_positions = [i for i, t in enumerate(seq) if t == "="]
    if len(op_positions) == 0:
        return 1.0
    if len(op_positions) >= 2:
        return op_positions[1] / n
    op_i = op_positions[0]
    later_eq = [e for e in eq_positions if e > op_i]
    if len(eq_positions) == 0:
        return 1.0
    if len(eq_positions) >= 2 or not later_eq:
        # extra '=' before the operator, or 2+ total -- provable at the 2nd occurrence overall
        all_special_sorted = sorted(op_positions + eq_positions)
        return all_special_sorted[1] / n
    eq_i = later_eq[0]
    u_x, u_y, u_z = seq[:op_i], seq[op_i + 1:eq_i], seq[eq_i + 1:]
    if not all(t in ("0", "1") for t in u_x + u_y + u_z):
        return 1.0
    if not (u_x and u_y and u_z):
        return 1.0

    def decode(bits):
        x = 0
        for k, b in enumerate(bits):
            if b == "1":
                x |= (1 << k)
        return x

    x, y = decode(u_x), decode(u_y)
    op = seq[op_i]
    z_true = x + y if op == "+" else x * y
    z_bits_true = []
    zz = z_true
    if zz == 0:
        z_bits_true = ["0"]
    else:
        while zz:
            z_bits_true.append("1" if zz & 1 else "0")
            zz >>= 1
    m = min(len(u_z), len(z_bits_true))
    for j in range(m):
        if u_z[j] != z_bits_true[j]:
            return (eq_i + 1 + j) / n
    if len(u_z) != len(z_bits_true):
        return 1.0
    return 1.0


def viol_compute_sqrt(seq):
    n = len(seq)
    eq_positions = [i for i, t in enumerate(seq) if t == "="]
    if len(eq_positions) == 0:
        return 1.0
    if len(eq_positions) >= 2:
        return eq_positions[1] / n
    eq_i = eq_positions[0]
    u_x, u_z = seq[:eq_i], seq[eq_i + 1:]
    if not all(t in ("0", "1") for t in u_x + u_z):
        return 1.0
    if not (u_x and u_z):
        return 1.0

    def decode(bits):
        x = 0
        for k, b in enumerate(bits):
            if b == "1":
                x |= (1 << k)
        return x

    x = decode(u_x)
    z_true = math.isqrt(x)  # exact integer sqrt -- x can have hundreds of bits, float sqrt loses precision
    z_bits_true = []
    zz = z_true
    if zz == 0:
        z_bits_true = ["0"]
    else:
        while zz:
            z_bits_true.append("1" if zz & 1 else "0")
            zz >>= 1
    m = min(len(u_z), len(z_bits_true))
    for j in range(m):
        if u_z[j] != z_bits_true[j]:
            return (eq_i + 1 + j) / n
    if len(u_z) != len(z_bits_true):
        return 1.0
    return 1.0


TASKS = {
    "even-pairs": {
        "viol_fn": viol_even_pairs,
        "computes": "first symbol equals last symbol (5-state WFSA)",
        "structural_note": "Binary alphabet, no marker/shape distinct from the semantic rule itself; "
                           "the accept condition is a purely global property (only checkable once "
                           "both endpoints are seen), so there is no possible 'shallow structural cue' "
                           "distinct from the real computation.",
    },
    "parity": {
        "viol_fn": viol_parity,
        "computes": "odd number of 1s (2-state WFSA)",
        "structural_note": "Cumulative XOR over the whole string; global property, no local shape to violate. "
                           "Negative control: every negative is structurally 'hard' by construction.",
    },
    "repeat-01": {
        "viol_fn": viol_repeat_01,
        "computes": "exact string (01)^n for some n>=0 (strict alternation starting with 0, even length)",
        "structural_note": "Extremely rigid grammar (one valid string per even length); local "
                           "alternation-mismatch is checkable at the first out-of-place bit.",
    },
    "first": {
        "viol_fn": viol_first,
        "computes": "first symbol is 1 (rest of the string is irrelevant)",
        "structural_note": "The entire task reduces to reading ONE bit; there is no deeper computation "
                           "to bypass via a shortcut -- the trivial solution IS the correct solution.",
    },
    "dyck-2-3": {
        "viol_fn": viol_dyck,
        "computes": "balanced bracket string, 2 bracket types, max nesting depth 3",
        "structural_note": "Stack-based matching; violations (closing an empty stack, wrong closing "
                           "type, unclosed at EOS) can occur at any position depending on where edits "
                           "or randomization land.",
    },
    "majority": {
        "viol_fn": viol_majority,
        "computes": "count(1s) > count(0s)",
        "structural_note": "Global count comparison over the whole string; like parity, no local shape "
                           "to violate before the end.",
    },
    "stack-manipulation": {
        "viol_fn": viol_stack_manipulation,
        "computes": "simulate a sequence of PUSH/POP operations on an initial stack; output must equal "
                   "the reversed final stack contents",
        "structural_note": "Rich structure (PUSH/POP validity, marker placement, final-stack match); "
                           "violations can be early (invalid op sequence) or late (correct ops, wrong "
                           "reported final stack).",
    },
    "marked-reversal": {
        "viol_fn": viol_marked_reversal,
        "computes": "w # reverse(w)",
        "structural_note": "Marker + transform family: violation is early if marker count/position is "
                           "wrong, late if the suffix content itself is wrong.",
    },
    "unmarked-reversal": {
        "viol_fn": viol_unmarked_reversal,
        "computes": "w + reverse(w), no marker",
        "structural_note": "Mirror-position check; violation is only provable once the LATER member of "
                           "a mismatched (i, n-1-i) pair is reached, which is on average around the "
                           "back half of the string.",
    },
    "marked-copy": {
        "viol_fn": viol_marked_copy,
        "computes": "w # w",
        "structural_note": "Marker + transform family, transform = identity.",
    },
    "missing-duplicate-string": {
        "viol_fn": viol_missing_duplicate_string,
        "computes": "w + w with exactly one '1' replaced by a MISSING placeholder",
        "structural_note": "Marker(_)-count family; violation provable at the 2nd '_' occurrence if "
                           "over-counted, or at the mirror position of a genuine content mismatch.",
    },
    "odds-first": {
        "viol_fn": viol_odds_first,
        "computes": "w # odds_then_evens(w)",
        "structural_note": "Marker + transform family, transform = odd-indexed-then-even-indexed reorder.",
    },
    "binary-addition": {
        "viol_fn": viol_binary_arith,
        "computes": "binary x + y = z (LSB-first bit encoding)",
        "structural_note": "Operator/equals-count family: extra operators are provable immediately at "
                           "the 2nd occurrence; a correct-shape-but-wrong-arithmetic z is only provable "
                           "at the first wrong bit of z, near the end of the string.",
    },
    "binary-multiplication": {
        "viol_fn": viol_binary_arith,
        "computes": "binary x * y = z (LSB-first bit encoding)",
        "structural_note": "Same family as binary-addition.",
    },
    "compute-sqrt": {
        "viol_fn": viol_compute_sqrt,
        "computes": "x = floor(sqrt(x)) (LSB-first bit encoding, one '=' separator)",
        "structural_note": "Equals-count family: extra '=' provable immediately; wrong z bits provable "
                           "near the end.",
    },
    "bucket-sort": {
        "viol_fn": viol_bucket_sort,
        "computes": "w # sorted(w), alphabet size 5",
        "structural_note": "Marker + transform family, transform = sort.",
    },
}


def main():
    import sys, time
    print("=== A1: FLaRe task specification audit (16 tasks) ===\n", flush=True)
    audit = {}
    for task, spec in TASKS.items():
        t0 = time.time()
        print(f"  starting {task}...", flush=True)
        toks_lines, labels = load_test(task)
        viol_fn = spec["viol_fn"]
        n_neg = n_hard = n_scrambled = 0
        example_hard, example_scrambled = None, None
        for line, label in zip(toks_lines, labels):
            if label == 1:
                continue
            seq = line.split()
            n_neg += 1
            frac = viol_fn(seq)
            if frac >= HARD_THRESHOLD:
                n_hard += 1
                if example_hard is None:
                    example_hard = " ".join(seq[:30])
            else:
                n_scrambled += 1
                if example_scrambled is None:
                    example_scrambled = " ".join(seq[:30])

        frac_hard = n_hard / n_neg if n_neg else float("nan")
        if task in ("even-pairs", "parity", "majority"):
            verdict = "likely well-designed (structurally shortcut-free: global answer, no local cue possible)"
        elif task == "first":
            verdict = "N/A -- trivial task, no deeper computation to shortcut around"
        elif frac_hard >= 0.5:
            verdict = "likely well-designed (mostly hard negatives)"
        elif frac_hard <= 0.15:
            verdict = "likely shortcut-vulnerable (mostly scrambled negatives)"
        else:
            verdict = "unclear pending empirical check"

        audit[task] = {
            "computes": spec["computes"],
            "structural_note": spec["structural_note"],
            "negative_generation": "mix of perturbed-positive (few random edits) and uniform-random "
                                   "strings, rejection-sampled against is_negative() (50/50 nominal "
                                   "proposal rate, see src/recognizers/string_sampling/sample_dataset.py)",
            "n_negatives": n_neg, "n_hard": n_hard, "n_scrambled": n_scrambled,
            "frac_hard": frac_hard, "frac_scrambled": 1 - frac_hard if frac_hard == frac_hard else float("nan"),
            "preliminary_verdict": verdict,
            "example_hard_negative": example_hard, "example_scrambled_negative": example_scrambled,
        }
        print(f"{task:28s} n_neg={n_neg:>5} hard={n_hard:>5} ({frac_hard*100:5.1f}%) "
              f"scrambled={n_scrambled:>5} ({(1-frac_hard)*100:5.1f}%)  -> {verdict}  "
              f"[{time.time()-t0:.2f}s]", flush=True)

    out_path = RESULTS / "flare_a1_task_audit.json"
    out_path.write_text(json.dumps(audit, indent=2))
    print(f"\nSaved {out_path}")
    print("A1 done.")


if __name__ == "__main__":
    main()
