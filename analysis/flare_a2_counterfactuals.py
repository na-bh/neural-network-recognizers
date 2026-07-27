"""A2 counterfactual-pair construction for the 12 non-parity FLaRe tasks
under audit (parity reuses the existing analysis/parity_counterfactuals.py).

Uniform strategy per task: build a genuine POSITIVE example directly (using
each task's own transform, matching the grammar read from
src/recognizers/hand_picked_languages/*.py in A1), then construct a "corrupt"
HARD NEGATIVE by flipping exactly the LAST meaningful output symbol (the
final content/answer symbol before EOS) to a different valid alphabet
symbol. This is the maximally favorable case per the modk/cyclenav
precedent: minimal subsequent computation needed to detect the flip.

Each make_pairs_<task>(n_pairs, target_len, rng) returns a list of dicts with
"clean"/"corrupt" token-INDEX sequences (via each task's own vocab), verified
against the task's own is_positive logic (ported from A1's viol_* checkers).

PYTHONPATH=src:analysis python analysis/flare_a2_counterfactuals.py
"""

import torch


def _vocab(task):
    return torch.load(f"languages/{task}/main.vocab", weights_only=False)["tokens"]


# ---------------------------------------------------------------- repeat-01
def make_pairs_repeat01(n_pairs, target_len, rng):
    vocab = _vocab("repeat-01")
    i0, i1 = vocab.index("0"), vocab.index("1")
    n = max(1, target_len // 2)
    clean_str = ["0", "1"] * n
    clean_str = clean_str[:2 * n]
    corrupt_str = clean_str[:-1] + (["1"] if clean_str[-1] == "0" else ["0"])
    tok = {"0": i0, "1": i1}
    clean = [tok[c] for c in clean_str]
    corrupt = [tok[c] for c in corrupt_str]
    return [{"clean": clean, "corrupt": corrupt} for _ in range(n_pairs)]


# ------------------------------------------------------------ unmarked-rev
def make_pairs_unmarked_reversal(n_pairs, target_len, rng):
    vocab = _vocab("unmarked-reversal")
    i0, i1 = vocab.index("0"), vocab.index("1")
    n = max(1, target_len // 2)
    out = []
    for _ in range(n_pairs):
        w = [int(rng.integers(0, 2)) for _ in range(n)]
        s = w + list(reversed(w))
        clean = [i0 if b == 0 else i1 for b in s]
        s_corrupt = s[:]
        s_corrupt[-1] = 1 - s_corrupt[-1]
        corrupt = [i0 if b == 0 else i1 for b in s_corrupt]
        out.append({"clean": clean, "corrupt": corrupt})
    return out


# ---------------------------------------------- marker + transform family
def _marker_transform_pairs(task, transform, n_pairs, target_len, rng, alphabet_bits=2, symbol_offset=0):
    vocab = _vocab(task)
    marker_i = vocab.index("#")
    bit_toks = [vocab.index(str(b + symbol_offset)) for b in range(alphabet_bits)]
    n = max(1, (target_len - 1) // 2)
    out = []
    for _ in range(n_pairs):
        w = [int(rng.integers(0, alphabet_bits)) for _ in range(n)]
        suf = transform(w)
        clean = [bit_toks[b] for b in w] + [marker_i] + [bit_toks[b] for b in suf]
        suf_corrupt = suf[:]
        orig = suf_corrupt[-1]
        alt = [b for b in range(alphabet_bits) if b != orig]
        suf_corrupt[-1] = alt[int(rng.integers(0, len(alt)))]
        corrupt = [bit_toks[b] for b in w] + [marker_i] + [bit_toks[b] for b in suf_corrupt]
        out.append({"clean": clean, "corrupt": corrupt})
    return out


def make_pairs_marked_copy(n_pairs, target_len, rng):
    return _marker_transform_pairs("marked-copy", lambda w: list(w), n_pairs, target_len, rng)


def make_pairs_marked_reversal(n_pairs, target_len, rng):
    return _marker_transform_pairs("marked-reversal", lambda w: list(reversed(w)), n_pairs, target_len, rng)


def make_pairs_odds_first(n_pairs, target_len, rng):
    return _marker_transform_pairs("odds-first", lambda w: w[::2] + w[1::2], n_pairs, target_len, rng)


def make_pairs_bucket_sort(n_pairs, target_len, rng):
    return _marker_transform_pairs("bucket-sort", lambda w: sorted(w), n_pairs, target_len, rng,
                                   alphabet_bits=5, symbol_offset=1)


# --------------------------------------------------------- binary-* family
def _random_big_int(rng, n_bits):
    """Build a random n_bits-wide Python int via a bit array -- avoids numpy
    rng.integers()'s int64 bound, which overflows for n_bits ~> 63."""
    bits = rng.integers(0, 2, size=n_bits)
    x = 0
    for k, b in enumerate(bits):
        if b:
            x |= (1 << k)
    return x


def _binary_arith_pairs(task, op_sym, compute_z, n_pairs, target_len, rng):
    vocab = _vocab(task)
    i0, i1 = vocab.index("0"), vocab.index("1")
    op_i = vocab.index(op_sym)
    eq_i = vocab.index("=")
    n_bits = max(2, (target_len - 2) // 3)
    out = []
    tries = 0
    while len(out) < n_pairs and tries < n_pairs * 50:
        tries += 1
        x = _random_big_int(rng, n_bits)
        y = _random_big_int(rng, n_bits)
        z = compute_z(x, y)
        z_bits = []
        zz = z
        if zz == 0:
            z_bits = [0]
        else:
            while zz:
                z_bits.append(zz & 1)
                zz >>= 1
        u_x = [(x >> k) & 1 for k in range(n_bits)]
        u_y = [(y >> k) & 1 for k in range(n_bits)]
        clean_str = u_x + [op_i] + u_y + [eq_i] + z_bits
        clean = [i1 if t == 1 else (i0 if t == 0 else t) for t in clean_str]
        z_bits_corrupt = z_bits[:]
        z_bits_corrupt[-1] = 1 - z_bits_corrupt[-1]
        corrupt_str = u_x + [op_i] + u_y + [eq_i] + z_bits_corrupt
        corrupt = [i1 if t == 1 else (i0 if t == 0 else t) for t in corrupt_str]
        out.append({"clean": clean, "corrupt": corrupt})
    return out


def make_pairs_binary_addition(n_pairs, target_len, rng):
    return _binary_arith_pairs("binary-addition", "+", lambda x, y: x + y, n_pairs, target_len, rng)


def make_pairs_binary_multiplication(n_pairs, target_len, rng):
    return _binary_arith_pairs("binary-multiplication", "×", lambda x, y: x * y, n_pairs, target_len, rng)


# ---------------------------------------------------------------- compute-sqrt
def make_pairs_compute_sqrt(n_pairs, target_len, rng):
    import math
    vocab = _vocab("compute-sqrt")
    i0, i1 = vocab.index("0"), vocab.index("1")
    eq_i = vocab.index("=")
    n_bits = max(2, target_len - 2)
    out = []
    for _ in range(n_pairs):
        x = _random_big_int(rng, n_bits)
        z = math.isqrt(x)
        z_bits = []
        zz = z
        if zz == 0:
            z_bits = [0]
        else:
            while zz:
                z_bits.append(zz & 1)
                zz >>= 1
        u_x = [(x >> k) & 1 for k in range(n_bits)]
        clean_str = u_x + [eq_i] + z_bits
        clean = [i1 if t == 1 else (i0 if t == 0 else t) for t in clean_str]
        z_bits_corrupt = z_bits[:]
        z_bits_corrupt[-1] = 1 - z_bits_corrupt[-1]
        corrupt_str = u_x + [eq_i] + z_bits_corrupt
        corrupt = [i1 if t == 1 else (i0 if t == 0 else t) for t in corrupt_str]
        out.append({"clean": clean, "corrupt": corrupt})
    return out


# ---------------------------------------------------------------- dyck-2-3
def make_pairs_dyck(n_pairs, target_len, rng, k=2, m=3):
    vocab = _vocab("dyck-2-3")
    out = []
    n_pairs_target_tokens = max(2, target_len // 2)
    for _ in range(n_pairs):
        stack = []
        seq = []
        for _ in range(n_pairs_target_tokens):
            can_close = len(stack) > 0
            can_open = len(stack) < m
            if can_open and (not can_close or rng.random() < 0.6):
                t = int(rng.integers(0, k))
                stack.append(t)
                seq.append(f"({t}")
            elif can_close:
                t = stack.pop()
                seq.append(f"){t}")
            else:
                break
        while stack:
            t = stack.pop()
            seq.append(f"){t}")
        clean = [vocab.index(s) for s in seq]
        # corrupt: flip the type of the LAST closing bracket
        last_close_idx = None
        for i in range(len(seq) - 1, -1, -1):
            if seq[i].startswith(")"):
                last_close_idx = i
                break
        seq_corrupt = seq[:]
        orig_type = int(seq[last_close_idx][1:])
        alt_type = [t for t in range(k) if t != orig_type][0]
        seq_corrupt[last_close_idx] = f"){alt_type}"
        corrupt = [vocab.index(s) for s in seq_corrupt]
        out.append({"clean": clean, "corrupt": corrupt})
    return out


# ---------------------------------------------------------- stack-manipulation
def make_pairs_stack_manipulation(n_pairs, target_len, rng):
    vocab = _vocab("stack-manipulation")
    i0, i1, ipush, ipop, imark = (vocab.index("0"), vocab.index("1"),
                                  vocab.index("PUSH"), vocab.index("POP"), vocab.index("#"))
    out = []
    tries = 0
    while len(out) < n_pairs and tries < n_pairs * 50:
        tries += 1
        n_stack = int(rng.integers(0, 4))
        init_stack = [int(rng.integers(0, 2)) for _ in range(n_stack)]
        stack = init_stack.copy()
        ops = []
        n_ops_target = max(2, target_len // 3)
        for _ in range(n_ops_target):
            can_pop = len(stack) > 0
            if can_pop and rng.random() < 0.4:
                ops.append("POP")
                stack.pop()
            else:
                b = int(rng.integers(0, 2))
                ops.append("PUSH")
                ops.append(str(b))
                stack.append(b)
        if not stack:
            continue  # need a nonempty result to flip
        result = list(reversed(stack))
        toks = {"0": i0, "1": i1, "PUSH": ipush, "POP": ipop}
        clean_str = init_stack + [toks[str(b)] for b in []]  # placeholder, built below
        clean = ([i0 if b == 0 else i1 for b in init_stack] +
                 [toks[o] for o in ops] + [imark] +
                 [i0 if b == 0 else i1 for b in result])
        result_corrupt = result[:]
        result_corrupt[-1] = 1 - result_corrupt[-1]
        corrupt = ([i0 if b == 0 else i1 for b in init_stack] +
                  [toks[o] for o in ops] + [imark] +
                  [i0 if b == 0 else i1 for b in result_corrupt])
        out.append({"clean": clean, "corrupt": corrupt})
    return out


# ------------------------------------------------------ missing-duplicate
def make_pairs_missing_duplicate_string(n_pairs, target_len, rng):
    vocab = _vocab("missing-duplicate-string")
    i0, i1, imiss = vocab.index("0"), vocab.index("1"), vocab.index("_")
    n = max(2, target_len // 2)
    out = []
    tries = 0
    while len(out) < n_pairs and tries < n_pairs * 50:
        tries += 1
        w = [int(rng.integers(0, 2)) for _ in range(n)]
        if 1 not in w:
            w[int(rng.integers(0, n))] = 1
        s = w + w
        ones_idx = [i for i, a in enumerate(s) if a == 1]
        miss_pos = ones_idx[int(rng.integers(0, len(ones_idx)))]
        s_clean = s[:]
        s_clean[miss_pos] = "M"
        # flip the last position that is NOT the missing marker
        flip_pos = len(s) - 1
        if flip_pos == miss_pos:
            flip_pos -= 1
        s_corrupt = s_clean[:]
        s_corrupt[flip_pos] = 1 - s_corrupt[flip_pos]

        def enc(seq):
            return [imiss if t == "M" else (i1 if t == 1 else i0) for t in seq]

        out.append({"clean": enc(s_clean), "corrupt": enc(s_corrupt)})
    return out


def make_pairs_parity(n_pairs, target_len, rng):
    vocab = _vocab("parity")
    i0, i1 = vocab.index("0"), vocab.index("1")
    n = max(1, target_len)
    out = []
    for _ in range(n_pairs):
        w = [int(rng.integers(0, 2)) for _ in range(n)]
        if sum(w) % 2 == 0:
            w[0] = 1 - w[0]  # ensure clean is accept (odd count of 1s)
        clean = [i0 if b == 0 else i1 for b in w]
        w_corrupt = w[:]
        w_corrupt[-1] = 1 - w_corrupt[-1]
        corrupt = [i0 if b == 0 else i1 for b in w_corrupt]
        out.append({"clean": clean, "corrupt": corrupt})
    return out


TASK_MAKERS = {
    "parity": make_pairs_parity,
    "repeat-01": make_pairs_repeat01,
    "unmarked-reversal": make_pairs_unmarked_reversal,
    "marked-copy": make_pairs_marked_copy,
    "marked-reversal": make_pairs_marked_reversal,
    "odds-first": make_pairs_odds_first,
    "bucket-sort": make_pairs_bucket_sort,
    "binary-addition": make_pairs_binary_addition,
    "binary-multiplication": make_pairs_binary_multiplication,
    "compute-sqrt": make_pairs_compute_sqrt,
    "dyck-2-3": make_pairs_dyck,
    "stack-manipulation": make_pairs_stack_manipulation,
    "missing-duplicate-string": make_pairs_missing_duplicate_string,
}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    for task, fn in TASK_MAKERS.items():
        pairs = fn(5, 20, rng)
        vocab = _vocab(task)
        print(f"=== {task} ({len(pairs)} pairs) ===")
        p = pairs[0]
        print("  clean  :", " ".join(vocab[t] for t in p["clean"]))
        print("  corrupt:", " ".join(vocab[t] for t in p["corrupt"]))
        assert p["clean"] != p["corrupt"]
        diff = sum(1 for a, b in zip(p["clean"], p["corrupt"]) if a != b)
        print(f"  n_differing_positions={diff} len={len(p['clean'])}")
