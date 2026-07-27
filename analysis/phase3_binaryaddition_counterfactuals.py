"""Phase 3 counterfactual construction for binary-addition. Task-specific
throughout -- no marker-family aliasing (this task has two structural
tokens, '+' and '=', LSB-first bit encoding, no marker, no length-equality
invariant, per Phase 1 P1).

All examples use FIXED lengths (LEN_X=LEN_Y=8, LEN_Z=10) so that swap/flip
positions are comparable across pairs and across conditions, mirroring
bucket-sort's fixed n_half=50 convention. LEN_Z=10 comfortably holds any
sum of two 8-bit values (max 510, needs 9 bits) with one bit of slack.

Condition 1 (structural features -- operator position, segment length,
digit counts): built via a VALUE-PRESERVING padding trick specific to
LSB-first encoding -- appending trailing '0' bits to a segment changes its
length (hence downstream token positions) WITHOUT changing its decoded
value, so clean/corrupt pairs can hold u_x, u_y, u_z content and
correctness EXACTLY fixed while isolating a pure structural/positional
perturbation. This is a cleaner isolation than bucket-sort's marker
pairs (which used independently-generated content), possible here because
of the specific arithmetic structure of the encoding.
  operator_position_pairs : pad u_x -> shifts len_x, operator position,
                             equals position, total_length together
                             (entangled by construction: u_x's length
                             determines everything downstream of it)
  segment_length_pairs    : pad u_y -> shifts len_y, equals position,
                             total_length; leaves operator position AND
                             len_x untouched (prefix through '+' identical)
  digit_count_pairs       : NOT padding-based (padding cannot change a
                             value's Hamming weight -- proven in Phase 1
                             Phase-3-design notes: for a fixed decoded
                             value, ones-count is invariant under trailing-
                             zero padding). Instead: clean u_x = low
                             ones-count pattern (single 1-bit), corrupt
                             u_x = high ones-count pattern (single 0-bit),
                             SAME u_y, SAME lengths/positions -- both
                             genuine correct sums, only u_x's digit-count
                             and value differ.

Condition 2 (arithmetic -- carry_state_pos1-4): rejection-sampled pairs at
fixed lengths, matched on correctness (BOTH genuine positives) and
length/position, differing specifically in the TRUE carry-in bit at
position k (computed from u_x/u_y only, per phase2_targets.py's
_true_carry_and_bits).

Condition 3 (target_computation, position-swept bit-flip): per Phase 1 P2's
already-specified design -- clean = genuine positive; corrupt = SAME u_x,
u_y, u_z EXCEPT one bit of u_z flipped at position k (preserves len_x/y/z,
structural validity; does NOT preserve ones_count_z, as documented in that
design -- flagged, not assumed away). Swept across low/mid/high k.

NO PATCHING IS RUN HERE -- pair construction + audit only.

PYTHONPATH=src:analysis python analysis/phase3_binaryaddition_counterfactuals.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "analysis")
import phase2_targets as T

RESULTS = Path("analysis_outputs/final_results")
OPERATOR, EQUALS = "+", "="
LEN_X, LEN_Y, LEN_Z = 8, 8, 10


def decode_lsb_first(bits):
    x = 0
    for k, b in enumerate(bits):
        if b == "1":
            x |= (1 << k)
    return x


def encode_lsb_first(value, length):
    assert value < (1 << length), f"value {value} does not fit in {length} bits"
    return [str((value >> k) & 1) for k in range(length)]


def rand_bits(n, rng):
    return [str(int(b)) for b in rng.integers(0, 2, size=n)]


def seq_tokens(u_x, u_y, u_z):
    return u_x + [OPERATOR] + u_y + [EQUALS] + u_z


def make_valid_example(len_x, len_y, len_z, rng):
    """Random u_x,u_y of given lengths; correct u_z zero-padded to exactly
    len_z bits. Caller must ensure len_z is large enough for any sum of an
    len_x-bit and a len_y-bit value (guaranteed for LEN_X=LEN_Y=8, LEN_Z=10)."""
    u_x = rand_bits(len_x, rng)
    u_y = rand_bits(len_y, rng)
    s = decode_lsb_first(u_x) + decode_lsb_first(u_y)
    u_z = encode_lsb_first(s, len_z)
    return u_x, u_y, u_z


def pad_zeros(bits, k):
    """Appends k trailing '0' bits (higher-order, LSB-first) -- value-preserving."""
    return bits + ["0"] * k


# ---------------------------------------------------------------------------
# condition 1: operator_position, segment_length, digit_count
# ---------------------------------------------------------------------------

def operator_position_pairs(n_pairs, rng, pad_k=8):
    """clean = u_x + '+' + u_y + '=' + u_z. corrupt = SAME u_x/u_y/u_z VALUES
    and tokens for u_y/u_z, but u_x zero-padded by pad_k trailing bits --
    shifts len_x, operator position (absolute+relative), equals position,
    total_length, while leaving u_x's decoded VALUE, u_y, u_z, and
    correctness exactly unchanged."""
    pairs = []
    for _ in range(n_pairs):
        u_x, u_y, u_z = make_valid_example(LEN_X, LEN_Y, LEN_Z, rng)
        clean = seq_tokens(u_x, u_y, u_z)
        corrupt = seq_tokens(pad_zeros(u_x, pad_k), u_y, u_z)
        pairs.append({
            "clean": clean, "corrupt": corrupt,
            "clean_len_x": len(u_x), "corrupt_len_x": len(u_x) + pad_k,
            "clean_operator_pos": len(u_x), "corrupt_operator_pos": len(u_x) + pad_k,
        })
    return pairs


def segment_length_pairs(n_pairs, rng, pad_k=8):
    """clean = u_x + '+' + u_y + '=' + u_z. corrupt = u_y zero-padded by
    pad_k trailing bits -- shifts len_y, equals position, total_length,
    while leaving len_x, operator position, u_x, u_y's decoded VALUE, u_z,
    and correctness exactly unchanged."""
    pairs = []
    for _ in range(n_pairs):
        u_x, u_y, u_z = make_valid_example(LEN_X, LEN_Y, LEN_Z, rng)
        clean = seq_tokens(u_x, u_y, u_z)
        corrupt = seq_tokens(u_x, pad_zeros(u_y, pad_k), u_z)
        pairs.append({
            "clean": clean, "corrupt": corrupt,
            "clean_len_y": len(u_y), "corrupt_len_y": len(u_y) + pad_k,
            "clean_equals_pos": len(u_x) + 1 + len(u_y),
            "corrupt_equals_pos": len(u_x) + 1 + len(u_y) + pad_k,
        })
    return pairs


def digit_count_pairs(n_pairs, rng):
    """clean u_x = single random 1-bit among LEN_X positions (low ones-
    count). corrupt u_x = single random 0-bit among LEN_X positions (high
    ones-count). SAME u_y (held identical), SAME lengths/positions/total_
    length; u_x's VALUE (hence u_z) differs, but correctness=True for both
    (both are genuine correct sums at their own value)."""
    pairs = []
    for _ in range(n_pairs):
        u_y = rand_bits(LEN_Y, rng)
        one_idx = int(rng.integers(0, LEN_X))
        zero_idx = int(rng.integers(0, LEN_X))
        u_x_low = ["0"] * LEN_X
        u_x_low[one_idx] = "1"
        u_x_high = ["1"] * LEN_X
        u_x_high[zero_idx] = "0"

        s_low = decode_lsb_first(u_x_low) + decode_lsb_first(u_y)
        s_high = decode_lsb_first(u_x_high) + decode_lsb_first(u_y)
        u_z_low = encode_lsb_first(s_low, LEN_Z)
        u_z_high = encode_lsb_first(s_high, LEN_Z)

        clean = seq_tokens(u_x_low, u_y, u_z_low)
        corrupt = seq_tokens(u_x_high, u_y, u_z_high)
        pairs.append({
            "clean": clean, "corrupt": corrupt,
            "clean_ones_x": sum(1 for t in u_x_low if t == "1"),
            "corrupt_ones_x": sum(1 for t in u_x_high if t == "1"),
        })
    return pairs


def audit_padding_pairs(pairs, len_key_clean, len_key_corrupt, expect_len_delta):
    audit = {"n_pairs": len(pairs)}
    audit["same_total_length_would_be_wrong"] = None  # padding INTENTIONALLY changes total length
    audit["length_delta_correct"] = all(
        p[len_key_corrupt] - p[len_key_clean] == expect_len_delta for p in pairs)
    # correctness must be preserved (padding is value-preserving)
    clean_correct = [T.binaddition_target_computation(p["clean"]) for p in pairs]
    corrupt_correct = [T.binaddition_target_computation(p["corrupt"]) for p in pairs]
    audit["clean_all_correct"] = all(c == 1 for c in clean_correct)
    audit["corrupt_all_correct"] = all(c == 1 for c in corrupt_correct)
    audit["correctness_preserved"] = audit["clean_all_correct"] and audit["corrupt_all_correct"]
    audit["target_property_isolated"] = audit["length_delta_correct"] and audit["correctness_preserved"]
    return audit


def audit_digit_count_pairs(pairs):
    audit = {"n_pairs": len(pairs)}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    audit["ones_x_clean_low"] = all(p["clean_ones_x"] == 1 for p in pairs)
    audit["ones_x_corrupt_high"] = all(p["corrupt_ones_x"] == LEN_X - 1 for p in pairs)
    clean_correct = [T.binaddition_target_computation(p["clean"]) for p in pairs]
    corrupt_correct = [T.binaddition_target_computation(p["corrupt"]) for p in pairs]
    audit["clean_all_correct"] = all(c == 1 for c in clean_correct)
    audit["corrupt_all_correct"] = all(c == 1 for c in corrupt_correct)
    audit["correctness_preserved"] = audit["clean_all_correct"] and audit["corrupt_all_correct"]
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["ones_x_clean_low"] and audit["ones_x_corrupt_high"]
        and audit["correctness_preserved"]
    )
    return audit


# ---------------------------------------------------------------------------
# condition 2: carry_state_pos1-4
# ---------------------------------------------------------------------------

def carry_state_pairs(k, n_pairs, rng, max_attempts_mult=400):
    """Rejection-sampled: both clean and corrupt are genuine correct sums at
    fixed LEN_X/LEN_Y/LEN_Z (same length/positions/total_length), differing
    in the TRUE carry-in bit at position k (computed from u_x/u_y only).
    clean always has carry_pos_k=0, corrupt always has carry_pos_k=1."""
    zero_group, one_group = [], []
    attempts = 0
    target_n = n_pairs * 3  # oversample each group a bit for pairing variety
    while (len(zero_group) < target_n or len(one_group) < target_n) and attempts < n_pairs * max_attempts_mult:
        attempts += 1
        u_x, u_y, u_z = make_valid_example(LEN_X, LEN_Y, LEN_Z, rng)
        seq = seq_tokens(u_x, u_y, u_z)
        carry = T.binaddition_carry_state_at_position(seq, k)
        if carry is None:
            continue
        (zero_group if carry == 0 else one_group).append(seq)
    if len(zero_group) < n_pairs or len(one_group) < n_pairs:
        raise RuntimeError(f"pos{k}: only got {len(zero_group)} carry=0 / {len(one_group)} carry=1 "
                           f"examples after {attempts} attempts")
    pairs = []
    for i in range(n_pairs):
        clean = zero_group[i]
        corrupt = one_group[i]
        pairs.append({
            "clean": clean, "corrupt": corrupt,
            "clean_carry": T.binaddition_carry_state_at_position(clean, k),
            "corrupt_carry": T.binaddition_carry_state_at_position(corrupt, k),
        })
    return pairs


def audit_carry_state_pairs(pairs, k):
    audit = {"n_pairs": len(pairs), "k": k}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    audit["clean_all_carry0"] = all(p["clean_carry"] == 0 for p in pairs)
    audit["corrupt_all_carry1"] = all(p["corrupt_carry"] == 1 for p in pairs)
    clean_correct = [T.binaddition_target_computation(p["clean"]) for p in pairs]
    corrupt_correct = [T.binaddition_target_computation(p["corrupt"]) for p in pairs]
    audit["clean_all_correct"] = all(c == 1 for c in clean_correct)
    audit["corrupt_all_correct"] = all(c == 1 for c in corrupt_correct)
    audit["correctness_matched_both_positive"] = audit["clean_all_correct"] and audit["corrupt_all_correct"]
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["clean_all_carry0"] and audit["corrupt_all_carry1"]
        and audit["correctness_matched_both_positive"]
    )
    return audit


# ---------------------------------------------------------------------------
# condition 3: target_computation, position-swept single-bit-flip in u_z
# ---------------------------------------------------------------------------

def target_computation_bitflip_pairs(flip_k, n_pairs, rng):
    """clean = genuine correct sum. corrupt = SAME u_x, u_y, SAME u_z except
    bit flip_k flipped -- guarantees an incorrect sum (decode changes by
    exactly +-2^flip_k), preserves len_x/y/z and structural validity exactly,
    does NOT preserve ones_count_z (documented, expected)."""
    pairs = []
    for _ in range(n_pairs):
        u_x, u_y, u_z = make_valid_example(LEN_X, LEN_Y, LEN_Z, rng)
        clean = seq_tokens(u_x, u_y, u_z)
        u_z_flipped = u_z[:]
        u_z_flipped[flip_k] = "1" if u_z[flip_k] == "0" else "0"
        corrupt = seq_tokens(u_x, u_y, u_z_flipped)
        pairs.append({"clean": clean, "corrupt": corrupt, "flip_k": flip_k})
    return pairs


def audit_target_computation_bitflip_pairs(pairs, flip_k):
    audit = {"n_pairs": len(pairs), "flip_k": flip_k}
    audit["same_length"] = all(len(p["clean"]) == len(p["corrupt"]) for p in pairs)
    diffs = [sum(1 for a, b in zip(p["clean"], p["corrupt"]) if a != b) for p in pairs]
    audit["differs_at_exactly_one_position"] = all(d == 1 for d in diffs)
    audit["len_x_preserved"] = all(
        T.binaddition_len_x(p["clean"]) == T.binaddition_len_x(p["corrupt"]) for p in pairs)
    audit["len_y_preserved"] = all(
        T.binaddition_len_y(p["clean"]) == T.binaddition_len_y(p["corrupt"]) for p in pairs)
    audit["len_z_preserved"] = all(
        T.binaddition_len_z(p["clean"]) == T.binaddition_len_z(p["corrupt"]) for p in pairs)
    clean_correct = [T.binaddition_target_computation(p["clean"]) for p in pairs]
    corrupt_correct = [T.binaddition_target_computation(p["corrupt"]) for p in pairs]
    audit["clean_all_correct"] = all(c == 1 for c in clean_correct)
    audit["corrupt_all_incorrect"] = all(c == 0 for c in corrupt_correct)
    audit["ones_count_z_NOT_preserved_expected"] = not all(
        T.binaddition_ones_fraction_z(p["clean"]) == T.binaddition_ones_fraction_z(p["corrupt"]) for p in pairs)
    audit["target_property_isolated"] = (
        audit["same_length"] and audit["differs_at_exactly_one_position"] and
        audit["len_x_preserved"] and audit["len_y_preserved"] and audit["len_z_preserved"] and
        audit["clean_all_correct"] and audit["corrupt_all_incorrect"]
    )
    return audit


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    op_pairs = operator_position_pairs(25, rng)
    op_audit = audit_padding_pairs(op_pairs, "clean_len_x", "corrupt_len_x", 8)
    print("=== operator_position_pairs audit ===")
    print(json.dumps(op_audit, indent=2))

    sl_pairs = segment_length_pairs(25, rng)
    sl_audit = audit_padding_pairs(sl_pairs, "clean_len_y", "corrupt_len_y", 8)
    print("=== segment_length_pairs audit ===")
    print(json.dumps(sl_audit, indent=2))

    dc_pairs = digit_count_pairs(25, rng)
    dc_audit = audit_digit_count_pairs(dc_pairs)
    print("=== digit_count_pairs audit ===")
    print(json.dumps(dc_audit, indent=2))

    for k in [1, 2, 3, 4]:
        cs_pairs = carry_state_pairs(k, 25, rng)
        cs_audit = audit_carry_state_pairs(cs_pairs, k)
        print(f"=== carry_state_pos{k}_pairs audit ===")
        print(json.dumps(cs_audit, indent=2))

    for label, k in [("low", 1), ("mid", 5), ("high", 8)]:
        tc_pairs = target_computation_bitflip_pairs(k, 25, rng)
        tc_audit = audit_target_computation_bitflip_pairs(tc_pairs, k)
        print(f"=== target_computation_bitflip_pairs [{label}, k={k}] audit ===")
        print(json.dumps(tc_audit, indent=2))
