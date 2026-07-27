"""Contrastive-pairing experiment, CPF1: dataset construction + audit.

Builds a PAIRED training set for marked-reversal from the EXISTING corrected
dataset (languages/marked-reversal-fixed/) -- no new positives or hard
negatives are sampled from the language; every string used here already
exists in that dataset's train split.

KEY TECHNICAL FINDING (worth flagging explicitly): F1's hard-negative
construction (fix_dataset_marked_reversal_f1.py) records the SWAP INDICES
(swap_i, swap_j) used to corrupt each hard negative, but never serializes
the pre-swap source positive string itself. However, a swap is its own
inverse -- applying the identical swap a second time exactly undoes it. So
the "constructed-from" positive for every hard negative IS exactly
recoverable (not approximated, not nearest-neighbor-matched) by re-applying
swap_i/swap_j to the hard negative's post-marker half. This is verified
per-example below (is_positive_local on every reconstruction), not assumed.

RATIO-PRESERVATION DESIGN DECISION: pairing every hard negative with a
freshly-reconstructed positive necessarily ADDS one new positive row per
hard negative. To hold the requested 50% positive / 25% uniform-random /
25% hard-negative ratio (rather than let it drift towards ~60/20/20), the
existing hard-negative and uniform-random rows are kept in full (unchanged,
25%ish each, matching F1's own realized split), and the 50% positive
population is reconstituted as: EXACTLY one reconstructed (paired) positive
per surviving hard negative (~25%), plus a SEEDED DOWNSAMPLE (without
replacement) of F1's original positive population to the SAME count
(~25%), serving as ordinary unpaired positives. This means roughly half of
F1's original positive rows are NOT reused here (documented explicitly
below, not silently dropped) -- their omission is what keeps the ratio at
50/25/25 instead of drifting upward as pairing is introduced.

FILE-ORDER VS ACTUAL BATCH CO-OCCURRENCE (also worth flagging explicitly):
this script places each pair as two CONSECUTIVE rows in the output file
with a shared pair_id, matching the letter of the request and giving a
human-auditable, inspectable pairing. However, rau's TrainingLoop.run()
(the shared FLaRe/rau training loop) calls
`random_shuffling_generator.shuffle(training_data)` on the FULL example
list at the START OF EVERY EPOCH, before batches are constructed --
so on-disk adjacency does NOT, by itself, guarantee batch co-occurrence at
training time (a full-list shuffle destroys pairwise adjacency). The
FILE-LEVEL pairing built here is necessary for CPF2 (which loads the
pair-id.txt side-channel and match rows) but is NOT sufficient on its own;
CPF2 (analysis/contrastive_training_loop.py) enforces actual same-batch
placement directly in a custom `generate_batches` override, independent of
file order.

Outputs (new directory, standard test/validation-short splits COPIED
unchanged from languages/marked-reversal-fixed/, only the train split
differs):
  languages/marked-reversal-fixed-contrastive/{main.tok,labels.txt,
    negative-kind.txt,next-symbols.jsonl,log-probabilities.txt,
    hard-negative-swap-meta.jsonl,pair-id.txt}
  languages/marked-reversal-fixed-contrastive/datasets/validation-short/{...}  (copied, unchanged)
  languages/marked-reversal-fixed-contrastive/datasets/test/{...}              (copied, unchanged)

PYTHONPATH=src:analysis python analysis/fix_dataset_marked_reversal_contrastive.py
"""

import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")
from recognizers.hand_picked_languages.marked_reversal import MarkedReversal
from recognizers.tools.jsonl import write_json_line
from recognizers.automata.reserved import ReservedSymbol

RESULTS = Path("analysis_outputs/final_results")
SRC_LANG_DIR = Path("languages/marked-reversal-fixed")
OUT_LANG_DIR = Path("languages/marked-reversal-fixed-contrastive")
RANDOM_SEED = 20260724
MARKER = "#"


def load_split(lang_dir):
    toks = (lang_dir / "main.tok").read_text().splitlines()
    labels = [int(x) for x in (lang_dir / "labels.txt").read_text().splitlines()]
    kinds = (lang_dir / "negative-kind.txt").read_text().splitlines()
    metas = [json.loads(line) for line in (lang_dir / "hard-negative-swap-meta.jsonl").read_text().splitlines()]
    seqs = [line.split() for line in toks]
    assert len(seqs) == len(labels) == len(kinds) == len(metas)
    return seqs, labels, kinds, metas


def is_positive_local(seq):
    if MARKER not in seq:
        return False
    m = seq.index(MARKER)
    before, after = seq[:m], seq[m + 1:]
    if len(before) != len(after):
        return False
    if seq.count(MARKER) != 1:
        return False
    return after == list(reversed(before))


def reconstruct_source_positive(seq, meta):
    """Inverts F1's swap construction exactly: re-applying the SAME
    swap_i/swap_j to the post-marker half undoes the original corruption
    (a swap is its own inverse). Returns the exact pre-swap positive
    tokens."""
    m = seq.index(MARKER)
    before, after = seq[:m], seq[m + 1:]
    i, j = meta["swap_i"], meta["swap_j"]
    after_orig = list(after)
    after_orig[i], after_orig[j] = after_orig[j], after_orig[i]
    return before + [MARKER] + after_orig


def compute_next_symbols_for_w(w):
    """Matches MarkedReversal._w_to_next_symbols exactly: len(w)+1
    unconstrained positions (covering w itself and the marker), then the
    reversed(w) symbols as deterministic single-symbol continuations, then
    EOS."""
    result = []
    r = ["0", "1", MARKER]
    for _ in range(len(w) + 1):
        result.append(list(r))
    for a in reversed(w):
        result.append([a])
    result.append([])  # EOS-only marker, handled specially below
    return result


def build_next_symbols_row(next_symbols_syms):
    row = []
    for i, syms in enumerate(next_symbols_syms):
        is_eos_row = (syms == [])
        row.append({"s": " ".join(syms), "e": is_eos_row})
    return row


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_LANG_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    base_lang = MarkedReversal()
    train_lang = base_lang.with_length_range((0, 40))

    seqs, labels, kinds, metas = load_split(SRC_LANG_DIR)
    n_total = len(seqs)

    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    hard_idx = [i for i, (l, k) in enumerate(zip(labels, kinds)) if l == 0 and k == "hard_negative"]
    uniform_idx = [i for i, (l, k) in enumerate(zip(labels, kinds)) if l == 0 and k == "uniform_random"]
    print(f"F1 train composition: n_total={n_total} n_positive={len(pos_idx)} "
          f"n_hard_negative={len(hard_idx)} n_uniform_random={len(uniform_idx)}", flush=True)

    # ------------------------------------------------------------------
    # reconstruct the exact pre-swap source positive for every hard negative
    # ------------------------------------------------------------------
    reconstruction_failures = 0
    reconstructed = []  # list of (hard_neg_idx, reconstructed_positive_tokens)
    for i in hard_idx:
        recon = reconstruct_source_positive(seqs[i], metas[i])
        if not is_positive_local(recon):
            reconstruction_failures += 1
            continue
        reconstructed.append((i, recon))
    print(f"reconstructed {len(reconstructed)}/{len(hard_idx)} source positives "
          f"({reconstruction_failures} failures)", flush=True)
    assert reconstruction_failures == 0, (
        f"{reconstruction_failures} hard negatives failed to reconstruct to a genuine "
        f"positive via swap inversion -- this should be impossible since a swap is its own "
        f"inverse; investigate before proceeding"
    )

    # ------------------------------------------------------------------
    # downsample F1's original positive population to match hard_idx count,
    # to hold the ratio at ~50/25/25 (see module docstring)
    # ------------------------------------------------------------------
    pos_pool = pos_idx[:]
    rng.shuffle(pos_pool)
    n_unpaired_positive_target = len(reconstructed)
    unpaired_pos_idx = pos_pool[:n_unpaired_positive_target]
    dropped_pos_idx = pos_pool[n_unpaired_positive_target:]
    print(f"keeping {len(unpaired_pos_idx)}/{len(pos_idx)} original positives as unpaired "
          f"(dropping {len(dropped_pos_idx)} to hold the 50/25/25 ratio)", flush=True)

    # ------------------------------------------------------------------
    # assemble output rows: pair units (2 rows, shared pair_id) +
    # unpaired-positive units (1 row) + uniform-random units (1 row)
    # ------------------------------------------------------------------
    units = []  # each unit is a list of row-dicts
    pair_id_counter = 0
    for hard_i, recon_tokens in reconstructed:
        pair_id = pair_id_counter
        pair_id_counter += 1
        w = recon_tokens[:recon_tokens.index(MARKER)]
        ns_syms = compute_next_symbols_for_w(w)
        n = len(w)
        log_prob = train_lang.n_log_prob - n * train_lang.parent.log_num_symbols
        pos_row = {
            "s": recon_tokens, "label": 1, "kind": "", "pair_id": pair_id,
            "log_prob": log_prob, "next_symbols_syms": ns_syms, "meta": {},
        }
        hard_row = {
            "s": seqs[hard_i], "label": 0, "kind": "hard_negative", "pair_id": pair_id,
            "log_prob": None, "next_symbols_syms": None, "meta": metas[hard_i],
        }
        units.append([pos_row, hard_row])

    for i in unpaired_pos_idx:
        w = seqs[i][:seqs[i].index(MARKER)]
        ns_syms = compute_next_symbols_for_w(w)
        n = len(w)
        log_prob = train_lang.n_log_prob - n * train_lang.parent.log_num_symbols
        units.append([{
            "s": seqs[i], "label": 1, "kind": "", "pair_id": None,
            "log_prob": log_prob, "next_symbols_syms": ns_syms, "meta": {},
        }])

    for i in uniform_idx:
        units.append([{
            "s": seqs[i], "label": 0, "kind": "uniform_random", "pair_id": None,
            "log_prob": None, "next_symbols_syms": None, "meta": {},
        }])

    rng.shuffle(units)  # shuffle at the UNIT level -- pair members stay adjacent
    rows = [row for unit in units for row in unit]

    print(f"assembled {len(rows)} output rows "
          f"({len(reconstructed)} paired positives + {len(reconstructed)} paired hard "
          f"negatives + {len(unpaired_pos_idx)} unpaired positives + {len(uniform_idx)} "
          f"unpaired uniform-random)", flush=True)

    # ------------------------------------------------------------------
    # write output files
    # ------------------------------------------------------------------
    alphabet_size = base_lang.alphabet_size()
    with (OUT_LANG_DIR / "main.tok").open("w") as tok_f, \
         (OUT_LANG_DIR / "labels.txt").open("w") as labels_f, \
         (OUT_LANG_DIR / "negative-kind.txt").open("w") as kind_f, \
         (OUT_LANG_DIR / "log-probabilities.txt").open("w") as logprob_f, \
         (OUT_LANG_DIR / "next-symbols.jsonl").open("w") as ns_f, \
         (OUT_LANG_DIR / "hard-negative-swap-meta.jsonl").open("w") as swapmeta_f, \
         (OUT_LANG_DIR / "pair-id.txt").open("w") as pairid_f:
        for r in rows:
            print(" ".join(r["s"]), file=tok_f)
            print(r["label"], file=labels_f)
            print(r["kind"], file=kind_f)
            print(r["pair_id"] if r["pair_id"] is not None else "", file=pairid_f)
            if r["label"]:
                print(r["log_prob"], file=logprob_f)
                write_json_line(build_next_symbols_row(r["next_symbols_syms"]), ns_f)
            write_json_line(r["meta"], swapmeta_f)

    # ------------------------------------------------------------------
    # copy validation-short and test splits UNCHANGED
    # ------------------------------------------------------------------
    for split in ["validation-short", "test"]:
        src = SRC_LANG_DIR / "datasets" / split
        dst = OUT_LANG_DIR / "datasets" / split
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    print("copied validation-short and test splits unchanged", flush=True)

    # ------------------------------------------------------------------
    # verification (a)/(b)/(c) per the user's spec
    # ------------------------------------------------------------------
    out_labels = [r["label"] for r in rows]
    out_kinds = [r["kind"] for r in rows]
    out_pair_ids = [r["pair_id"] for r in rows]

    n_out_total = len(rows)
    n_out_pos = sum(out_labels)
    n_out_hard = sum(1 for k in out_kinds if k == "hard_negative")
    n_out_uniform = sum(1 for k in out_kinds if k == "uniform_random")

    # (a) every hard negative has a paired positive in the training set
    pair_id_to_rows = {}
    for idx, pid in enumerate(out_pair_ids):
        if pid is not None:
            pair_id_to_rows.setdefault(pid, []).append(idx)
    all_pairs_have_exactly_one_pos_one_hard = all(
        len(idxs) == 2 and
        {out_labels[idxs[0]], out_labels[idxs[1]]} == {0, 1}
        for idxs in pair_id_to_rows.values()
    )
    n_hard_with_pair = sum(1 for i, k in enumerate(out_kinds) if k == "hard_negative" and out_pair_ids[i] is not None)
    check_a_every_hard_negative_paired = (n_hard_with_pair == n_out_hard) and all_pairs_have_exactly_one_pos_one_hard

    # (b) pair_id metadata preserved: consecutive placement + round-trip readable
    check_b_pairs_consecutive_in_file = all(
        len(idxs) == 2 and abs(idxs[0] - idxs[1]) == 1
        for idxs in pair_id_to_rows.values()
    )
    pairid_file_lines = (OUT_LANG_DIR / "pair-id.txt").read_text().splitlines()
    check_b_pairid_file_matches_rows = (
        len(pairid_file_lines) == n_out_total and
        [int(x) if x else None for x in pairid_file_lines] == out_pair_ids
    )

    # (c) test set unchanged
    def dir_files_identical(a, b):
        a_files = sorted(p.name for p in a.iterdir() if p.is_file())
        b_files = sorted(p.name for p in b.iterdir() if p.is_file())
        if a_files != b_files:
            return False
        return all((a / f).read_bytes() == (b / f).read_bytes() for f in a_files)

    check_c_test_unchanged = dir_files_identical(
        SRC_LANG_DIR / "datasets" / "test", OUT_LANG_DIR / "datasets" / "test")
    check_c_validation_unchanged = dir_files_identical(
        SRC_LANG_DIR / "datasets" / "validation-short", OUT_LANG_DIR / "datasets" / "validation-short")

    verification = {
        "a_every_hard_negative_has_paired_positive": check_a_every_hard_negative_paired,
        "b_pairs_consecutive_in_file": check_b_pairs_consecutive_in_file,
        "b_pairid_file_matches_row_metadata": check_b_pairid_file_matches_rows,
        "c_test_set_unchanged": check_c_test_unchanged,
        "c_validation_short_unchanged": check_c_validation_unchanged,
    }
    print("\n=== verification ===")
    print(json.dumps(verification, indent=2))
    assert all(verification.values()), f"verification failed: {verification}"

    composition = {
        "n_total": n_out_total, "n_positive": n_out_pos,
        "n_positive_paired": len(reconstructed), "n_positive_unpaired": len(unpaired_pos_idx),
        "n_hard_negative": n_out_hard, "n_uniform_random": n_out_uniform,
        "positive_fraction": n_out_pos / n_out_total,
        "hard_negative_fraction": n_out_hard / n_out_total,
        "uniform_random_fraction": n_out_uniform / n_out_total,
        "n_original_positives_dropped_to_hold_ratio": len(dropped_pos_idx),
    }
    print("\n=== output composition ===")
    print(json.dumps(composition, indent=2))

    out = {
        "task": "marked-reversal",
        "experiment": "contrastive-pairing (CPF1: dataset construction + audit)",
        "description": (
            "Paired training file built ENTIRELY from languages/marked-reversal-fixed's "
            "existing train split -- no new positives or hard negatives sampled from the "
            "language. Every hard negative's exact pre-swap source positive is recovered by "
            "re-applying its recorded swap_i/swap_j (a swap is its own inverse), not "
            "approximated via nearest-neighbor matching. The 50/25/25 positive/uniform/hard "
            "ratio is held by downsampling F1's original positive population (without "
            "replacement) to match the surviving hard-negative count, since pairing "
            "necessarily adds one new positive row per hard negative."
        ),
        "technical_notes": {
            "reconstruction_method": "swap inversion (exact, not nearest-neighbor), verified "
                                      "against is_positive_local for every reconstructed row",
            "file_order_vs_batch_cooccurrence": (
                "Pairs are placed as consecutive rows here for auditability, but rau's shared "
                "TrainingLoop.run() reshuffles the FULL training example list every epoch before "
                "batching, so on-disk adjacency alone does NOT guarantee same-batch placement at "
                "training time. CPF2's custom generate_batches override enforces actual "
                "co-occurrence directly, independent of file order -- see "
                "analysis/contrastive_training_loop.py."
            ),
        },
        "random_seed": RANDOM_SEED,
        "source_dataset": str(SRC_LANG_DIR),
        "language_output_directory": str(OUT_LANG_DIR),
        "source_composition": {
            "n_total": n_total, "n_positive": len(pos_idx),
            "n_hard_negative": len(hard_idx), "n_uniform_random": len(uniform_idx),
        },
        "reconstruction": {
            "n_hard_negatives_attempted": len(hard_idx),
            "n_reconstructed_successfully": len(reconstructed),
            "n_reconstruction_failures": reconstruction_failures,
        },
        "output_composition": composition,
        "verification": verification,
    }
    out_path = RESULTS / "fix_dataset_marked_reversal_contrastive.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
