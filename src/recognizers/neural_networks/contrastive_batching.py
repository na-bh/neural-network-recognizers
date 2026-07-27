import logging
from pathlib import Path
from typing import Optional

from .batching import group_into_batches

logger = logging.getLogger('main')


def load_pair_ids(pair_id_file: Path, n_expected: int) -> list[Optional[int]]:
    """pair-id.txt has one line per training example, in the SAME ORDER as
    main.tok/main.prepared (prepare_data.py preserves line order exactly,
    and load_prepared_data_from_directory returns a list in that same
    order) -- an empty line means "not paired", an integer means the
    example belongs to that pair_id."""
    lines = pair_id_file.read_text().splitlines()
    if len(lines) != n_expected:
        raise ValueError(
            f'pair-id.txt has {len(lines)} lines but training data has '
            f'{n_expected} examples -- these must be built from the SAME '
            f'main.tok, in the same order'
        )
    return [int(x) if x else None for x in lines]


def build_pair_partner_map(training_data: list, pair_ids: list[Optional[int]]) -> dict[int, int]:
    """Builds id(example) -> id(partner_example) for every paired example.
    Relies on Python list.shuffle() permuting an existing list's ORDER
    in-place without recreating the element objects -- so id() of each
    Example tuple stays a stable identifier across every later shuffle,
    even though this map is built once, right after loading, before any
    shuffling has occurred."""
    if len(training_data) != len(pair_ids):
        raise ValueError('training_data and pair_ids must be the same length')
    by_pair_id: dict[int, list[int]] = {}
    for example, pid in zip(training_data, pair_ids):
        if pid is not None:
            by_pair_id.setdefault(pid, []).append(id(example))
    partner_map: dict[int, int] = {}
    for pid, ids in by_pair_id.items():
        if len(ids) != 2:
            raise ValueError(
                f'pair_id {pid} has {len(ids)} members, expected exactly 2'
            )
        a, b = ids
        partner_map[a] = b
        partner_map[b] = a
    logger.info(f'contrastive batching: loaded {len(by_pair_id)} pairs '
                f'({len(partner_map)} paired examples out of {len(training_data)} total)')
    return partner_map


def _remove_by_identity(lst: list, item) -> None:
    """list.remove(x) uses == to find a match, which is ambiguous for
    Example tuples (they contain torch.Tensor, and tuple.__eq__ does an
    elementwise tensor comparison that raises 'Boolean value of Tensor
    with more than one value is ambiguous' the moment it hits another
    same-shape tensor). Batch membership must be tracked by object
    IDENTITY instead."""
    for idx, x in enumerate(lst):
        if x is item:
            del lst[idx]
            return
    raise ValueError('item not found in list by identity')


def group_into_batches_pair_aware(examples, is_small_enough, pair_partner_map: dict[int, int]):
    """Same length-sorted, token-budget-greedy grouping as group_into_batches
    (reused unchanged, so the SAME batches would be produced for an
    unpaired dataset or for the validation set, where pair_partner_map has
    no matching ids), followed by a post-processing merge pass: for every
    pair whose two members landed in different batches, move the second
    member into the first member's batch. Since paired members are
    GUARANTEED to have identical sequence length (a swap can't change
    length), this merge can only ever add exactly one already-fitting-length
    item to a batch that already contains at least one item of that same
    length -- the token-budget overflow this can cause is bounded and
    small, not unbounded; not enforced strictly (documented simplification,
    logged below), matching this project's established practice of
    documenting minor constructive deviations rather than leaving them
    silent.
    """
    batches = [list(b) for b in group_into_batches(examples, is_small_enough)]
    if not pair_partner_map:
        return batches
    owner_batch: dict[int, int] = {}
    for bi, batch in enumerate(batches):
        for ex in batch:
            owner_batch[id(ex)] = bi
    id_to_example: dict[int, object] = {}
    for batch in batches:
        for ex in batch:
            id_to_example[id(ex)] = ex

    seen_pairs = set()
    n_pairs_registered = 0
    n_pairs_present = 0
    n_merged = 0
    n_overflowed_budget = 0
    for a_id, b_id in pair_partner_map.items():
        pair_key = frozenset((a_id, b_id))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        n_pairs_registered += 1
        if a_id not in owner_batch or b_id not in owner_batch:
            # one (or both) members are not present in THIS example list --
            # e.g. this call is generating VALIDATION batches, which carry
            # no pairing at all, so every registered training pair is
            # legitimately absent here. Nothing to do, and NOT a violation.
            continue
        n_pairs_present += 1
        bi_a, bi_b = owner_batch[a_id], owner_batch[b_id]
        if bi_a == bi_b:
            continue
        n_merged += 1
        example_b = id_to_example[b_id]
        target_batch = batches[bi_a]
        source_batch = batches[bi_b]
        _remove_by_identity(source_batch, example_b)
        target_batch.append(example_b)
        owner_batch[b_id] = bi_a
        max_len_target = max(len(x[0]) for x in target_batch)
        if not is_small_enough(len(target_batch), max_len_target):
            n_overflowed_budget += 1

    batches = [b for b in batches if b]

    # Hard post-condition check (not just logged stats): every pair whose
    # both members are present in this example list must now share a
    # batch. Runs every epoch -- cheap (O(n)) relative to a training epoch,
    # and this is the one invariant the whole experiment depends on.
    if pair_partner_map:
        final_owner: dict[int, int] = {}
        for bi, batch in enumerate(batches):
            for ex in batch:
                final_owner[id(ex)] = bi
        violations = set()
        for a_id, b_id in pair_partner_map.items():
            if a_id not in final_owner or b_id not in final_owner:
                continue
            if final_owner[a_id] != final_owner[b_id]:
                violations.add(frozenset((a_id, b_id)))
        if violations:
            raise RuntimeError(
                f'contrastive batching invariant violated: {len(violations)} pairs '
                f'(of {n_pairs_present} present in this batch set) do NOT share a batch '
                f'after the merge pass -- this should be impossible; investigate before '
                f'trusting any results from this run'
            )

    if n_pairs_present:
        logger.info(
            f'contrastive batching: {n_pairs_present}/{n_pairs_registered} registered pairs '
            f'present in this batch set, {n_merged} required a cross-batch merge to '
            f'co-locate, {n_overflowed_budget} merges pushed their target batch over '
            f'max_tokens_per_batch (not strictly re-split, per the documented '
            f'bounded-overflow simplification), verified all present pairs co-located '
            f'this epoch'
        )
    return batches
