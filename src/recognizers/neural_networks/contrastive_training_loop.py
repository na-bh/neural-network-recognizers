import dataclasses

import numpy as np

from .training_loop import RecognitionTrainingLoop
from .contrastive_batching import group_into_batches_pair_aware


@dataclasses.dataclass
class ContrastiveRecognitionTrainingLoop(RecognitionTrainingLoop):
    """Identical to RecognitionTrainingLoop in every respect (loss function,
    evaluation, optimizer, early stopping) EXCEPT batch construction: after
    the standard length-sorted, token-budget-greedy grouping, pairs
    registered in pair_partner_map are forced into the same batch via a
    post-processing merge. pair_partner_map is set externally (via
    set_pair_partner_map) after training data is loaded, since the mapping
    is keyed on the identity of the loaded Example objects, which do not
    exist yet at TrainingLoop construction time. When pair_partner_map is
    empty (e.g. while generating VALIDATION batches for this same
    experiment, since validation-short is copied unchanged and carries no
    pairing), this reduces to byte-for-byte the same batches
    RecognitionTrainingLoop would produce.

    Also accumulates per-call batch-size statistics (mean/max tokens,
    fraction of batches over max_tokens_per_batch, mean overflow magnitude
    among the ones that overflow) into self.batch_stats_log -- one entry
    per generate_batches() call, i.e. one for the single up-front
    validation-batch call plus one per training epoch. Distinguished by
    n_pairs_present (0 for the validation call, since no training pair ids
    ever match a validation example's id; >0 for every training epoch).
    Written out to disk by the caller (train.py) after training finishes,
    for post-hoc analysis of whether overflow-heavy epochs/trials show
    systematically different results."""

    def __post_init__(self):
        self.pair_partner_map: dict[int, int] = {}
        self.batch_stats_log: list[dict] = []

    def set_pair_partner_map(self, pair_partner_map: dict[int, int]) -> None:
        self.pair_partner_map = pair_partner_map

    def generate_batches(self, examples, max_tokens):
        batches = group_into_batches_pair_aware(
            examples,
            lambda b, n: b * n <= max_tokens,
            self.pair_partner_map,
        )
        self.batch_stats_log.append(
            _compute_batch_stats(batches, max_tokens, self.pair_partner_map)
        )
        return batches


def _compute_batch_stats(batches, max_tokens, pair_partner_map):
    token_counts = []
    for batch in batches:
        max_len = max(len(x[0]) for x in batch)
        token_counts.append(len(batch) * max_len)
    token_counts = np.array(token_counts, dtype=float)
    over_budget = token_counts[token_counts > max_tokens]
    n_pairs_present = 0
    if pair_partner_map:
        present_ids = {id(ex) for batch in batches for ex in batch}
        seen = set()
        for a_id, b_id in pair_partner_map.items():
            key = frozenset((a_id, b_id))
            if key in seen:
                continue
            seen.add(key)
            if a_id in present_ids and b_id in present_ids:
                n_pairs_present += 1
    return {
        "n_pairs_present": n_pairs_present,
        "n_batches": len(batches),
        "max_tokens_per_batch": max_tokens,
        "mean_tokens_per_batch": float(np.mean(token_counts)) if len(token_counts) else None,
        "max_tokens_per_batch_observed": float(np.max(token_counts)) if len(token_counts) else None,
        "fraction_batches_over_budget": float(len(over_budget) / len(token_counts)) if len(token_counts) else None,
        "mean_overflow_pct_among_overflowed": (
            float(np.mean((over_budget - max_tokens) / max_tokens * 100)) if len(over_budget) else None
        ),
    }
