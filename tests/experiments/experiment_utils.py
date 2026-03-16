"""
Shared utilities for HDC experiment tests.

Provides helpers used across multiple experiment files that are too specific
to live in noise_injection.py (which has no external dependencies).
"""

import copy
import random

from encoding_methods.encoding_and_search_milvus import find_closest_match_db
from tests.experiments.noise_injection import _NOISE_FUNCS


def _inject_single_field_noise(person: dict, field: str, rng: random.Random) -> dict:
    """
    Return a deep copy of `person` with exactly one field corrupted.

    Parameters
    ----------
    person : dict
        Normalized person dict (output of normalize_person_data).
    field : str
        The display field name to corrupt.  "address" is treated as the
        attrs.address key (matching how _NOISE_FUNCS stores it).
    rng : random.Random
        Seeded RNG for reproducibility.

    Returns
    -------
    dict
        Deep copy of person with only the specified field corrupted.
    """
    # Map display name "address" to the internal key "attrs.address"
    lookup_key = "attrs.address" if field == "address" else field

    noisy = copy.deepcopy(person)
    func = _NOISE_FUNCS.get(lookup_key)
    if func is None:
        return noisy

    if lookup_key.startswith("attrs."):
        attr_key = lookup_key.split(".", 1)[1]
        attrs = noisy.get("attrs")
        if isinstance(attrs, dict):
            current = attrs.get(attr_key, [])
            noisy["attrs"][attr_key] = func(current, rng)
    else:
        current = noisy.get(lookup_key)
        noisy[lookup_key] = func(current, rng)

    return noisy


def _compute_metrics(
    canonical_persons,
    identity_to_milvus_ids,
    milvus_id_to_identity,
    top_k,
    test_collection,
    query_builder,
):
    """
    Iterate over all canonical identities, build a query per identity using
    query_builder(canonical_person, identity_idx), search, and aggregate metrics.

    Returns
    -------
    tuple[float, float, float]
        (recall_at_k, mrr, hit_at_1)
    """
    recall_hits = 0
    hit_at_1_hits = 0
    reciprocal_ranks = []
    total = 0

    for identity_idx, canonical in enumerate(canonical_persons):
        query_person = query_builder(canonical, identity_idx)

        matches = find_closest_match_db(
            query_person,
            threshold=0.0,
            limit=top_k + 1,
            collection_name=test_collection,
        )

        # Exclude any result that exactly matches a stored variant of this identity
        # by identity label — the query itself was never inserted, so we do not
        # exclude by milvus id, just cap at top_k.
        neighbours = matches[:top_k]

        # Recall@K
        hit = any(
            milvus_id_to_identity.get(m["id"]) == identity_idx
            for m in neighbours
        )
        if hit:
            recall_hits += 1

        # Hit@1
        if neighbours and milvus_id_to_identity.get(neighbours[0]["id"]) == identity_idx:
            hit_at_1_hits += 1

        # MRR
        rr = 0.0
        for rank, m in enumerate(neighbours, 1):
            if milvus_id_to_identity.get(m["id"]) == identity_idx:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        total += 1

    recall_at_k = recall_hits / total if total > 0 else 0.0
    mrr         = sum(reciprocal_ranks) / total if total > 0 else 0.0
    hit_at_1    = hit_at_1_hits / total if total > 0 else 0.0

    return recall_at_k, mrr, hit_at_1
