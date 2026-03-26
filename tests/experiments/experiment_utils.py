"""
Shared utilities for HDC experiment tests.

Provides helpers used across multiple experiment files that are too specific
to live in noise_injection.py (which has no external dependencies).
"""

import copy
import json
import random
from datetime import datetime
from pathlib import Path

from dummy_data.generacion_base_de_datos import generate_data_chunk
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from tests.experiments.conftest import dataframe_row_to_person_dict
from tests.experiments.noise_injection import _NOISE_FUNCS, inject_noise
from utils.person_data_normalization import normalize_person_data


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


def generate_canonical_persons(n: int) -> list:
    """Generate N normalized canonical person dicts from synthetic data."""
    df = generate_data_chunk(n)
    persons = []
    for _, row in df.iterrows():
        raw = dataframe_row_to_person_dict(row)
        persons.append(normalize_person_data(raw))
    return persons


def insert_noisy_variants(
    canonical_persons,
    variants_per_identity: int,
    noise_fraction: float,
    seed: int,
    col_name: str,
    **store_kwargs,
) -> tuple:
    """
    Generate noisy variants for each canonical person and insert them into Milvus.

    Does NOT call col.flush() — callers are responsible for flushing.

    Parameters
    ----------
    canonical_persons : list
        Normalized person dicts (output of generate_canonical_persons or similar).
    variants_per_identity : int
        Number of noisy variants to generate per canonical identity.
    noise_fraction : float
        Fraction of fields to corrupt, passed to inject_noise().
    seed : int
        Base RNG seed; each (identity, variant) pair uses a deterministic offset.
    col_name : str
        Milvus collection name.
    **store_kwargs
        Extra keyword arguments forwarded to store_person() (e.g. field_weights,
        excluded_fields for the field-weighting experiment).

    Returns
    -------
    tuple[list, dict]
        (identity_to_milvus_ids, milvus_id_to_identity)
        identity_to_milvus_ids[i] is the list of Milvus IDs for identity i.
        milvus_id_to_identity maps each Milvus ID back to its identity index.
    """
    n = len(canonical_persons)
    identity_to_milvus_ids = [[] for _ in range(n)]
    milvus_id_to_identity  = {}

    for identity_idx, canonical in enumerate(canonical_persons):
        for variant_idx in range(variants_per_identity):
            rng = random.Random(
                seed + identity_idx * variants_per_identity + variant_idx
            )
            noisy     = inject_noise(canonical, noise_fraction, rng)
            milvus_id = store_person(noisy, collection_name=col_name, **store_kwargs)
            identity_to_milvus_ids[identity_idx].append(milvus_id)
            milvus_id_to_identity[milvus_id] = identity_idx

    return identity_to_milvus_ids, milvus_id_to_identity


def run_dedup_recall(
    canonical_persons,
    identity_to_milvus_ids,
    milvus_id_to_identity,
    variants_per_identity: int,
    noise_fraction: float,
    seed: int,
    top_k: int,
    col_name: str,
) -> tuple:
    """
    Evaluate deduplication recall@K over all stored variants.

    For each stored record (identified by its Milvus ID), reconstructs the same
    noisy query used during insertion, queries top-(K+1), excludes self, and checks
    whether any of the remaining top-K results belongs to the same canonical identity.

    Parameters
    ----------
    canonical_persons : list
        Normalized canonical person dicts.
    identity_to_milvus_ids : list
        identity_to_milvus_ids[i] → list of Milvus IDs for identity i.
    milvus_id_to_identity : dict
        Maps each Milvus ID to its canonical identity index.
    variants_per_identity : int
        Number of variants stored per identity.
    noise_fraction : float
        Noise fraction used during insertion (reproduced here for query generation).
    seed : int
        Base RNG seed (must match the one used in insert_noisy_variants).
    top_k : int
        K for Recall@K.
    col_name : str
        Milvus collection name.

    Returns
    -------
    tuple[float, float, float, int, int]
        (recall_at_k, mrr, hit_at_1, hits, total)
    """
    hits       = 0
    total      = 0
    mrr_sum    = 0.0
    hit1_count = 0

    for identity_idx, milvus_ids in enumerate(identity_to_milvus_ids):
        for variant_idx, query_milvus_id in enumerate(milvus_ids):
            query_rng = random.Random(
                seed + identity_idx * variants_per_identity + variant_idx
            )
            query_person = inject_noise(
                canonical_persons[identity_idx], noise_fraction, query_rng
            )

            matches = find_closest_match_db(
                query_person,
                threshold=0.0,
                limit=top_k + 1,
                collection_name=col_name,
            )
            neighbours = [m for m in matches if m["id"] != query_milvus_id][:top_k]

            # Recall@K
            hit = any(
                milvus_id_to_identity.get(m["id"]) == identity_idx
                for m in neighbours
            )
            if hit:
                hits += 1

            # Hit@1
            if neighbours and milvus_id_to_identity.get(neighbours[0]["id"]) == identity_idx:
                hit1_count += 1

            # MRR
            rr = 0.0
            for rank, m in enumerate(neighbours, 1):
                if milvus_id_to_identity.get(m["id"]) == identity_idx:
                    rr = 1.0 / rank
                    break
            mrr_sum += rr
            total   += 1

    recall_at_k = hits / total if total > 0 else 0.0
    mrr         = mrr_sum / total if total > 0 else 0.0
    hit_at_1    = hit1_count / total if total > 0 else 0.0
    return recall_at_k, mrr, hit_at_1, hits, total


def save_report(prefix: str, mode: str, report: dict) -> Path:
    """
    Serialise `report` to test_results/{prefix}_{mode}_{timestamp}.json.

    Returns the Path of the saved file.
    """
    project_root = Path(__file__).resolve().parents[2]
    output_dir   = project_root / "test_results"
    output_dir.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{prefix}_{mode}_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2))
    return output_path
