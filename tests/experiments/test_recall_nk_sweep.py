"""
2D sweep experiment: collection size (N) × top_k → Recall@k.

Isolates the independent effects of N (number of identities) and k (top-k
search depth) on deduplication recall, keeping all other variables fixed.

Design
------
For each vector_mode in {binary, float}:
  For each N in N_VALUES:
    1. Create a fresh Milvus collection.
    2. Generate N identities × VARIANTS_PER_IDENTITY noisy variants.
    3. Insert and flush all records.
    4. For every stored record as a query:
       - Search top-(max_k + 1), exclude self.
       - Record the ordered neighbour list.
    5. For each k in K_VALUES, slice neighbour lists and compute:
       - recall_at_k  : fraction of queries where ≥1 duplicate appears in top-k.
       - recall_at_2  : same fixed k=2 metric for cross-row comparison.
    6. Drop the collection and move to the next (mode, N) pair.

Output
------
Saves test_results/recall_nk_sweep_<timestamp>.json with a flat results list.
Prints a pivot table per mode to stdout.

Run
---
    pytest tests/experiments/test_recall_nk_sweep.py -v -s
"""

import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import database_utils.milvus_db_connection as milvus_conn
from configs.settings import DEFAULT_SEED, HDC_DIM
from database_utils.milvus_db_connection import _collection_cache, ensure_people_collection
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from tests.experiments.experiment_utils import generate_canonical_persons
from tests.experiments.noise_injection import inject_noise

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

N_VALUES      = [200, 1000, 5000]
K_VALUES      = [2, 3, 5]
NOISE_FRACTION = 0.3
VARIANTS       = 3          # variants_per_identity
SEED           = DEFAULT_SEED
MODES          = ["binary", "float"]

_MAX_K = max(K_VALUES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collection(mode: str) -> str:
    """Create a fresh ephemeral Milvus collection for the given mode."""
    name = f"people_nksweep_{uuid.uuid4().hex[:8]}"
    ensure_people_collection(name)
    return name


def _drop_collection(name: str, mode: str) -> None:
    """Drop the ephemeral collection and evict it from the connection cache."""
    _collection_cache.pop(f"{name}_{mode}", None)
    try:
        col = ensure_people_collection(name)
        col.drop()
    except Exception as exc:
        print(f"  [WARN] Could not drop collection '{name}': {exc}")


def _insert_and_flush(
    canonical_persons: list,
    noise_fraction: float,
    seed: int,
    variants: int,
    col_name: str,
) -> tuple:
    """
    Insert noisy variants for every canonical identity; return ground-truth maps.

    Returns
    -------
    identity_to_milvus_ids : list[list[int]]
    milvus_id_to_identity  : dict[int, int]
    """
    n = len(canonical_persons)
    identity_to_milvus_ids = [[] for _ in range(n)]
    milvus_id_to_identity: dict = {}

    for identity_idx, canonical in enumerate(canonical_persons):
        for variant_idx in range(variants):
            rng = random.Random(seed + identity_idx * variants + variant_idx)
            noisy     = inject_noise(canonical, noise_fraction, rng)
            milvus_id = store_person(noisy, collection_name=col_name)
            identity_to_milvus_ids[identity_idx].append(milvus_id)
            milvus_id_to_identity[milvus_id] = identity_idx

    col = ensure_people_collection(col_name)
    col.flush()
    return identity_to_milvus_ids, milvus_id_to_identity


def _evaluate_all_k(
    canonical_persons: list,
    identity_to_milvus_ids: list,
    milvus_id_to_identity: dict,
    noise_fraction: float,
    seed: int,
    variants: int,
    col_name: str,
    k_values: list,
    fixed_k: int = 2,
) -> dict:
    """
    Query every stored record once (limit = max_k + 1) and derive recall for
    all k values in a single pass over the data.

    Returns
    -------
    dict mapping k → {"hits": int, "total": int, "recall": float}
    Plus a "recall_at_2" key derived from fixed_k.
    """
    max_k = max(k_values)

    # Counters: {k: hits}
    hits   = {k: 0 for k in k_values}
    total  = 0
    hits_2 = 0   # fixed recall@2

    for identity_idx, milvus_ids in enumerate(identity_to_milvus_ids):
        for variant_idx, query_milvus_id in enumerate(milvus_ids):
            # Reproduce the exact noisy variant used during insertion
            query_rng    = random.Random(seed + identity_idx * variants + variant_idx)
            query_person = inject_noise(
                canonical_persons[identity_idx], noise_fraction, query_rng
            )

            # Single Milvus search — request max_k + 1 to allow self-exclusion
            matches    = find_closest_match_db(
                query_person,
                threshold=0.0,
                limit=max_k + 1,
                collection_name=col_name,
            )
            neighbours = [m for m in matches if m["id"] != query_milvus_id]

            # Accumulate hits for each k by slicing the shared neighbour list
            for k in k_values:
                top = neighbours[:k]
                if any(milvus_id_to_identity.get(m["id"]) == identity_idx for m in top):
                    hits[k] += 1

            # Fixed recall@2
            top2 = neighbours[:fixed_k]
            if any(milvus_id_to_identity.get(m["id"]) == identity_idx for m in top2):
                hits_2 += 1

            total += 1

    results = {}
    for k in k_values:
        recall = hits[k] / total if total > 0 else 0.0
        results[k] = {"hits": hits[k], "total": total, "recall": recall}

    results["recall_at_2_fixed"] = {
        "hits": hits_2,
        "total": total,
        "recall": hits_2 / total if total > 0 else 0.0,
    }
    return results


def _print_pivot(mode: str, rows: list) -> None:
    """Print a pivot table for one mode: rows=N, cols=k."""
    k_cols = sorted({r["top_k"] for r in rows})
    n_vals = sorted({r["n_identities"] for r in rows})

    # Build lookup: (n, k) → recall_at_k
    cell = {}
    for r in rows:
        cell[(r["n_identities"], r["top_k"])] = r["recall_at_k"]

    header  = f"  {'N':>8} | " + " | ".join(f"k={k:<5}" for k in k_cols)
    divider = "  " + "-" * 9 + "+" + ("+".join(["-" * 8] * len(k_cols)))
    print(f"\n  Mode: {mode}")
    print(header)
    print(divider)
    for n in n_vals:
        vals = " | ".join(f"{cell.get((n, k), float('nan')):.3f} " for k in k_cols)
        print(f"  {n:>8} | {vals}")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestRecallNKSweep:

    def test_recall_nk_sweep(self):
        all_results = []

        for mode in MODES:
            # Switch Milvus vector mode
            original_mode       = milvus_conn.VECTOR_MODE
            milvus_conn.VECTOR_MODE = mode

            mode_rows = []

            try:
                for n in N_VALUES:
                    total_records = n * VARIANTS
                    print(
                        f"\n[SWEEP] mode={mode}  N={n}  "
                        f"records={total_records}  "
                        f"k_values={K_VALUES}"
                    )

                    # --- Generate data (outside timed section) ---
                    canonical_persons = generate_canonical_persons(n)

                    # --- Create collection ---
                    col_name = _make_collection(mode)
                    print(f"  Collection: {col_name}")

                    t0 = time.perf_counter()
                    try:
                        # --- Insert + flush ---
                        identity_to_milvus_ids, milvus_id_to_identity = _insert_and_flush(
                            canonical_persons,
                            NOISE_FRACTION,
                            SEED,
                            VARIANTS,
                            col_name,
                        )
                        t_insert = time.perf_counter() - t0
                        print(f"  Inserted & flushed {total_records} records in {t_insert:.1f}s")

                        # --- Evaluate all k values in one pass ---
                        t1 = time.perf_counter()
                        k_metrics = _evaluate_all_k(
                            canonical_persons,
                            identity_to_milvus_ids,
                            milvus_id_to_identity,
                            NOISE_FRACTION,
                            SEED,
                            VARIANTS,
                            col_name,
                            K_VALUES,
                            fixed_k=2,
                        )
                        t_query = time.perf_counter() - t1
                        print(f"  Queried {total_records} records in {t_query:.1f}s")

                        # --- Flatten into result rows, one per (mode, N, k) ---
                        for k in K_VALUES:
                            m        = k_metrics[k]
                            fixed_m  = k_metrics["recall_at_2_fixed"]
                            row = {
                                "vector_mode":    mode,
                                "n_identities":   n,
                                "top_k":          k,
                                "total_records":  total_records,
                                "recall_at_k":    round(m["recall"], 6),
                                "recall_at_2":    round(fixed_m["recall"], 6),
                                "hits_at_k":      m["hits"],
                                "total_queries":  m["total"],
                                "insert_time_s":  round(t_insert, 2),
                                "query_time_s":   round(t_query, 2),
                            }
                            all_results.append(row)
                            mode_rows.append(row)
                            print(
                                f"  recall@{k}={m['recall']:.3f}  "
                                f"recall@2_fixed={fixed_m['recall']:.3f}  "
                                f"({m['hits']}/{m['total']})"
                            )

                    finally:
                        _drop_collection(col_name, mode)

            finally:
                milvus_conn.VECTOR_MODE = original_mode

            _print_pivot(mode, mode_rows)

        # --- Save JSON ---
        project_root = Path(__file__).resolve().parents[2]
        output_dir   = project_root / "test_results"
        output_dir.mkdir(exist_ok=True)
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path  = output_dir / f"recall_nk_sweep_{timestamp}.json"

        report = {
            "config": {
                "n_values":              N_VALUES,
                "k_values":              K_VALUES,
                "noise_fraction":        NOISE_FRACTION,
                "variants_per_identity": VARIANTS,
                "hdim":                  HDC_DIM,
                "seed":                  SEED,
            },
            "results": all_results,
        }
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\n[SWEEP] Results saved to {output_path.name}")

        # Sanity assertion: at least one result should have recall > 0
        assert any(r["recall_at_k"] > 0 for r in all_results), (
            "All recall@k values are 0 — something is wrong with encoding or search."
        )
