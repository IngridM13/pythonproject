"""
Experiment 4 — Scalability for HDC-based data reconciliation.

Measures how insertion time, query time, and deduplication recall@K scale
with the number of identities stored in Milvus, for both binary and float
vector modes.

Setup per N
-----------
1. Generate N synthetic canonical identities.
2. For each identity, produce V noisy variants using inject_noise().
3. Insert all N×V records into an ephemeral Milvus collection.
4. Record insertion wall-clock time (first store_person call → col.flush()).
5. For each inserted record, query top-(K+1), exclude self, check whether any
   of the remaining top-K results belongs to the same identity.
6. Record query wall-clock time (total time for all queries).
7. Report recall@K = hits / (N×V).

Run
---
    pytest tests/experiments/test_scalability.py -v -s

Environment variables
---------------------
    SCALABILITY_N_VALUES    Comma-separated list of N values (default: from settings)
    SCALABILITY_V           Noisy variants per identity (default: 3)
    SCALABILITY_NOISE       Noise fraction passed to inject_noise (default: 0.30)
    SCALABILITY_K           K for recall@K (default: 5)
    SCALABILITY_SEED        RNG seed (default: 42)
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
from configs.settings import (
    DEFAULT_SEED,
    HDC_DIM,
    SCALABILITY_K,
    SCALABILITY_N_VALUES,
    SCALABILITY_NOISE,
    SCALABILITY_SEED,
    SCALABILITY_V,
)
from database_utils.milvus_db_connection import ensure_people_collection
from dummy_data.generacion_base_de_datos import generate_data_chunk
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from utils.person_data_normalization import normalize_person_data
from tests.experiments.conftest import dataframe_row_to_person_dict
from tests.experiments.noise_injection import inject_noise


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestScalability:

    def test_scalability(self):
        # --- Config from env ---
        raw_n_values = os.environ.get("SCALABILITY_N_VALUES", "")
        if raw_n_values.strip():
            n_values = [int(x.strip()) for x in raw_n_values.split(",") if x.strip()]
        else:
            n_values = list(SCALABILITY_N_VALUES)

        variants_per_identity = int(os.environ.get("SCALABILITY_V", SCALABILITY_V))
        noise_fraction        = float(os.environ.get("SCALABILITY_NOISE", SCALABILITY_NOISE))
        top_k                 = int(os.environ.get("SCALABILITY_K", SCALABILITY_K))
        seed                  = int(os.environ.get("SCALABILITY_SEED", SCALABILITY_SEED))

        config = {
            "n_values":              n_values,
            "variants_per_identity": variants_per_identity,
            "noise_fraction":        noise_fraction,
            "top_k":                 top_k,
            "hdim":                  HDC_DIM,
            "seed":                  seed,
        }

        project_root = Path(__file__).resolve().parents[2]
        output_dir   = project_root / "test_results"
        output_dir.mkdir(exist_ok=True)

        for mode in ["binary", "float"]:
            original_mode = milvus_conn.VECTOR_MODE
            milvus_conn.VECTOR_MODE = mode

            try:
                mode_results = []

                print(f"\n[SCALE] mode={mode}  n_values={n_values}  "
                      f"variants_per_identity={variants_per_identity}  "
                      f"noise_fraction={noise_fraction}  top_k={top_k}  seed={seed}")

                for n in n_values:
                    total_records = n * variants_per_identity

                    # Create a fresh ephemeral collection for this N
                    col_name = f"scale_{uuid.uuid4().hex[:10]}"
                    col = ensure_people_collection(col_name, include_embedding=False)

                    print(f"\n[SCALE] mode={mode}  N={n}  "
                          f"total_records={total_records}  collection={col_name}")

                    try:
                        # --- Generate canonical identities ---
                        df = generate_data_chunk(n)
                        canonical_persons = []
                        for _, row in df.iterrows():
                            raw = dataframe_row_to_person_dict(row)
                            canonical_persons.append(normalize_person_data(raw))

                        # --- Insert noisy variants, recording insertion time ---
                        identity_to_milvus_ids: list = [[] for _ in range(n)]
                        milvus_id_to_identity:  dict = {}

                        insert_start = time.perf_counter()

                        for identity_idx, canonical in enumerate(canonical_persons):
                            for variant_idx in range(variants_per_identity):
                                variant_rng = random.Random(
                                    seed
                                    + identity_idx * variants_per_identity
                                    + variant_idx
                                )
                                noisy = inject_noise(canonical, noise_fraction, variant_rng)
                                milvus_id = store_person(noisy, collection_name=col_name)
                                identity_to_milvus_ids[identity_idx].append(milvus_id)
                                milvus_id_to_identity[milvus_id] = identity_idx

                        col.flush()
                        insert_end = time.perf_counter()
                        insert_time_s = insert_end - insert_start

                        print(f"[SCALE] Inserted & flushed {total_records} records  "
                              f"insert_time={insert_time_s:.2f}s")

                        # --- Evaluate recall@K, recording query time ---
                        hits  = 0
                        total = 0

                        query_start = time.perf_counter()

                        for identity_idx, milvus_ids in enumerate(identity_to_milvus_ids):
                            for variant_idx, query_milvus_id in enumerate(milvus_ids):
                                query_rng = random.Random(
                                    seed
                                    + identity_idx * variants_per_identity
                                    + variant_idx
                                )
                                query_person = inject_noise(
                                    canonical_persons[identity_idx],
                                    noise_fraction,
                                    query_rng,
                                )

                                matches = find_closest_match_db(
                                    query_person,
                                    threshold=0.0,
                                    limit=top_k + 1,
                                    collection_name=col_name,
                                )

                                neighbours = [
                                    m for m in matches if m["id"] != query_milvus_id
                                ][:top_k]

                                hit = any(
                                    milvus_id_to_identity.get(m["id"]) == identity_idx
                                    for m in neighbours
                                )
                                if hit:
                                    hits += 1
                                total += 1

                        query_end = time.perf_counter()
                        query_time_s = query_end - query_start

                        recall_at_k = hits / total if total > 0 else 0.0
                        print(
                            f"[SCALE] mode={mode}  N={n}  "
                            f"recall@{top_k}={recall_at_k:.3f}  "
                            f"({hits}/{total})  "
                            f"query_time={query_time_s:.2f}s"
                        )

                        mode_results.append({
                            "n":              n,
                            "total_records":  total_records,
                            "recall_at_k":    round(recall_at_k, 6),
                            "hits":           hits,
                            "total":          total,
                            "insert_time_s":  round(insert_time_s, 4),
                            "query_time_s":   round(query_time_s, 4),
                        })

                    finally:
                        try:
                            col.drop()
                        except Exception as drop_err:
                            print(
                                f"[SCALE] Warning: could not drop collection "
                                f"{col_name}: {drop_err}"
                            )

                # --- Save JSON report ---
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"scalability_{mode}_{timestamp}.json"
                report = {
                    "mode":    mode,
                    "config":  config,
                    "results": mode_results,
                }
                output_path.write_text(json.dumps(report, indent=2))
                print(f"\n[SCALE] Results saved to {output_path.name}")

                # --- Print summary table ---
                col_n      = 7
                col_total  = 14
                col_recall = 10
                col_insert = 12
                col_query  = 12
                col_chart  = BAR_WIDTH = 30

                print(f"\nMode: {mode}")
                print(
                    f"  {'N':>{col_n}}  "
                    f"{'Total Records':>{col_total}}  "
                    f"{'Recall@' + str(top_k):>{col_recall}}  "
                    f"{'Insert(s)':>{col_insert}}  "
                    f"{'Query(s)':>{col_query}}  "
                    f"Chart"
                )
                print(
                    f"  {'-'*col_n}  "
                    f"{'-'*col_total}  "
                    f"{'-'*col_recall}  "
                    f"{'-'*col_insert}  "
                    f"{'-'*col_query}  "
                    f"{'-'*col_chart}"
                )
                for row in mode_results:
                    filled = round(row["recall_at_k"] * col_chart)
                    chart  = "#" * filled + "-" * (col_chart - filled)
                    print(
                        f"  {row['n']:>{col_n}}  "
                        f"{row['total_records']:>{col_total}}  "
                        f"{row['recall_at_k']:>{col_recall}.3f}  "
                        f"{row['insert_time_s']:>{col_insert}.2f}  "
                        f"{row['query_time_s']:>{col_query}.2f}  "
                        f"{chart}"
                    )

            finally:
                milvus_conn.VECTOR_MODE = original_mode
