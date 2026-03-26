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

import os
import sys
import time
import uuid

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import database_utils.milvus_db_connection as milvus_conn
from configs.settings import (
    HDC_DIM,
    SCALABILITY_K,
    SCALABILITY_N_VALUES,
    SCALABILITY_NOISE,
    SCALABILITY_SEED,
    SCALABILITY_V,
)
from database_utils.milvus_db_connection import ensure_people_collection
from tests.experiments.experiment_utils import (
    generate_canonical_persons,
    insert_noisy_variants,
    run_dedup_recall,
    save_report,
)


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
                    col = ensure_people_collection(col_name)

                    print(f"\n[SCALE] mode={mode}  N={n}  "
                          f"total_records={total_records}  collection={col_name}")

                    try:
                        # --- Generate canonical identities ---
                        canonical_persons = generate_canonical_persons(n)

                        # --- Insert noisy variants, recording insertion time ---
                        insert_start = time.perf_counter()
                        identity_to_milvus_ids, milvus_id_to_identity = insert_noisy_variants(
                            canonical_persons, variants_per_identity, noise_fraction,
                            seed, col_name,
                        )
                        col.flush()
                        insert_time_s = time.perf_counter() - insert_start

                        print(f"[SCALE] Inserted & flushed {total_records} records  "
                              f"insert_time={insert_time_s:.2f}s")

                        # --- Evaluate recall@K, recording query time ---
                        query_start = time.perf_counter()
                        recall_at_k, _, _, hits, total = run_dedup_recall(
                            canonical_persons,
                            identity_to_milvus_ids,
                            milvus_id_to_identity,
                            variants_per_identity,
                            noise_fraction,
                            seed,
                            top_k,
                            col_name,
                        )
                        query_time_s = time.perf_counter() - query_start

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
                output_path = save_report("scalability", mode, {
                    "mode":    mode,
                    "config":  config,
                    "results": mode_results,
                })
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
