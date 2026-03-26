"""
Experiment 8 — Dimensionality Sweep for HDC-based data reconciliation.

Measures how recall@K, MRR, and Hit@1 change as the hypervector
dimension is varied, holding all other parameters fixed.

Setup per dimension
-------------------
1. Patch module-level HDC_DIM / DIMENSION to the target dimension.
2. Create an ephemeral Milvus collection using that dimension.
3. Generate N synthetic canonical identities.
4. For each identity, produce V noisy variants using inject_noise().
5. Insert all N×V records; measure total encoding+insert wall-clock time.
6. Flush the collection.
7. For each stored record, query top-(K+1), exclude self, compute
   Recall@K, MRR, and Hit@1; measure total query wall-clock time.
8. Drop the ephemeral collection.
9. Restore the original HDC_DIM values.

Run
---
    pytest tests/experiments/test_dimensionality.py -v -s

Environment variables
---------------------
    DIM_SWEEP_VALUES    Comma-separated list of HDC dims to test (default: 1000,2000,5000,10000)
    DIM_SWEEP_N         Number of canonical identities (default: 200)
    DIM_SWEEP_V         Noisy variants per identity (default: 3)
    DIM_SWEEP_NOISE     Noise fraction passed to inject_noise (default: 0.30)
    DIM_SWEEP_K         K for Recall@K (default: 5)
    DIM_SWEEP_SEED      RNG seed (default: 42)
"""

import os
import sys
import time
import uuid

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import database_utils.milvus_db_connection as milvus_conn
import encoding_methods.encoding_and_search_milvus as enc_module
from configs.settings import (
    DIM_SWEEP_K,
    DIM_SWEEP_N,
    DIM_SWEEP_NOISE,
    DIM_SWEEP_SEED,
    DIM_SWEEP_V,
    DIM_SWEEP_VALUES,
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

class TestDimensionalitySweep:

    def test_dimensionality_sweep(self):
        # --- Config from env ---
        raw_dims = os.environ.get("DIM_SWEEP_VALUES", "")
        if raw_dims.strip():
            dim_values = [int(x.strip()) for x in raw_dims.split(",") if x.strip()]
        else:
            dim_values = list(DIM_SWEEP_VALUES)

        n_identities          = int(os.environ.get("DIM_SWEEP_N",     DIM_SWEEP_N))
        variants_per_identity = int(os.environ.get("DIM_SWEEP_V",     DIM_SWEEP_V))
        noise_fraction        = float(os.environ.get("DIM_SWEEP_NOISE", DIM_SWEEP_NOISE))
        top_k                 = int(os.environ.get("DIM_SWEEP_K",     DIM_SWEEP_K))
        seed                  = int(os.environ.get("DIM_SWEEP_SEED",  DIM_SWEEP_SEED))
        total_records         = n_identities * variants_per_identity

        # --- Pre-generate canonical identities once (shared across all dims) ---
        canonical_persons = generate_canonical_persons(n_identities)

        config = {
            "dim_values":            dim_values,
            "n_identities":          n_identities,
            "variants_per_identity": variants_per_identity,
            "noise_fraction":        noise_fraction,
            "top_k":                 top_k,
            "seed":                  seed,
        }

        for mode in ["binary", "float"]:
            # Switch vector mode inline (same pattern as test_field_weighting.py)
            original_mode = milvus_conn.VECTOR_MODE
            milvus_conn.VECTOR_MODE = mode

            try:
                mode_results = []

                for dim in dim_values:
                    # Patch module-level HDC_DIM / DIMENSION
                    original_dim_enc    = enc_module.HDC_DIM
                    original_dim_milvus = milvus_conn.HDC_DIM

                    enc_module.HDC_DIM    = dim
                    enc_module.DIMENSION  = dim
                    milvus_conn.HDC_DIM   = dim

                    col_name = f"dim_{uuid.uuid4().hex[:10]}"
                    col = ensure_people_collection(col_name)

                    print(
                        f"\n[DIM] mode={mode}  dim={dim}  "
                        f"collection={col_name}"
                    )

                    try:
                        # --- Insert all noisy variants; measure wall-clock ---
                        insert_start = time.perf_counter()
                        identity_to_milvus_ids, milvus_id_to_identity = insert_noisy_variants(
                            canonical_persons, variants_per_identity, noise_fraction,
                            seed, col_name,
                        )
                        col.flush()
                        insert_time = time.perf_counter() - insert_start

                        print(
                            f"[DIM] Inserted & flushed {total_records} records "
                            f"for dim={dim}  insert_time={insert_time:.2f}s"
                        )

                        # --- Evaluate Recall@K, MRR, Hit@1; measure wall-clock ---
                        query_start = time.perf_counter()
                        recall_at_k, mrr, hit_at_1, hits, total = run_dedup_recall(
                            canonical_persons,
                            identity_to_milvus_ids,
                            milvus_id_to_identity,
                            variants_per_identity,
                            noise_fraction,
                            seed,
                            top_k,
                            col_name,
                        )
                        query_time = time.perf_counter() - query_start

                        print(
                            f"[DIM] mode={mode}  dim={dim}  "
                            f"recall@{top_k}={recall_at_k:.3f}  "
                            f"mrr={mrr:.3f}  hit@1={hit_at_1:.3f}  "
                            f"({hits}/{total})  "
                            f"query_time={query_time:.2f}s"
                        )

                        mode_results.append(
                            {
                                "dim":            dim,
                                "total_records":  total_records,
                                "recall_at_k":    round(recall_at_k, 6),
                                "mrr":            round(mrr, 6),
                                "hit_at_1":       round(hit_at_1, 6),
                                "insert_time_s":  round(insert_time, 4),
                                "query_time_s":   round(query_time, 4),
                            }
                        )

                    finally:
                        # Drop ephemeral collection regardless of success/failure
                        try:
                            col.drop()
                            # Also evict from cache so next dim gets a fresh collection
                            cache_key = f"{col_name}_{mode}"
                            milvus_conn._collection_cache.pop(cache_key, None)
                        except Exception as drop_err:
                            print(
                                f"[DIM] Warning: could not drop collection "
                                f"{col_name}: {drop_err}"
                            )

                        # Restore original HDC_DIM values
                        enc_module.HDC_DIM    = original_dim_enc
                        enc_module.DIMENSION  = original_dim_enc
                        milvus_conn.HDC_DIM   = original_dim_milvus

                # --- Save JSON report ---
                output_path = save_report("dimensionality", mode, {
                    "mode":    mode,
                    "config":  config,
                    "results": mode_results,
                })
                print(f"\n[DIM] Results saved to {output_path.name}")

                # --- Print summary table ---
                col_dim    = 10
                col_total  = 14
                col_recall = 10
                col_mrr    =  8
                col_hit1   =  7
                col_insert = 10
                col_query  = 10

                print(f"\nMode: {mode}")
                print(
                    f"  {'HDC_DIM':>{col_dim}}  "
                    f"{'Total Records':>{col_total}}  "
                    f"{'Recall@' + str(top_k):>{col_recall}}  "
                    f"{'MRR':>{col_mrr}}  "
                    f"{'Hit@1':>{col_hit1}}  "
                    f"{'Insert(s)':>{col_insert}}  "
                    f"{'Query(s)':>{col_query}}  "
                    f"Chart"
                )
                print(
                    f"  {'-'*col_dim}  "
                    f"{'-'*col_total}  "
                    f"{'-'*col_recall}  "
                    f"{'-'*col_mrr}  "
                    f"{'-'*col_hit1}  "
                    f"{'-'*col_insert}  "
                    f"{'-'*col_query}  "
                    f"{'-'*40}"
                )
                for r in mode_results:
                    recall = r["recall_at_k"]
                    filled = round(recall * 40)
                    chart  = "#" * filled + "-" * (40 - filled)
                    print(
                        f"  {r['dim']:>{col_dim}}  "
                        f"{r['total_records']:>{col_total}}  "
                        f"{recall:>{col_recall}.1%}  "
                        f"{r['mrr']:>{col_mrr}.3f}  "
                        f"{r['hit_at_1']:>{col_hit1}.1%}  "
                        f"{r['insert_time_s']:>{col_insert}.2f}  "
                        f"{r['query_time_s']:>{col_query}.2f}  "
                        f"{chart}"
                    )

            finally:
                milvus_conn.VECTOR_MODE = original_mode
