"""
Experiment 8 — Dimensionality Sweep for HDC data reconciliation.

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
from dummy_data.generacion_base_de_datos import generate_data_chunk
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from utils.person_data_normalization import normalize_person_data
from tests.experiments.noise_injection import inject_noise
from tests.experiments.conftest import dataframe_row_to_person_dict


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
        df = generate_data_chunk(n_identities)
        canonical_persons = []
        for _, row in df.iterrows():
            raw = dataframe_row_to_person_dict(row)
            canonical_persons.append(normalize_person_data(raw))

        config = {
            "dim_values":            dim_values,
            "n_identities":          n_identities,
            "variants_per_identity": variants_per_identity,
            "noise_fraction":        noise_fraction,
            "top_k":                 top_k,
            "seed":                  seed,
        }

        project_root = Path(__file__).resolve().parents[2]
        output_dir   = project_root / "test_results"
        output_dir.mkdir(exist_ok=True)

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
                    col = ensure_people_collection(col_name, include_embedding=False)

                    print(
                        f"\n[DIM] mode={mode}  dim={dim}  "
                        f"collection={col_name}"
                    )

                    try:
                        # --- Insert all noisy variants; measure wall-clock ---
                        identity_to_milvus_ids: list = [[] for _ in range(n_identities)]
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
                                milvus_id = store_person(
                                    noisy,
                                    collection_name=col_name,
                                )
                                identity_to_milvus_ids[identity_idx].append(milvus_id)
                                milvus_id_to_identity[milvus_id] = identity_idx

                        col.flush()
                        insert_time = time.perf_counter() - insert_start

                        print(
                            f"[DIM] Inserted & flushed {total_records} records "
                            f"for dim={dim}  insert_time={insert_time:.2f}s"
                        )

                        # --- Evaluate Recall@K, MRR, Hit@1; measure wall-clock ---
                        hits   = 0
                        total  = 0
                        mrr_sum   = 0.0
                        hit1_count = 0

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

                                # Recall@K
                                hit = any(
                                    milvus_id_to_identity.get(m["id"]) == identity_idx
                                    for m in neighbours
                                )
                                if hit:
                                    hits += 1

                                # MRR
                                rr = 0.0
                                for rank, m in enumerate(neighbours, 1):
                                    if milvus_id_to_identity.get(m["id"]) == identity_idx:
                                        rr = 1.0 / rank
                                        break
                                mrr_sum += rr

                                # Hit@1
                                if neighbours and milvus_id_to_identity.get(neighbours[0]["id"]) == identity_idx:
                                    hit1_count += 1

                                total += 1

                        query_time = time.perf_counter() - query_start

                        recall_at_k = hits / total if total > 0 else 0.0
                        mrr         = mrr_sum / total if total > 0 else 0.0
                        hit_at_1    = hit1_count / total if total > 0 else 0.0

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
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"dimensionality_{mode}_{timestamp}.json"
                report = {
                    "mode":    mode,
                    "config":  config,
                    "results": mode_results,
                }
                output_path.write_text(json.dumps(report, indent=2))
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
