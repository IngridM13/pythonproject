"""
Experiment 4 — Scalability for HDC-based data reconciliation.

Measures how insertion time, query time, and deduplication recall@K scale
with the number of identities stored in Milvus, for both binary and float
vector modes.

Setup per N (incremental across the sweep)
--------------------------------------------
Sizes are processed in ascending order and the collection is grown
incrementally instead of being rebuilt from scratch at every N: the
identities (and their noisy variants) generated and inserted for a smaller N
are kept and reused as the base for the next, larger N.

1. Generate only the *delta* of new synthetic canonical identities needed to
   go from the previous N to the current one (first step generates N0).
2. For each new identity, produce V noisy variants using inject_noise().
3. Insert only the new N_delta×V records into the (already-populated,
   persistent-for-this-mode) Milvus collection.
4. Record insertion wall-clock time for that delta only (marginal insert
   time — the cost of growing the collection from N_prev to N).
5. For every inserted record so far (old + new), query top-(K+1), exclude
   self, check whether any of the remaining top-K results belongs to the
   same identity. This must be re-run in full each step because the search
   space (and thus recall) changes as the collection grows.
6. Record query wall-clock time (total time for all queries at this N).
7. Report recall@K = hits / (N×V).
8. Drop the collection once per mode, after finishing the whole sweep
   (instead of once per N).

Run
---
    pytest tests/experiments/test_scalability.py -v -s

Environment variables
---------------------
    SCALABILITY_N_VALUES    Comma-separated list of N values (default: from settings)
    SCALABILITY_V           Noisy variants per identity (default: 3)
    SCALABILITY_NOISE       Noise fraction passed to inject_noise (default: 0.30)
    SCALABILITY_TOP_K           K for recall@K (default: 5)
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
    SCALABILITY_TOP_K,
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
        top_k                 = int(os.environ.get("SCALABILITY_TOP_K", SCALABILITY_TOP_K))
        seed                  = int(os.environ.get("SCALABILITY_SEED", SCALABILITY_SEED))

        config = {
            "n_values":              n_values,
            "variants_per_identity": variants_per_identity,
            "noise_fraction":        noise_fraction,
            "top_k":                 top_k,
            "hdim":                  HDC_DIM,
            "seed":                  seed,
        }

        n_values_sorted = sorted(n_values)
        if n_values_sorted != n_values:
            print(f"[SCALE] Note: n_values reordered ascending for incremental "
                  f"reuse: {n_values} -> {n_values_sorted}")

        for mode in ["binary", "float"]:
            original_mode = os.environ.get("MILVUS_VECTOR_MODE")
            os.environ["MILVUS_VECTOR_MODE"] = mode

            try:
                mode_results = []

                print(f"\n[SCALE] mode={mode}  n_values={n_values_sorted}  "
                      f"variants_per_identity={variants_per_identity}  "
                      f"noise_fraction={noise_fraction}  top_k={top_k}  seed={seed}")

                # One persistent collection per mode — grown incrementally
                # instead of recreated from scratch at every N.
                col_name = f"scale_{uuid.uuid4().hex[:10]}"
                col = ensure_people_collection(col_name)

                canonical_persons: list = []
                identity_to_milvus_ids: list = []
                milvus_id_to_identity: dict = {}

                try:
                    for n in n_values_sorted:
                        total_records = n * variants_per_identity
                        delta_n = n - len(canonical_persons)

                        print(f"\n[SCALE] mode={mode}  N={n}  "
                              f"(+{max(delta_n, 0)} new identities)  "
                              f"total_records={total_records}  collection={col_name}")

                        if delta_n > 0:
                            # --- Generate only the new canonical identities ---
                            delta_persons = generate_canonical_persons(delta_n)
                            offset = len(canonical_persons)

                            # --- Insert their noisy variants, recording marginal insertion time ---
                            insert_start = time.perf_counter()
                            delta_id_to_mid, delta_mid_to_id = insert_noisy_variants(
                                delta_persons, variants_per_identity, noise_fraction,
                                seed, col_name, identity_offset=offset,
                            )
                            col.flush()
                            insert_time_s = time.perf_counter() - insert_start

                            canonical_persons.extend(delta_persons)
                            identity_to_milvus_ids.extend(delta_id_to_mid)
                            milvus_id_to_identity.update(delta_mid_to_id)

                            print(f"[SCALE] Inserted & flushed {delta_n * variants_per_identity} "
                                  f"new records (marginal)  insert_time={insert_time_s:.2f}s")
                        else:
                            insert_time_s = 0.0
                            print("[SCALE] No new identities needed for this N — reusing prior data.")

                        # --- Evaluate recall@K over the full accumulated collection ---
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
                if original_mode is None:
                    os.environ.pop("MILVUS_VECTOR_MODE", None)
                else:
                    os.environ["MILVUS_VECTOR_MODE"] = original_mode
