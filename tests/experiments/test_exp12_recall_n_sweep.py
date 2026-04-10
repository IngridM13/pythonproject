"""
Experiment 12 — Recall@1 under noise across collection sizes.

Extends Experiment 1 by sweeping the collection size N and introducing
a query budget M: only M randomly sampled records (out of N) are used
as queries, making the experiment feasible at large N.

Design
------
For each vector_mode in {binary, float}:
  For each N in N_VALUES:
    1. Generate N canonical persons and insert them into a fresh collection.
    2. Sample M records (M ≤ N) at random from the inserted set.
    3. For each sampled record, inject noise (without inserting it) and
       query top-1 against the collection.
    4. Compute Recall@1 = fraction of queries where top-1 == original.
    5. Drop the collection and move to the next (mode, N) pair.

Output
------
Saves test_results/exp12_recall_n_sweep_<timestamp>.json with one row
per (mode, N).

Run
---
    pytest tests/experiments/test_exp12_recall_n_sweep.py -v -s

Environment variables
---------------------
    EXP12_N_VALUES     Comma-separated collection sizes (default: from settings)
    EXP12_M_QUERIES    Number of query records sampled per N (default: 200)
    EXP12_NOISE_LEVEL  Noise level for query corruption (default: 0.30)
    EXP12_SEED         RNG seed (default: 42)
    EXP12_MODES        Comma-separated modes to run (default: binary,float)
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
    EXP12_M_QUERIES,
    EXP12_N_VALUES,
    EXP12_NOISE_LEVEL,
    EXP12_SEED,
    HDC_DIM,
)
from database_utils.milvus_db_connection import ensure_people_collection
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from tests.experiments.experiment_utils import generate_canonical_persons
from tests.experiments.noise_injection import inject_noise


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestExp12RecallNSweep:

    def test_exp12_recall_n_sweep(self):
        # --- Config from env ---
        raw_n = os.environ.get("EXP12_N_VALUES", "")
        n_values = (
            [int(x.strip()) for x in raw_n.split(",") if x.strip()]
            if raw_n.strip()
            else list(EXP12_N_VALUES)
        )
        m_queries    = int(os.environ.get("EXP12_M_QUERIES",   EXP12_M_QUERIES))
        noise_level  = float(os.environ.get("EXP12_NOISE_LEVEL", EXP12_NOISE_LEVEL))
        seed         = int(os.environ.get("EXP12_SEED",          EXP12_SEED))
        raw_modes    = os.environ.get("EXP12_MODES", "binary,float")
        modes        = [m.strip() for m in raw_modes.split(",") if m.strip()]

        print(
            f"\n[EXP12] n_values={n_values}  m_queries={m_queries}  "
            f"noise_level={noise_level}  seed={seed}  modes={modes}"
        )

        all_results = []

        for mode in modes:
            original_mode = milvus_conn.VECTOR_MODE
            milvus_conn.VECTOR_MODE = mode

            try:
                print(f"\n[EXP12] ── mode={mode} {'─' * 55}")

                for n in n_values:
                    m = min(m_queries, n)
                    col_name = f"exp12_{uuid.uuid4().hex[:10]}"

                    print(
                        f"\n[EXP12] mode={mode}  N={n}  M={m}  "
                        f"noise={noise_level}  collection={col_name}"
                    )

                    ensure_people_collection(col_name)

                    try:
                        rng = random.Random(seed)

                        # --- 1. Generate and insert N canonical records ---
                        print(f"[EXP12] Generating {n} canonical records...")
                        canonical_persons = generate_canonical_persons(n)

                        t_insert_start = time.perf_counter()
                        milvus_ids = []
                        for person in canonical_persons:
                            mid = store_person(person, collection_name=col_name)
                            milvus_ids.append(mid)

                        col = ensure_people_collection(col_name)
                        col.flush()
                        total_insert_time_s = time.perf_counter() - t_insert_start
                        print(
                            f"[EXP12] Inserted & flushed {n} records "
                            f"in {total_insert_time_s:.2f}s"
                        )

                        # --- 2. Sample M records to use as queries ---
                        sample_indices = rng.sample(range(n), m)

                        # --- 3. Query with noisy versions (not inserted) ---
                        hits = 0
                        total_query_ms = 0.0

                        for q_idx, idx in enumerate(sample_indices):
                            noisy = inject_noise(canonical_persons[idx], noise_level, rng)

                            t0 = time.perf_counter()
                            matches = find_closest_match_db(
                                noisy,
                                threshold=0.0,
                                limit=1,
                                collection_name=col_name,
                            )
                            total_query_ms += (time.perf_counter() - t0) * 1000

                            if matches and matches[0]["id"] == milvus_ids[idx]:
                                hits += 1

                            done = q_idx + 1
                            if done % 50 == 0 or done == m:
                                print(
                                    f"[EXP12]   queried {done}/{m}  "
                                    f"recall@1={hits / done:.3f}"
                                )

                        recall_at_1    = hits / m if m > 0 else 0.0
                        avg_query_ms   = total_query_ms / m if m > 0 else 0.0

                        print(
                            f"[EXP12] RESULT  mode={mode}  N={n}  M={m}  "
                            f"recall@1={recall_at_1:.3f}  "
                            f"avg_query={avg_query_ms:.1f}ms  "
                            f"insert={total_insert_time_s:.2f}s"
                        )

                        all_results.append({
                            "mode":               mode,
                            "n":                  n,
                            "m_queries":          m,
                            "noise_level":        noise_level,
                            "recall_at_1":        round(recall_at_1, 6),
                            "hits":               hits,
                            "avg_query_time_ms":  round(avg_query_ms, 3),
                            "insert_time_s":      round(total_insert_time_s, 4),
                        })

                    finally:
                        try:
                            col.drop()
                        except Exception as e:
                            print(f"[EXP12] Warning: could not drop {col_name}: {e}")

            finally:
                milvus_conn.VECTOR_MODE = original_mode

        # --- Save JSON ---
        project_root = Path(__file__).resolve().parents[2]
        output_dir   = project_root / "test_results"
        output_dir.mkdir(exist_ok=True)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"exp12_recall_n_sweep_{timestamp}.json"

        report = {
            "experiment": "Experiment 12 — Recall@1 under noise across collection sizes",
            "timestamp": timestamp,
            "config": {
                "n_values":    n_values,
                "m_queries":   m_queries,
                "noise_level": noise_level,
                "hdim":        HDC_DIM,
                "seed":        seed,
                "modes":       modes,
            },
            "results": all_results,
        }
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\n[EXP12] Results saved to {output_path.name}")

        # --- Summary table ---
        print(f"\n{'mode':<8}  {'N':>8}  {'M':>6}  {'Recall@1':>10}  {'Avg Q (ms)':>12}  {'Insert (s)':>12}")
        print("-" * 65)
        for r in all_results:
            print(
                f"{r['mode']:<8}  {r['n']:>8}  {r['m_queries']:>6}  "
                f"{r['recall_at_1']:>10.3f}  {r['avg_query_time_ms']:>12.1f}  "
                f"{r['insert_time_s']:>12.2f}"
            )

        assert any(r["recall_at_1"] > 0 for r in all_results), (
            "All recall@1 values are 0 — something is wrong with encoding or search."
        )
