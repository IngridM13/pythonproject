"""
Experiment 12 — Recall@1 under noise across collection sizes.

Extends Experiment 1 by sweeping the collection size N and introducing
a query budget M: only M randomly sampled records (out of N) are used
as queries, making the experiment feasible at large N.

Design
------
For each vector_mode in {binary, float}:
  N_VALUES is processed in ascending order against a single collection that
  is grown incrementally, instead of being rebuilt from scratch at every N:
  For each N in sorted(N_VALUES):
    1. Generate only the delta of new canonical persons needed to go from
       the previous N to the current one, and insert them (first step
       generates N0 from scratch).
    2. Sample M records (M ≤ N) at random from the full inserted set so far
       — the same sample is reused across all noise levels for that
       (mode, N) so results are directly paired.
    3. For each noise level in NOISE_LEVELS:
       For each sampled record, inject noise at that level (without inserting
       it) and query top-1 against the collection.
       Compute Recall@1 = fraction of queries where top-1 == original.
    4. Move to the next N (same collection, same mode). Drop the collection
       once per mode, after finishing the whole N sweep.

Note: "insert_time_s" is the marginal insertion time for this step (growing
from N_prev to N), not the time to insert all N records from an empty
collection.

Output
------
Saves exp12_recall_n_sweep[_exhaustive]_<timestamp>.json with one row per
(mode, N, noise_level), to test_results/ (nprobe=8) or test_results_128/
(nprobe=128) depending on the active HDC_NPROBE mode.

Run
---
    pytest tests/experiments/test_exp12_recall_n_sweep.py -v -s

Environment variables
---------------------
    EXP12_N_VALUES      Comma-separated collection sizes (default: from settings)
    EXP12_M_QUERIES     Number of query records sampled per N (default: from settings)
    EXP12_NOISE_LEVELS  Comma-separated noise levels for query corruption (default: from settings, 0.20 and 0.30)
    EXP12_SEED          RNG seed (default: from settings)
    EXP12_MODES         Comma-separated modes to run (default: binary,float)
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
    EXP12_NOISE_LEVELS,
    EXP12_SEED,
    HDC_DIM,
)
from database_utils.milvus_db_connection import ensure_people_collection, get_nprobe
from encoding_methods.encoding_and_search_milvus import search_for_eval, store_people_batch
from tests.experiments.experiment_utils import generate_canonical_persons, resolve_results_dir_and_suffix
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
        m_queries = int(os.environ.get("EXP12_M_QUERIES", EXP12_M_QUERIES))
        raw_noise = os.environ.get("EXP12_NOISE_LEVELS", "")
        noise_levels = (
            [float(x.strip()) for x in raw_noise.split(",") if x.strip()]
            if raw_noise.strip()
            else list(EXP12_NOISE_LEVELS)
        )
        seed      = int(os.environ.get("EXP12_SEED", EXP12_SEED))
        raw_modes = os.environ.get("EXP12_MODES", "binary,float")
        modes     = [m.strip() for m in raw_modes.split(",") if m.strip()]

        n_values_sorted = sorted(n_values)
        if n_values_sorted != n_values:
            print(f"[EXP12] Note: n_values reordered ascending for incremental "
                  f"reuse: {n_values} -> {n_values_sorted}")
        n_values = n_values_sorted

        print(
            f"\n[EXP12] n_values={n_values}  m_queries={m_queries}  "
            f"noise_levels={noise_levels}  seed={seed}  modes={modes}"
        )

        all_results = []

        for mode in modes:
            original_mode = os.environ.get("MILVUS_VECTOR_MODE", "binary")
            os.environ["MILVUS_VECTOR_MODE"] = mode

            try:
                print(f"\n[EXP12] ── mode={mode} {'─' * 55}")

                # One persistent collection per mode — grown incrementally
                # across the N sweep instead of recreated from scratch each time.
                col_name = f"exp12_{uuid.uuid4().hex[:10]}"
                col      = ensure_people_collection(col_name)

                canonical_persons: list = []
                milvus_ids: list        = []

                try:
                    for n in n_values:
                        m = min(m_queries, n)

                        print(f"\n[EXP12] mode={mode}  N={n}  M={m}  collection={col_name}")

                        # --- 1. Generate and insert only the delta of new canonical records ---
                        delta_n = n - len(canonical_persons)
                        if delta_n > 0:
                            print(f"[EXP12] Generating {delta_n} new canonical records "
                                  f"({len(canonical_persons)} -> {n})...")
                            delta_persons = generate_canonical_persons(delta_n)

                            t_insert_start = time.perf_counter()
                            delta_mids = store_people_batch(delta_persons, collection_name=col_name)
                            canonical_persons.extend(delta_persons)
                            milvus_ids.extend(delta_mids)

                            col.flush()
                            total_insert_time_s = time.perf_counter() - t_insert_start
                            print(
                                f"[EXP12] Inserted & flushed {delta_n} new records "
                                f"(marginal) in {total_insert_time_s:.2f}s"
                            )
                        else:
                            total_insert_time_s = 0.0
                            print("[EXP12] No new records needed for this N — reusing prior data.")

                        # --- 2. Sample M records to use as queries (shared across noise levels) ---
                        rng = random.Random(seed)
                        sample_indices = rng.sample(range(n), m)

                        # --- 3. For each noise level, query with noisy versions (not inserted) ---
                        for noise_level in noise_levels:
                            noise_rng = random.Random(seed)
                            hits = 0
                            total_query_ms = 0.0

                            for q_idx, idx in enumerate(sample_indices):
                                noisy = inject_noise(canonical_persons[idx], noise_level, noise_rng)

                                t0 = time.perf_counter()
                                matches = search_for_eval(
                                    noisy,
                                    1,
                                    collection_name=col_name,
                                )
                                total_query_ms += (time.perf_counter() - t0) * 1000

                                if matches and matches[0]["id"] == milvus_ids[idx]:
                                    hits += 1

                                done = q_idx + 1
                                if done % 50 == 0 or done == m:
                                    print(
                                        f"[EXP12]   noise={noise_level}  queried {done}/{m}  "
                                        f"recall@1={hits / done:.3f}"
                                    )

                            recall_at_1  = hits / m if m > 0 else 0.0
                            avg_query_ms = total_query_ms / m if m > 0 else 0.0

                            print(
                                f"[EXP12] RESULT  mode={mode}  N={n}  M={m}  noise={noise_level}  "
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
                os.environ["MILVUS_VECTOR_MODE"] = original_mode

        # --- Save JSON ---
        nprobe = get_nprobe()
        output_dir, suffix = resolve_results_dir_and_suffix(nprobe)
        output_dir.mkdir(exist_ok=True)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"exp12_recall_n_sweep{suffix}_{timestamp}.json"

        report = {
            "experiment": "Experiment 12 — Recall@1 under noise across collection sizes",
            "timestamp": timestamp,
            "nprobe": nprobe,
            "config": {
                "n_values":     n_values,
                "m_queries":    m_queries,
                "noise_levels": noise_levels,
                "hdim":         HDC_DIM,
                "seed":         seed,
                "modes":        modes,
            },
            "results": all_results,
        }
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\n[EXP12] Results saved to {output_path.name}")

        # --- Summary table ---
        print(f"\n{'mode':<8}  {'N':>8}  {'noise':>6}  {'M':>6}  {'Recall@1':>10}  {'Avg Q (ms)':>12}  {'Insert (s)':>12}")
        print("-" * 75)
        for r in all_results:
            print(
                f"{r['mode']:<8}  {r['n']:>8}  {r['noise_level']:>6}  {r['m_queries']:>6}  "
                f"{r['recall_at_1']:>10.3f}  {r['avg_query_time_ms']:>12.1f}  "
                f"{r['insert_time_s']:>12.2f}"
            )

        assert any(r["recall_at_1"] > 0 for r in all_results), (
            "All recall@1 values are 0 — something is wrong with encoding or search."
        )
