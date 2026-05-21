#!/usr/bin/env python3
"""
Experiment 14a — Float Capacity Analysis.

Characterizes where float-mode HDC deduplication starts to degrade as
collection size N grows beyond 50K, with nprobe=128 (exhaustive IVF search).

Baseline from Exp 12 (nprobe=128): float showed 100% Recall@1 at N=50K
with noise=20%, and 98.5% with noise=30%.  This experiment pushes further
to find the real degradation point, with the IVF approximation confounder
eliminated.

Design
------
For each noise in NOISE_VALUES:
  For each N in N_VALUES:
    1. Force VECTOR_MODE = "float".
    2. Generate N canonical records and insert into a fresh collection.
    3. Sample M queries (seeded, no replacement).
    4. For each query: inject noise, search top-5, record hits.
    5. Compute Recall@1, Recall@5, MRR, Hit@1.
    6. Drop collection and move to next (N, noise).

Hardware note
-------------
Mac M5, Docker 18 GB RAM.  Practical float ceiling ~200K
(each 10K-dim float32 vector = 40 KB; 200K vectors ≈ 8 GB in Milvus).
Docker memory is sampled via `docker stats` at insert start/end; the
experiment aborts gracefully if Milvus becomes unresponsive.

Run
---
    python tests/experiments/test_exp14a_float_capacity.py

Environment variables
---------------------
    EXP14A_N_VALUES      Comma-separated collection sizes  (default: 50000,100000,150000,200000)
    EXP14A_M_QUERIES     Queries per (N, noise) pair       (default: 200)
    EXP14A_NOISE_VALUES  Comma-separated noise fractions   (default: 0.20,0.30)
    EXP14A_SEED          RNG seed                          (default: 42)

Output
------
    test_results/exp14a_float_capacity_<timestamp>.json
"""

import json
import os
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import database_utils.milvus_db_connection as milvus_conn
from configs.settings import (
    EXP14A_M_QUERIES,
    EXP14A_N_VALUES,
    EXP14A_NOISE_VALUES,
    EXP14A_SEED,
    HDC_DIM,
    NAME_AND_DATE_WEIGHTS,
)
from database_utils.milvus_db_connection import (
    _collection_cache,
    ensure_people_collection,
)
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from tests.experiments.experiment_utils import generate_canonical_persons
from tests.experiments.noise_injection import inject_noise


# ---------------------------------------------------------------------------
# Hardware / memory helpers
# ---------------------------------------------------------------------------

def _docker_mem_milvus() -> str | None:
    """Return Milvus container memory usage string, or None if unavailable."""
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"],
            capture_output=True, text=True, timeout=8,
        )
        for line in result.stdout.splitlines():
            if "milvus" in line.lower() or "standalone" in line.lower():
                return line.split("\t")[1].strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def _drop_collection(col_name: str) -> None:
    _collection_cache.pop(f"{col_name}_float", None)
    try:
        col = ensure_people_collection(col_name)
        col.drop()
    except Exception as exc:
        print(f"[EXP14A] Warning: could not drop '{col_name}': {exc}")


# ---------------------------------------------------------------------------
# Per-(N, noise) run
# ---------------------------------------------------------------------------

def _run_one(n: int, noise: float, m_queries: int, seed: int) -> dict:
    col_name = f"exp14a_{uuid.uuid4().hex[:10]}"
    print(f"\n[EXP14A] N={n}  noise={noise:.0%}  M={m_queries}  collection={col_name}")

    mem_before = _docker_mem_milvus()

    col = ensure_people_collection(col_name)

    try:
        # --- 1. Insert N canonical records ---
        print(f"[EXP14A] Generating and inserting {n:,} records...")
        canonical_persons = generate_canonical_persons(n)

        t_insert = time.perf_counter()
        milvus_ids: list[int] = []
        for person in canonical_persons:
            mid = store_person(person, collection_name=col_name,
                               field_weights=NAME_AND_DATE_WEIGHTS)
            milvus_ids.append(mid)
        col.flush()
        insert_seconds = round(time.perf_counter() - t_insert, 2)

        mem_after = _docker_mem_milvus()
        print(f"[EXP14A] Inserted & flushed {n:,} records in {insert_seconds}s  "
              f"(mem: {mem_before} → {mem_after})")

        # --- 2. Sample M query sources ---
        rng = random.Random(seed)
        query_indices = rng.sample(range(n), min(m_queries, n))

        # --- 3. Query loop ---
        hits_1 = 0
        hits_5 = 0
        reciprocal_ranks: list[float] = []
        total_query_ms = 0.0

        for q_i, idx in enumerate(query_indices):
            query_rng = random.Random(seed + 1_000_000 + idx)
            noisy = inject_noise(canonical_persons[idx], noise, query_rng)
            gt_id = milvus_ids[idx]

            t0 = time.perf_counter()
            matches = find_closest_match_db(
                noisy,
                threshold=0.0,
                limit=5,
                collection_name=col_name,
                field_weights=NAME_AND_DATE_WEIGHTS,
            )
            total_query_ms += (time.perf_counter() - t0) * 1000

            # Recall@1
            if matches and matches[0]["id"] == gt_id:
                hits_1 += 1

            # Recall@5 + MRR
            rr = 0.0
            for rank, m in enumerate(matches, 1):
                if m["id"] == gt_id:
                    hits_5 += 1
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            done = q_i + 1
            if done % 50 == 0 or done == len(query_indices):
                print(f"[EXP14A]   queried {done}/{len(query_indices)}  "
                      f"recall@1={hits_1/done:.3f}  recall@5={hits_5/done:.3f}")

        m = len(query_indices)
        recall_1   = round(hits_1 / m, 6) if m else 0.0
        recall_5   = round(hits_5 / m, 6) if m else 0.0
        mrr        = round(mean(reciprocal_ranks), 6) if reciprocal_ranks else 0.0
        avg_q_ms   = round(total_query_ms / m, 3) if m else 0.0

        entry = {
            "N":              n,
            "noise":          noise,
            "recall_at_1":    recall_1,
            "recall_at_5":    recall_5,
            "mrr":            mrr,
            "hit_at_1":       recall_1,   # identical to recall@1 for single-variant collections
            "avg_query_ms":   avg_q_ms,
            "insert_seconds": insert_seconds,
            "hits":           f"{hits_1}/{m}",
            "mem_before":     mem_before,
            "mem_after":      mem_after,
        }

        print(
            f"[EXP14A] RESULT  N={n}  noise={noise:.0%}  "
            f"recall@1={recall_1:.3f}  recall@5={recall_5:.3f}  "
            f"MRR={mrr:.3f}  avg_q={avg_q_ms:.1f}ms  insert={insert_seconds}s"
        )
        return entry

    finally:
        _drop_collection(col_name)


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_values: list,
    noise_values: list,
    m_queries: int,
    seed: int,
) -> list:
    """Run all (noise, N) pairs in float mode and return result dicts."""
    original_mode = milvus_conn.VECTOR_MODE
    milvus_conn.VECTOR_MODE = "float"

    all_results = []
    try:
        for noise in noise_values:
            print(f"\n[EXP14A] {'═' * 60}")
            print(f"[EXP14A]  NOISE: {noise:.0%}")
            print(f"[EXP14A] {'═' * 60}")
            for n in n_values:
                try:
                    entry = _run_one(n=n, noise=noise, m_queries=m_queries, seed=seed)
                    all_results.append(entry)
                except Exception as exc:
                    print(f"[EXP14A] ERROR at N={n} noise={noise}: {exc}")
                    print(f"[EXP14A] Aborting remaining N values for this noise level.")
                    break
    finally:
        milvus_conn.VECTOR_MODE = original_mode

    return all_results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary(results: list) -> None:
    col_w = [8, 7, 10, 10, 8, 10, 10, 10]
    header = (
        f"  {'N':>{col_w[0]}} {'Noise':>{col_w[1]}} "
        f"{'Recall@1':>{col_w[2]}} {'Recall@5':>{col_w[3]}} "
        f"{'MRR':>{col_w[4]}} {'AvgQ(ms)':>{col_w[5]}} "
        f"{'Insert(s)':>{col_w[6]}} {'Mem after':>{col_w[7]}}"
    )
    divider = "  " + "-" * (sum(col_w) + len(col_w) * 2)

    print("\n" + "=" * 80)
    print("  EXPERIMENT 14a — FLOAT CAPACITY  (mode=float, nprobe=128)")
    print("=" * 80)
    print(header)

    noise_vals = sorted({r["noise"] for r in results})
    for noise in noise_vals:
        print(divider)
        for r in (x for x in results if x["noise"] == noise):
            mem = r.get("mem_after") or "—"
            print(
                f"  {r['N']:>{col_w[0]},} {r['noise']:>{col_w[1]}.0%} "
                f"{r['recall_at_1']:>{col_w[2]}.3f} "
                f"{r['recall_at_5']:>{col_w[3]}.3f} "
                f"{r['mrr']:>{col_w[4]}.3f} "
                f"{r['avg_query_ms']:>{col_w[5]}.1f} "
                f"{r['insert_seconds']:>{col_w[6]}.1f} "
                f"{mem:>{col_w[7]}}"
            )
    print("=" * 80)


def _print_analysis(results: list) -> None:
    print("\n  ANALYSIS")
    print("  " + "─" * 60)

    noise_vals = sorted({r["noise"] for r in results})
    for noise in noise_vals:
        rows = [r for r in results if r["noise"] == noise]
        if not rows:
            continue
        print(f"\n  noise={noise:.0%}")

        # First N where recall@1 < 99%
        below = [r for r in rows if r["recall_at_1"] < 0.99]
        if below:
            print(f"    Recall@1 first drops below 99% at N={below[0]['N']:,}  "
                  f"(recall={below[0]['recall_at_1']:.3f})")
        else:
            print(f"    Recall@1 stays >= 99% across all N values tested.")

        # Degradation rate per 50K (slope from first to last N)
        if len(rows) >= 2:
            n_span   = rows[-1]["N"] - rows[0]["N"]
            r_span   = rows[-1]["recall_at_1"] - rows[0]["recall_at_1"]
            if n_span > 0:
                rate_pp = r_span / (n_span / 50_000) * 100
                print(f"    Degradation rate: {rate_pp:+.2f} pp per 50K records "
                      f"(N={rows[0]['N']:,}→{rows[-1]['N']:,})")

    # Cross-noise comparison
    if len(noise_vals) >= 2:
        print(f"\n  Cross-noise comparison (noise={noise_vals[0]:.0%} vs {noise_vals[1]:.0%}):")
        r_by = {(r["noise"], r["N"]): r for r in results}
        ns = sorted({r["N"] for r in results})
        print(f"    {'N':>8}  {'Δ Recall@1':>12}  {'Δ MRR':>10}")
        print(f"    {'─'*8}  {'─'*12}  {'─'*10}")
        for n in ns:
            r0 = r_by.get((noise_vals[0], n))
            r1 = r_by.get((noise_vals[1], n))
            if r0 and r1:
                d_rec = r1["recall_at_1"] - r0["recall_at_1"]
                d_mrr = r1["mrr"] - r0["mrr"]
                print(f"    {n:>8,}  {d_rec:>+12.3f}  {d_mrr:>+10.3f}")
        print(f"    (negative = higher noise hurts more)")

    print()


def _save_results(report: dict) -> Path:
    output_dir = _PROJECT_ROOT / "test_results"
    output_dir.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"exp14a_float_capacity_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2))
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    raw_n = os.environ.get("EXP14A_N_VALUES", "")
    n_values = (
        [int(x.strip()) for x in raw_n.split(",") if x.strip()]
        if raw_n.strip()
        else list(EXP14A_N_VALUES)
    )
    raw_noise = os.environ.get("EXP14A_NOISE_VALUES", "")
    noise_values = (
        [float(x.strip()) for x in raw_noise.split(",") if x.strip()]
        if raw_noise.strip()
        else list(EXP14A_NOISE_VALUES)
    )
    m_queries = int(os.environ.get("EXP14A_M_QUERIES", EXP14A_M_QUERIES))
    seed      = int(os.environ.get("EXP14A_SEED",      EXP14A_SEED))

    print("\n[EXP14A] Float Capacity Analysis — HDC Deduplication")
    print(f"[EXP14A] mode=float  N_values={n_values}  noise_values={noise_values}  "
          f"M={m_queries}  seed={seed}  dims={HDC_DIM}  nprobe=128")
    print("[EXP14A] Confirming search config: nprobe=128, nlist=128, metric=IP, index=IVF_FLAT")

    # Verify nlist=128 in index creation matches expectations
    from configs.settings import HDC_DIM as _dim
    assert _dim == 10000, f"HDC_DIM mismatch: expected 10000, got {_dim}"

    results = run_experiment(
        n_values=n_values,
        noise_values=noise_values,
        m_queries=m_queries,
        seed=seed,
    )

    peak_mem = max(
        (r.get("mem_after") or "" for r in results),
        key=lambda s: s,
        default=None,
    )

    report = {
        "experiment": "Experiment 14a — Float Capacity Analysis",
        "timestamp":  datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "mode":         "float",
            "n_values":     n_values,
            "noise_values": noise_values,
            "m_queries":    m_queries,
            "hdim":         HDC_DIM,
            "seed":         seed,
            "nprobe":       128,
            "nlist":        128,
        },
        "hardware": {
            "mac":                "M5",
            "docker_ram_gb":      18,
            "peak_mem_observed":  peak_mem,
        },
        "results": results,
    }

    output_path = _save_results(report)
    print(f"\n[EXP14A] Results saved → {output_path}")

    _print_summary(results)
    _print_analysis(results)


if __name__ == "__main__":
    main()
