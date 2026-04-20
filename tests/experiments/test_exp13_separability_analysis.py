#!/usr/bin/env python3
"""
Experiment 13 — Separability Analysis for HDC-based Deduplication.

Measures the separability between positive similarities (ground truth duplicates)
and negative similarities (best false positives) across collection sizes and modes.

For each (mode, N) pair:
  1. Insert N canonical records into a fresh Milvus collection.
  2. Select M=200 random query sources (seeded, no replacement).
  3. For each query: inject noise (configurable via EXP13_NOISE, default 0.30),
     capture sim_pos, sim_neg, gap, and rank of the ground truth.
  4. Aggregate gap distribution statistics and save to JSON.

Similarity conventions
----------------------
  Binary mode  : Milvus returns Hamming distance → sim = 1 - (dist / dims)
  Float mode   : Milvus returns inner product     → sim = ip / dims
Both are normalised to [0, 1] so gaps are directly comparable across modes.

NOTE: Milvus hard-caps topk at 16 384. For N=20 000 the search uses limit=16 384
      instead, so the ground truth is not guaranteed to appear; affected queries
      are counted and skipped. To lift the cap, set proxy.maxTopK in milvus.yaml
      and update MILVUS_MAX_TOPK in this file.

Run
---
    python experiments/exp13_separability_analysis.py

Environment variables
---------------------
    EXP13_N_VALUES   Comma-separated collection sizes  (default: 1000,5000,10000,20000)
    EXP13_M_QUERIES  Queries per (mode, N)             (default: 200)
    EXP13_NOISE      Noise fraction for inject_noise   (default: 0.30)
    EXP13_SEED       RNG seed                          (default: 42)
    EXP13_MODES      Comma-separated modes             (default: binary,float)

Output
------
    test_results/exp13_separability_{timestamp}.json
"""

import json
import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Path setup — make project root importable when run directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import database_utils.milvus_db_connection as milvus_conn
from configs.settings import (
    EXP13_M_QUERIES,
    EXP13_N_VALUES,
    EXP13_NOISE,
    EXP13_SEED,
    HDC_DIM,
    NAME_AND_DATE_WEIGHTS,
)
from database_utils.milvus_db_connection import (
    _collection_cache,
    ensure_people_collection,
    get_vector_mode,
)
from encoding_methods.encoding_and_search_milvus import (
    _encode_for_milvus,
    encode_person,
    store_person,
)
from tests.experiments.experiment_utils import generate_canonical_persons
from tests.experiments.noise_injection import inject_noise
from utils.person_data_normalization import normalize_person_data


# Milvus hard limit for topk; raise proxy.maxTopK in milvus.yaml to go higher.
MILVUS_MAX_TOPK = 16_384

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list, p: float) -> float:
    """p-th percentile of a pre-sorted list (linear interpolation, 0 <= p <= 100)."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    k = (n - 1) * p / 100.0
    lo = int(k)
    hi = lo + 1
    if hi >= n:
        return float(sorted_data[lo])
    frac = k - lo
    return float(sorted_data[lo]) * (1.0 - frac) + float(sorted_data[hi]) * frac


# ---------------------------------------------------------------------------
# Core search helper — bypasses find_closest_match_db's 3× multiplier
# ---------------------------------------------------------------------------

def _search_full(
    query_person: dict,
    limit: int,
    col_name: str,
    field_weights=NAME_AND_DATE_WEIGHTS,
) -> list:
    """
    Run a Milvus ANN search and return all hits as [{id, sim}] sorted by sim desc.

    The effective search limit is capped at MILVUS_MAX_TOPK (16 384), which is
    Milvus's hard limit. For N > 16 384 the ground truth is not guaranteed to
    appear; missing queries are counted and skipped in the caller.

    Similarity is normalised to [0, 1] for both modes:
      binary  : sim = 1 - hamming_dist / HDC_DIM
      float   : sim = inner_product / HDC_DIM
    """
    effective_limit = min(limit, MILVUS_MAX_TOPK)

    col = ensure_people_collection(col_name)
    normalized = normalize_person_data(query_person)
    qhv = encode_person(normalized, field_weights=field_weights)
    qpayload = _encode_for_milvus(qhv)

    mode = get_vector_mode()
    if mode == "binary":
        search_params = {"metric_type": "HAMMING", "params": {"nprobe": 128}}
        metric = "HAMMING"
    else:
        search_params = {"metric_type": "IP", "params": {"nprobe": 128}}
        metric = "IP"

    results = col.search(
        data=[qpayload],
        anns_field="hv",
        param=search_params,
        limit=effective_limit,
        output_fields=[],
        consistency_level="Strong",
    )
    hits = results[0] if results else []

    out = []
    for h in hits:
        if metric == "HAMMING":
            sim = 1.0 - (h.distance / float(HDC_DIM))
        else:
            sim = float(h.distance) / float(HDC_DIM)
        out.append({"id": int(h.id), "sim": sim})

    out.sort(key=lambda x: x["sim"], reverse=True)
    return out


def _drop_collection(col_name: str, mode: str) -> None:
    """Drop ephemeral collection and evict from connection cache."""
    _collection_cache.pop(f"{col_name}_{mode}", None)
    try:
        col = ensure_people_collection(col_name)
        col.drop()
    except Exception as exc:
        print(f"[EXP13] Warning: could not drop '{col_name}': {exc}")


# ---------------------------------------------------------------------------
# Per-(mode, N) experiment
# ---------------------------------------------------------------------------

def _run_one(
    mode: str,
    n: int,
    m_queries: int,
    noise: float,
    seed: int,
) -> dict:
    """
    Run separability analysis for one (mode, N) pair.

    Returns the result dict (without raw arrays truncated — full lists included).
    """
    col_name = f"exp13_{uuid.uuid4().hex[:10]}"
    print(f"\n[EXP13] mode={mode}  N={n}  M={m_queries}  collection={col_name}")

    col = ensure_people_collection(col_name)

    try:
        # 1. Generate and insert N canonical records
        print(f"[EXP13] Generating {n} canonical records...")
        canonical_persons = generate_canonical_persons(n)
        canonical_ids: list[int] = []
        for person in canonical_persons:
            mid = store_person(person, collection_name=col_name,
                               field_weights=NAME_AND_DATE_WEIGHTS)
            canonical_ids.append(mid)
        col.flush()
        print(f"[EXP13] Inserted and flushed {n} records.")

        # 2. Select M query sources (deterministic, without replacement)
        rng = random.Random(seed)
        query_indices = rng.sample(range(n), min(m_queries, n))

        # 3. Query loop
        gaps: list[float] = []
        sim_pos_list: list[float] = []
        sim_neg_list: list[float] = []
        rank_pos_list: list[int] = []
        missing = 0

        for q_i, idx in enumerate(query_indices):
            # Deterministic per-query seed: avoids coupling queries to each other
            query_rng = random.Random(seed + 1_000_000 + idx)
            noisy_query = inject_noise(canonical_persons[idx], noise, query_rng)
            ground_truth_id = canonical_ids[idx]

            results = _search_full(noisy_query, limit=n, col_name=col_name)

            # Locate ground truth in ranked results
            rank_pos = None
            sim_pos = None
            for rank, r in enumerate(results, 1):
                if r["id"] == ground_truth_id:
                    rank_pos = rank
                    sim_pos = r["sim"]
                    break

            if sim_pos is None:
                # Should never occur with limit == collection_size
                missing += 1
                print(f"[EXP13] WARNING: ground truth missing for query {q_i} "
                      f"(idx={idx}, gt_id={ground_truth_id})")
                continue

            # sim_neg: similarity of the best false positive
            #   rank_pos == 1 → falso positivo = rank 2
            #   rank_pos  > 1 → falso positivo = rank 1
            if rank_pos == 1:
                sim_neg = results[1]["sim"] if len(results) > 1 else sim_pos
            else:
                sim_neg = results[0]["sim"]

            gap = sim_pos - sim_neg
            gaps.append(gap)
            sim_pos_list.append(sim_pos)
            sim_neg_list.append(sim_neg)
            rank_pos_list.append(rank_pos)

            done = q_i + 1
            if done % 50 == 0 or done == len(query_indices):
                pct_pos = 100.0 * sum(g > 0 for g in gaps) / len(gaps)
                print(
                    f"[EXP13]   queried {done}/{len(query_indices)}  "
                    f"avg_gap={mean(gaps):.4f}  pct_positive={pct_pos:.1f}%"
                )

        if missing:
            print(f"[EXP13] {missing} queries had no ground truth in results — skipped.")

        # 4. Aggregate statistics
        sorted_gaps = sorted(gaps)
        n_total = len(gaps)
        n_positive = sum(g > 0 for g in gaps)
        n_collision = n_total - n_positive
        recall_at_1 = (
            sum(r == 1 for r in rank_pos_list) / len(rank_pos_list)
            if rank_pos_list else 0.0
        )

        entry = {
            "mode": mode,
            "N": n,
            "noise": noise,
            "gaps":    [round(g, 6) for g in gaps],
            "sim_pos": [round(s, 6) for s in sim_pos_list],
            "sim_neg": [round(s, 6) for s in sim_neg_list],
            "ranks":   rank_pos_list,
            "gap_mean": round(mean(gaps), 6) if gaps else 0.0,
            "gap_std":  round(stdev(gaps), 6) if len(gaps) > 1 else 0.0,
            "gap_min":  round(min(gaps), 6) if gaps else 0.0,
            "gap_max":  round(max(gaps), 6) if gaps else 0.0,
            "gap_p25":  round(_percentile(sorted_gaps, 25), 6),
            "gap_p75":  round(_percentile(sorted_gaps, 75), 6),
            "pct_gap_positive": round(100.0 * n_positive / n_total, 2) if n_total else 0.0,
            "pct_collision":    round(100.0 * n_collision / n_total, 2) if n_total else 0.0,
            "recall_at_1":      round(recall_at_1, 6),
        }

        print(
            f"[EXP13] RESULT  mode={mode}  N={n}  "
            f"gap_mean={entry['gap_mean']:.4f}  gap_std={entry['gap_std']:.4f}  "
            f"pct_positive={entry['pct_gap_positive']:.1f}%  "
            f"collision={entry['pct_collision']:.1f}%  "
            f"recall@1={entry['recall_at_1']:.3f}"
        )
        return entry

    finally:
        _drop_collection(col_name, mode)


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_values: list,
    m_queries: int,
    noise: float,
    seed: int,
    modes: list,
) -> list:
    """Run all (mode, N) pairs and return list of result dicts."""
    all_results = []

    for mode in modes:
        original_mode = milvus_conn.VECTOR_MODE
        milvus_conn.VECTOR_MODE = mode

        try:
            print(f"\n[EXP13] {'═' * 60}")
            print(f"[EXP13]  MODE: {mode.upper()}")
            print(f"[EXP13] {'═' * 60}")

            for n in n_values:
                entry = _run_one(mode=mode, n=n, m_queries=m_queries,
                                 noise=noise, seed=seed)
                all_results.append(entry)

        finally:
            milvus_conn.VECTOR_MODE = original_mode

    return all_results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary(all_results: list) -> None:
    """Print a compact summary table to stdout."""
    col_w = [8, 8, 10, 9, 14, 15, 10]
    header = (
        f"  {'Mode':<{col_w[0]}} {'N':>{col_w[1]}} "
        f"{'Gap_mean':>{col_w[2]}} {'Gap_std':>{col_w[3]}} "
        f"{'Pct_positive':>{col_w[4]}} {'Pct_collision':>{col_w[5]}} "
        f"{'Recall@1':>{col_w[6]}}"
    )
    divider = "  " + "-" * (sum(col_w) + len(col_w) * 2)

    print("\n" + "=" * 80)
    print("  EXPERIMENT 13 — SEPARABILITY SUMMARY")
    noise_vals = sorted({r.get("noise", "?") for r in all_results})
    noise_str  = ", ".join(f"{v:.0%}" if isinstance(v, float) else str(v) for v in noise_vals)
    print(f"  noise={noise_str}  |  sim normalised to [0,1]  |  gap = sim_pos - sim_neg")
    print("=" * 80)
    print(header)
    print(divider)
    for r in all_results:
        print(
            f"  {r['mode']:<{col_w[0]}} {r['N']:>{col_w[1]}} "
            f"{r['gap_mean']:>{col_w[2]}.4f} "
            f"{r['gap_std']:>{col_w[3]}.4f} "
            f"{r['pct_gap_positive']:>{col_w[4]-1}.1f}% "
            f"{r['pct_collision']:>{col_w[5]-1}.1f}% "
            f"{r['recall_at_1']:>{col_w[6]}.3f}"
        )
    print("=" * 80)


def _save_results(report: dict) -> Path:
    """Serialise report to test_results/exp13_separability_{timestamp}.json."""
    output_dir = _PROJECT_ROOT / "test_results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"exp13_separability_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2))
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Read config from environment (with settings.py defaults)
    raw_n = os.environ.get("EXP13_N_VALUES", "")
    n_values = (
        [int(x.strip()) for x in raw_n.split(",") if x.strip()]
        if raw_n.strip()
        else list(EXP13_N_VALUES)
    )
    m_queries = int(os.environ.get("EXP13_M_QUERIES", EXP13_M_QUERIES))
    noise     = float(os.environ.get("EXP13_NOISE",    EXP13_NOISE))
    seed      = int(os.environ.get("EXP13_SEED",       EXP13_SEED))
    raw_modes = os.environ.get("EXP13_MODES", "binary,float")
    modes     = [m.strip() for m in raw_modes.split(",") if m.strip()]

    print("\n[EXP13] Separability Analysis — HDC Deduplication")
    print(f"[EXP13] N_values={n_values}  M={m_queries}  noise={noise}  "
          f"seed={seed}  modes={modes}  dims={HDC_DIM}")

    all_results = run_experiment(
        n_values=n_values,
        m_queries=m_queries,
        noise=noise,
        seed=seed,
        modes=modes,
    )

    report = {
        "metadata": {
            "noise":    noise,
            "dims":     HDC_DIM,
            "M":        m_queries,
            "seed":     seed,
            "modes":    modes,
            "N_values": n_values,
        },
        "results": all_results,
    }

    output_path = _save_results(report)
    print(f"\n[EXP13] Results saved → {output_path}")

    _print_summary(all_results)


if __name__ == "__main__":
    main()
