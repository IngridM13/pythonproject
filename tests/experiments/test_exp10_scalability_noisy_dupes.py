"""
Experiment 10 — Scalability with Noisy Duplicates for HDC-based data reconciliation.

Evaluates how recall scales when the database contains both clean original records
and noisy duplicate variants. This models a realistic deduplication scenario where
the BD holds a mix of golden records and corrupted copies of those same records.

Setup per N
-----------
1. Generate N clean canonical records and insert them into Milvus.
2. Select n_source = int(N × noise_ratio) // duplicates_per_original originals
   without replacement; for each, generate duplicates_per_original independent
   noisy variants using inject_noise(). Total noisy records ≈ int(N × noise_ratio).
3. Insert the noisy duplicates into the same collection (they act as both
   distractors and queries).
4. For each noisy duplicate, query top-(K+1), exclude self, check whether the
   original record it was derived from appears in the top-K results.
5. Compute Recall@1, Recall@K, Recall@D (D=duplicates_per_original), MRR, Hit@1.
6. Drop the collection between each (mode, N) pair.

Run
---
    pytest tests/experiments/test_exp10_scalability_noisy_dupes.py -v -s

Environment variables
---------------------
    EXP10_COLLECTION_SIZES        Comma-separated N values (default: from settings)
    EXP10_NOISE_RATIO             Fraction of noisy duplicates relative to N (default: 0.20)
    EXP10_NOISE_LEVEL             Corruption level passed to inject_noise (default: 0.30)
    EXP10_TOP_K                   K for Recall@K (default: 5)
    EXP10_SEED                    RNG seed (default: 42)
    EXP10_DUPLICATES_PER_ORIGINAL Noisy variants generated per source original (default: 3)
    EXP10_MODES                   Comma-separated modes to run, e.g. "float" or "binary,float" (default: binary,float)
"""

import csv
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
    EXP10_COLLECTION_SIZES,
    EXP10_DUPLICATES_PER_ORIGINAL,
    EXP10_NOISE_LEVEL,
    EXP10_NOISE_RATIO,
    EXP10_SEED,
    EXP10_TOP_K,
    HDC_DIM,
)
from database_utils.milvus_db_connection import ensure_people_collection
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from tests.experiments.experiment_utils import generate_canonical_persons
from tests.experiments.noise_injection import inject_noise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_noisy_duplicate(original_record: dict, noise_level: float, rng: random.Random) -> dict:
    """Return a corrupted copy of original_record at the given noise level."""
    return inject_noise(original_record, noise_level, rng)


def evaluate_recall(
    neighbours: list,
    original_id: int,
    top_k: int,
    top_d: int,
) -> dict:
    """
    Compute per-query recall metrics.

    Parameters
    ----------
    neighbours : list
        Top-K results from find_closest_match_db with self excluded.
    original_id : int
        Milvus ID of the original record the query was derived from.
    top_k : int
        K for Recall@K.
    top_d : int
        D for Recall@D (= duplicates_per_original).

    Returns
    -------
    dict with keys: recall_at_1, recall_at_k, recall_at_d, mrr, hit_at_1
    """
    hit_at_1    = bool(neighbours) and neighbours[0]["id"] == original_id
    recall_at_k = any(m["id"] == original_id for m in neighbours[:top_k])
    recall_at_d = any(m["id"] == original_id for m in neighbours[:top_d])
    mrr = 0.0
    for rank, m in enumerate(neighbours[:top_k], 1):
        if m["id"] == original_id:
            mrr = 1.0 / rank
            break
    return {
        "recall_at_1": float(hit_at_1),
        "recall_at_k": float(recall_at_k),
        "recall_at_d": float(recall_at_d),
        "mrr":         mrr,
        "hit_at_1":    float(hit_at_1),
    }


def _save_results(
    output_dir: Path,
    mode: str,
    config: dict,
    rows: list,
    top_d: int,
) -> tuple:
    """Save CSV and JSON reports. Returns (csv_path, json_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    recall_d_col = f"recall@{top_d}"
    csv_path = output_dir / f"exp10_{mode}_{timestamp}.csv"
    fieldnames = [
        "mode", "N", "noise_ratio", "noise_level", "duplicates_per_original",
        "recall@1", "recall@5", recall_d_col, "mrr", "hit@1",
        "avg_query_time_ms", "total_insert_time_s",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / f"exp10_{mode}_{timestamp}.json"
    json_path.write_text(json.dumps({
        "experiment": "Experiment 10 — Scalability with Noisy Duplicates",
        "timestamp":  timestamp,
        "mode":       mode,
        "config":     config,
        "results":    rows,
    }, indent=2))

    return csv_path, json_path


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestExp10ScalabilityNoisyDupes:

    def test_exp10_scalability_noisy_dupes(self):
        # --- Config from env ---
        raw_sizes = os.environ.get("EXP10_COLLECTION_SIZES", "")
        collection_sizes = (
            [int(x.strip()) for x in raw_sizes.split(",") if x.strip()]
            if raw_sizes.strip()
            else list(EXP10_COLLECTION_SIZES)
        )
        noise_ratio             = float(os.environ.get("EXP10_NOISE_RATIO",             EXP10_NOISE_RATIO))
        noise_level             = float(os.environ.get("EXP10_NOISE_LEVEL",             EXP10_NOISE_LEVEL))
        top_k                   = int(os.environ.get("EXP10_TOP_K",                     EXP10_TOP_K))
        seed                    = int(os.environ.get("EXP10_SEED",                      EXP10_SEED))
        duplicates_per_original = int(os.environ.get("EXP10_DUPLICATES_PER_ORIGINAL",  EXP10_DUPLICATES_PER_ORIGINAL))
        raw_modes               = os.environ.get("EXP10_MODES", "binary,float")
        modes                   = [m.strip() for m in raw_modes.split(",") if m.strip()]

        config = {
            "collection_sizes":        collection_sizes,
            "noise_ratio":             noise_ratio,
            "noise_level":             noise_level,
            "top_k":                   top_k,
            "duplicates_per_original": duplicates_per_original,
            "hdim":                    HDC_DIM,
            "seed":                    seed,
        }

        output_dir = Path(__file__).resolve().parents[2] / "test_results" / "exp10_scalability_noisy_dupes"
        recall_d_col = f"recall@{duplicates_per_original}"

        print(
            f"\n[EXP10] sizes={collection_sizes}  noise_ratio={noise_ratio}  "
            f"noise_level={noise_level}  top_k={top_k}  "
            f"duplicates_per_original={duplicates_per_original}  seed={seed}"
        )

        for mode in modes:
            original_mode = milvus_conn.VECTOR_MODE
            milvus_conn.VECTOR_MODE = mode

            try:
                mode_rows = []
                print(f"\n[EXP10] ── mode={mode} {'─' * 55}")

                for n in collection_sizes:
                    n_noisy   = int(n * noise_ratio)
                    n_sources = n_noisy // duplicates_per_original
                    col_name  = f"exp10_{uuid.uuid4().hex[:10]}"

                    print(
                        f"\n[EXP10] mode={mode}  N={n}  n_sources={n_sources}  "
                        f"duplicates_per_original={duplicates_per_original}  "
                        f"n_noisy={n_sources * duplicates_per_original}  "
                        f"total={n + n_sources * duplicates_per_original}  collection={col_name}"
                    )

                    col = ensure_people_collection(col_name, include_embedding=False)

                    try:
                        rng = random.Random(seed)

                        # --- 1. Generate and insert N canonical records ---
                        print(f"[EXP10] Generating {n} canonical records...")
                        canonical_persons    = generate_canonical_persons(n)
                        canonical_milvus_ids = []

                        insert_start = time.perf_counter()
                        for person in canonical_persons:
                            mid = store_person(person, collection_name=col_name)
                            canonical_milvus_ids.append(mid)

                        # --- 2. Generate and insert noisy duplicates ---
                        # Pick n_sources originals without replacement; each gets
                        # duplicates_per_original independent noisy variants.
                        source_indices = rng.sample(range(n), min(n_sources, n))
                        actual_noisy   = len(source_indices) * duplicates_per_original
                        print(
                            f"[EXP10] Inserting {actual_noisy} noisy duplicates "
                            f"({len(source_indices)} sources × {duplicates_per_original})..."
                        )

                        # noisy_entries: list of (noisy_milvus_id, original_milvus_id, noisy_person)
                        noisy_entries = []
                        for src_idx in source_indices:
                            for dup_i in range(duplicates_per_original):
                                noisy    = generate_noisy_duplicate(canonical_persons[src_idx], noise_level, rng)
                                noisy_id = store_person(noisy, collection_name=col_name)
                                noisy_entries.append((noisy_id, canonical_milvus_ids[src_idx], noisy))

                        col.flush()
                        total_insert_time_s = time.perf_counter() - insert_start

                        print(
                            f"[EXP10] Inserted & flushed {n + actual_noisy} records  "
                            f"insert_time={total_insert_time_s:.2f}s"
                        )

                        # --- 3. Evaluate recall over all noisy duplicates ---
                        recall1_sum    = 0.0
                        recallk_sum    = 0.0
                        recalld_sum    = 0.0
                        mrr_sum        = 0.0
                        hit1_sum       = 0.0
                        total_query_ms = 0.0
                        total_queries  = len(noisy_entries)

                        for q_idx, (noisy_mid, original_mid, noisy_person) in enumerate(noisy_entries):
                            q_start = time.perf_counter()
                            matches = find_closest_match_db(
                                noisy_person,
                                threshold=0.0,
                                limit=top_k + 1,
                                collection_name=col_name,
                            )
                            total_query_ms += (time.perf_counter() - q_start) * 1000

                            neighbours = [m for m in matches if m["id"] != noisy_mid][:top_k]
                            metrics    = evaluate_recall(neighbours, original_mid, top_k, duplicates_per_original)

                            recall1_sum += metrics["recall_at_1"]
                            recallk_sum += metrics["recall_at_k"]
                            recalld_sum += metrics["recall_at_d"]
                            mrr_sum     += metrics["mrr"]
                            hit1_sum    += metrics["hit_at_1"]

                            done = q_idx + 1
                            if done % 100 == 0 or done == total_queries:
                                print(
                                    f"[EXP10]   queried {done}/{total_queries}  "
                                    f"recall@{top_k}={recallk_sum / done:.3f}  "
                                    f"recall@{duplicates_per_original}={recalld_sum / done:.3f}"
                                )

                        total       = total_queries
                        recall_at_1 = recall1_sum / total if total > 0 else 0.0
                        recall_at_k = recallk_sum / total if total > 0 else 0.0
                        recall_at_d = recalld_sum / total if total > 0 else 0.0
                        mrr         = mrr_sum      / total if total > 0 else 0.0
                        hit_at_1    = hit1_sum     / total if total > 0 else 0.0
                        avg_q_ms    = total_query_ms / total if total > 0 else 0.0

                        print(
                            f"[EXP10] RESULT  mode={mode}  N={n}  "
                            f"recall@1={recall_at_1:.3f}  recall@{top_k}={recall_at_k:.3f}  "
                            f"recall@{duplicates_per_original}={recall_at_d:.3f}  "
                            f"MRR={mrr:.3f}  Hit@1={hit_at_1:.3f}  "
                            f"avg_query={avg_q_ms:.1f}ms  insert={total_insert_time_s:.2f}s"
                        )

                        mode_rows.append({
                            "mode":                    mode,
                            "N":                       n,
                            "noise_ratio":             noise_ratio,
                            "noise_level":             noise_level,
                            "duplicates_per_original": duplicates_per_original,
                            "recall@1":                round(recall_at_1, 6),
                            "recall@5":                round(recall_at_k, 6),
                            recall_d_col:              round(recall_at_d, 6),
                            "mrr":                     round(mrr, 6),
                            "hit@1":                   round(hit_at_1, 6),
                            "avg_query_time_ms":       round(avg_q_ms, 3),
                            "total_insert_time_s":     round(total_insert_time_s, 4),
                        })

                    finally:
                        try:
                            col.drop()
                        except Exception as drop_err:
                            print(f"[EXP10] Warning: could not drop {col_name}: {drop_err}")

                # --- Save results ---
                csv_path, json_path = _save_results(output_dir, mode, config, mode_rows, duplicates_per_original)
                print(f"\n[EXP10] CSV  → {csv_path}")
                print(f"[EXP10] JSON → {json_path}")

                # --- Summary table ---
                BAR_WIDTH = 25
                col_n   = 8
                col_tot = 12
                col_r1  = 10
                col_rk  = 10
                col_rd  = 10
                col_mrr = 8
                col_ins = 12
                col_q   = 11

                print(f"\nSummary — mode: {mode}  (D={duplicates_per_original})")
                print(
                    f"  {'N':>{col_n}}  {'Total Rec':>{col_tot}}  "
                    f"{'Recall@1':>{col_r1}}  {'Recall@' + str(top_k):>{col_rk}}  "
                    f"{'Recall@D':>{col_rd}}  {'MRR':>{col_mrr}}  "
                    f"{'Insert(s)':>{col_ins}}  {'Avg Q(ms)':>{col_q}}  Chart (Recall@{top_k})"
                )
                print(
                    f"  {'-'*col_n}  {'-'*col_tot}  "
                    f"{'-'*col_r1}  {'-'*col_rk}  "
                    f"{'-'*col_rd}  {'-'*col_mrr}  "
                    f"{'-'*col_ins}  {'-'*col_q}  {'-'*BAR_WIDTH}"
                )
                for row in mode_rows:
                    filled    = round(row["recall@5"] * BAR_WIDTH)
                    chart     = "#" * filled + "-" * (BAR_WIDTH - filled)
                    n_src     = int(row["N"] * noise_ratio) // duplicates_per_original
                    total_rec = row["N"] + n_src * duplicates_per_original
                    print(
                        f"  {row['N']:>{col_n}}  "
                        f"{total_rec:>{col_tot}}  "
                        f"{row['recall@1']:>{col_r1}.3f}  "
                        f"{row['recall@5']:>{col_rk}.3f}  "
                        f"{row[recall_d_col]:>{col_rd}.3f}  "
                        f"{row['mrr']:>{col_mrr}.3f}  "
                        f"{row['total_insert_time_s']:>{col_ins}.2f}  "
                        f"{row['avg_query_time_ms']:>{col_q}.1f}  "
                        f"{chart}"
                    )

            finally:
                milvus_conn.VECTOR_MODE = original_mode
