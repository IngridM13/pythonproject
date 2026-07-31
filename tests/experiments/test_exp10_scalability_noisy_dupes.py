"""
Experiment 10 — Scalability with Noisy Duplicates for HDC-based data reconciliation.

Evaluates how recall scales when the database contains both clean original records
and noisy duplicate variants. This models a realistic deduplication scenario where
the BD holds a mix of golden records and corrupted copies of those same records.

Setup per N (incremental across the sweep)
--------------------------------------------
Sizes are processed in ascending order and the collection is grown
incrementally instead of being rebuilt from scratch at every N: the clean
records and noisy duplicates generated for a smaller N are kept and reused
as the base for the next, larger N.

1. Generate only the delta of new clean canonical records needed to go from
   the previous N to the current one, and insert them into Milvus (first
   step generates N0 from scratch).
2. Compute the target n_sources = int(N × noise_ratio) // duplicates_per_original
   for this N. Select only the *additional* sources needed beyond what was
   already used at the previous N (without replacement, drawn from records
   not yet used as a source); for each, generate duplicates_per_original
   independent noisy variants using inject_noise(). Total noisy records
   accumulate to ≈ int(N × noise_ratio).
3. Insert the new noisy duplicates into the same collection (they act as
   both distractors and queries), alongside all previously inserted ones.
4. For every noisy duplicate accumulated so far (old + new), query
   top-(K+1), exclude self, check whether the original record it was
   derived from appears in the top-K results. This is re-run in full at
   each N because the search space changes as the collection grows.
5. Compute Recall@1, Recall@K, Recall@D (D=duplicates_per_original), MRR, Hit@1.
6. Drop the collection once per mode, after finishing the whole sweep
   (instead of once per (mode, N) pair).

Note: with this incremental design, the per-row "total_insert_time_s" is the
*marginal* insertion time for this step (growing from N_prev to N), not the
time to insert all N + noisy records from an empty collection.

Run
---
    pytest tests/experiments/test_exp10_scalability_noisy_dupes.py -v -s

Environment variables
---------------------
    EXP10_COLLECTION_SIZES        Comma-separated N values (default: from settings)
    EXP10_NOISE_RATIO             Fraction of noisy duplicates relative to N (default: 0.20)
    EXP10_NOISE_LEVEL             Corruption level passed to inject_noise (default: 0.30)
    EXP10_TOP_K                   K for Recall@K (default: 3)
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
from database_utils.milvus_db_connection import ensure_people_collection, get_nprobe
from encoding_methods.encoding_and_search_milvus import search_for_eval, store_people_batch
from tests.experiments.experiment_utils import generate_canonical_persons, resolve_results_dir_and_suffix
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
    mode: str,
    config: dict,
    rows: list,
    top_k: int,
    top_d: int,
) -> tuple:
    """Save CSV and JSON reports. Returns (csv_path, json_path)."""
    nprobe = get_nprobe()
    output_dir, suffix = resolve_results_dir_and_suffix(nprobe)
    output_dir = output_dir / "exp10_scalability_noisy_dupes"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    recall_d_col = f"recall@{top_d}"
    recall_k_col = f"recall@{top_k}"
    csv_path = output_dir / f"exp10_{mode}{suffix}_{timestamp}.csv"
    fieldnames = [
        "mode", "N", "noise_ratio", "noise_level", "duplicates_per_original",
        "recall@1", recall_k_col, recall_d_col, "mrr", "hit@1",
        "avg_query_time_ms", "total_insert_time_s",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / f"exp10_{mode}{suffix}_{timestamp}.json"
    json_path.write_text(json.dumps({
        "experiment": "Experiment 10 — Scalability with Noisy Duplicates",
        "timestamp":  timestamp,
        "mode":       mode,
        "nprobe":     nprobe,
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
        collection_sizes_sorted = sorted(collection_sizes)
        if collection_sizes_sorted != collection_sizes:
            print(f"[EXP10] Note: collection_sizes reordered ascending for "
                  f"incremental reuse: {collection_sizes} -> {collection_sizes_sorted}")
        collection_sizes = collection_sizes_sorted
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

        recall_d_col = f"recall@{duplicates_per_original}"

        print(
            f"\n[EXP10] sizes={collection_sizes}  noise_ratio={noise_ratio}  "
            f"noise_level={noise_level}  top_k={top_k}  "
            f"duplicates_per_original={duplicates_per_original}  seed={seed}"
        )

        for mode in modes:
            original_mode = os.environ.get("MILVUS_VECTOR_MODE", "binary")
            os.environ["MILVUS_VECTOR_MODE"] = mode

            try:
                mode_rows = []
                print(f"\n[EXP10] ── mode={mode} {'─' * 55}")

                # One persistent collection per mode, grown incrementally
                # across the N sweep instead of recreated from scratch each time.
                col_name = f"exp10_{uuid.uuid4().hex[:10]}"
                col      = ensure_people_collection(col_name)

                # Single RNG instance for the whole mode — its stream advances
                # across N steps so each step draws genuinely new samples,
                # instead of resampling from scratch with a reset seed.
                rng = random.Random(seed)

                canonical_persons: list    = []
                canonical_milvus_ids: list = []
                noisy_entries: list        = []   # (noisy_mid, original_mid, noisy_person)
                used_source_indices: set   = set()

                try:
                    for n in collection_sizes:
                        n_noisy_target   = int(n * noise_ratio)
                        n_sources_target = n_noisy_target // duplicates_per_original

                        print(
                            f"\n[EXP10] mode={mode}  N={n}  n_sources_target={n_sources_target}  "
                            f"duplicates_per_original={duplicates_per_original}  "
                            f"n_noisy_target={n_sources_target * duplicates_per_original}  "
                            f"total_target={n + n_sources_target * duplicates_per_original}  "
                            f"collection={col_name}"
                        )

                        insert_start = time.perf_counter()

                        # --- 1. Generate and insert only the delta of new canonical records ---
                        delta_n = n - len(canonical_persons)
                        if delta_n > 0:
                            print(f"[EXP10] Generating {delta_n} new canonical records "
                                  f"({len(canonical_persons)} -> {n})...")
                            delta_persons = generate_canonical_persons(delta_n)
                            delta_mids = store_people_batch(delta_persons, collection_name=col_name)
                            canonical_persons.extend(delta_persons)
                            canonical_milvus_ids.extend(delta_mids)

                        # --- 2. Generate and insert only the delta of new noisy duplicates ---
                        # Pick additional sources (without replacement, excluding
                        # sources already used at a smaller N) so previously
                        # created duplicates are kept untouched.
                        delta_sources_needed = max(0, n_sources_target - len(used_source_indices))
                        available_indices    = [i for i in range(n) if i not in used_source_indices]
                        new_source_indices   = rng.sample(
                            available_indices, min(delta_sources_needed, len(available_indices))
                        )

                        print(
                            f"[EXP10] Inserting {len(new_source_indices) * duplicates_per_original} "
                            f"new noisy duplicates ({len(new_source_indices)} new sources × "
                            f"{duplicates_per_original})..."
                        )

                        new_noisy_persons: list = []
                        new_noisy_sources: list = []
                        for src_idx in new_source_indices:
                            for dup_i in range(duplicates_per_original):
                                noisy = generate_noisy_duplicate(canonical_persons[src_idx], noise_level, rng)
                                new_noisy_persons.append(noisy)
                                new_noisy_sources.append(src_idx)
                            used_source_indices.add(src_idx)

                        new_noisy_ids = store_people_batch(new_noisy_persons, collection_name=col_name)
                        for noisy_id, src_idx, noisy in zip(new_noisy_ids, new_noisy_sources, new_noisy_persons):
                            noisy_entries.append((noisy_id, canonical_milvus_ids[src_idx], noisy))

                        col.flush()
                        total_insert_time_s = time.perf_counter() - insert_start

                        actual_noisy = len(noisy_entries)
                        print(
                            f"[EXP10] Inserted & flushed this step's delta  "
                            f"(cumulative total={len(canonical_persons) + actual_noisy} records)  "
                            f"marginal_insert_time={total_insert_time_s:.2f}s"
                        )

                        # --- 3. Evaluate recall over ALL noisy duplicates accumulated so far ---
                        # Must be recomputed in full each step: the search space
                        # (collection contents) has grown since the last N.
                        recall1_sum    = 0.0
                        recallk_sum    = 0.0
                        recalld_sum    = 0.0
                        mrr_sum        = 0.0
                        hit1_sum       = 0.0
                        total_query_ms = 0.0
                        total_queries  = len(noisy_entries)

                        for q_idx, (noisy_mid, original_mid, noisy_person) in enumerate(noisy_entries):
                            q_start = time.perf_counter()
                            matches = search_for_eval(
                                noisy_person,
                                top_k + 1,
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
                            f"avg_query={avg_q_ms:.1f}ms  marginal_insert={total_insert_time_s:.2f}s"
                        )

                        recall_k_col = f"recall@{top_k}"
                        mode_rows.append({
                            "mode":                    mode,
                            "N":                       n,
                            "noise_ratio":             noise_ratio,
                            "noise_level":             noise_level,
                            "duplicates_per_original": duplicates_per_original,
                            "recall@1":                round(recall_at_1, 6),
                            recall_k_col:              round(recall_at_k, 6),
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
                csv_path, json_path = _save_results(mode, config, mode_rows, top_k, duplicates_per_original)
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
                recall_k_col = f"recall@{top_k}"
                for row in mode_rows:
                    filled    = round(row[recall_k_col] * BAR_WIDTH)
                    chart     = "#" * filled + "-" * (BAR_WIDTH - filled)
                    n_src     = int(row["N"] * noise_ratio) // duplicates_per_original
                    total_rec = row["N"] + n_src * duplicates_per_original
                    print(
                        f"  {row['N']:>{col_n}}  "
                        f"{total_rec:>{col_tot}}  "
                        f"{row['recall@1']:>{col_r1}.3f}  "
                        f"{row[recall_k_col]:>{col_rk}.3f}  "
                        f"{row[recall_d_col]:>{col_rd}.3f}  "
                        f"{row['mrr']:>{col_mrr}.3f}  "
                        f"{row['total_insert_time_s']:>{col_ins}.2f}  "
                        f"{row['avg_query_time_ms']:>{col_q}.1f}  "
                        f"{chart}"
                    )

            finally:
                os.environ["MILVUS_VECTOR_MODE"] = original_mode
