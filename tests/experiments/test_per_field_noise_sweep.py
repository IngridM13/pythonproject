"""
Per-Field Noise Sweep experiment for HDC-based data reconciliation.

Sweeps a range of noise levels for each individual field and measures how
retrieval quality (Recall@K, MRR, Hit@1) degrades as noise in that field
increases, while all other fields are kept clean.

Setup
-----
1. Generate N synthetic canonical identities.
2. For each identity, produce V noisy variants using inject_noise().
3. Insert all N×V records into Milvus once (reused for all combinations).
4. For each (field, noise_level) combination:
   - noise_level=0: query with clean canonical person (shared baseline).
   - noise_level>0: corrupt only the target field, leave all others clean.
5. Print one results table per field to stdout.
6. Save a JSON report to test_results/per_field_sweep_<mode>_<timestamp>.json.

Run
---
    pytest tests/experiments/test_per_field_noise_sweep.py -v -s

Environment variables
---------------------
    PER_FIELD_SWEEP_FIELDS        Comma-separated field names (default: PER_FIELD_SWEEP_FIELDS from settings)
    PER_FIELD_SWEEP_NOISE_LEVELS  Comma-separated ints 0-100 (default: PER_FIELD_SWEEP_NOISE_LEVELS from settings)
    PER_FIELD_SWEEP_N             Number of canonical identities (default: PER_FIELD_SWEEP_N from settings)
    PER_FIELD_SWEEP_V             Noisy variants per identity (default: PER_FIELD_SWEEP_V from settings)
    PER_FIELD_SWEEP_K             K for Recall@K (default: PER_FIELD_SWEEP_K from settings)
    PER_FIELD_SWEEP_SEED          RNG seed (default: PER_FIELD_SWEEP_SEED from settings)
"""

import os
import random
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import (
    HDC_DIM,
    PER_FIELD_SWEEP_FIELDS,
    PER_FIELD_SWEEP_NOISE_LEVELS,
    PER_FIELD_SWEEP_N,
    PER_FIELD_SWEEP_V,
    PER_FIELD_SWEEP_K,
    PER_FIELD_SWEEP_SEED,
)
from tests.experiments.experiment_utils import (
    _inject_single_field_noise,
    _compute_metrics,
    generate_canonical_persons,
    insert_noisy_variants,
    save_report,
)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("with_vector_mode")
class TestPerFieldNoiseSweep:

    def test_per_field_noise_sweep(self, with_vector_mode, test_collection):
        # --- Config from env ---
        raw_fields = os.environ.get("PER_FIELD_SWEEP_FIELDS", None)
        if raw_fields is not None:
            fields = [f.strip() for f in raw_fields.split(",") if f.strip()]
        else:
            fields = list(PER_FIELD_SWEEP_FIELDS)

        raw_levels = os.environ.get("PER_FIELD_SWEEP_NOISE_LEVELS", None)
        if raw_levels is not None:
            noise_levels = [int(x.strip()) for x in raw_levels.split(",") if x.strip()]
        else:
            noise_levels = list(PER_FIELD_SWEEP_NOISE_LEVELS)

        n_identities          = int(os.environ.get("PER_FIELD_SWEEP_N", PER_FIELD_SWEEP_N))
        variants_per_identity = int(os.environ.get("PER_FIELD_SWEEP_V", PER_FIELD_SWEEP_V))
        top_k                 = int(os.environ.get("PER_FIELD_SWEEP_K", PER_FIELD_SWEEP_K))
        seed                  = int(os.environ.get("PER_FIELD_SWEEP_SEED", PER_FIELD_SWEEP_SEED))

        mode = with_vector_mode
        total_records = n_identities * variants_per_identity

        print(
            f"\n[PER_FIELD_SWEEP] mode={mode}, n_identities={n_identities}, "
            f"variants_per_identity={variants_per_identity}, "
            f"noise_levels={noise_levels}, top_k={top_k}, seed={seed}"
        )
        print(f"[PER_FIELD_SWEEP] Fields under sweep: {fields}")
        print(f"[PER_FIELD_SWEEP] Total records to insert: {total_records}")

        # --- 1. Generate canonical identities ---
        canonical_persons = generate_canonical_persons(n_identities)

        # --- 2. Generate variants and insert into Milvus ONCE ---
        identity_to_milvus_ids, milvus_id_to_identity = insert_noisy_variants(
            canonical_persons, variants_per_identity, 0.30, seed, test_collection
        )

        # --- 3. Flush once — reused for all (field, noise_level) combinations ---
        from database_utils.milvus_db_connection import ensure_people_collection
        col = ensure_people_collection(test_collection)
        col.flush()
        print(f"[PER_FIELD_SWEEP] Inserted & flushed {total_records} records.")

        # --- 4. Compute clean baseline (noise_level=0) once, shared across all fields ---
        print("[PER_FIELD_SWEEP] Computing baseline (clean canonical queries)...")

        def _baseline_query(canonical, identity_idx):
            return canonical

        base_recall, base_mrr, base_hit1 = _compute_metrics(
            canonical_persons,
            identity_to_milvus_ids,
            milvus_id_to_identity,
            top_k,
            test_collection,
            _baseline_query,
        )

        print(
            f"[PER_FIELD_SWEEP] BASELINE  recall@{top_k}={base_recall:.3f}  "
            f"MRR={base_mrr:.3f}  Hit@1={base_hit1:.3f}"
        )

        # --- 5. Sweep each field across all noise levels ---
        RED    = "\033[91m"
        YELLOW = "\033[93m"
        GREEN  = "\033[92m"
        RESET  = "\033[0m"

        BAR_WIDTH = 40

        def _recall_color(recall):
            if recall >= 0.9:
                return GREEN
            elif recall >= 0.6:
                return YELLOW
            return RED

        def _recall_bar(recall):
            filled = round(recall * BAR_WIDTH)
            return "#" * filled + "-" * (BAR_WIDTH - filled)

        sweep_results = {}  # field -> list of {noise_level, recall_at_k, mrr, hit_at_1}

        for field in fields:
            print()
            print("=" * 85)
            print(f" Per-Field Noise Sweep  —  field: {field}  (mode: {mode})")
            print("=" * 85)
            print(
                f"  {'Noise%':>7}  {'Recall@' + str(top_k):>9}  {'MRR':>7}  {'Hit@1':>7}  Chart"
            )
            print(
                f"  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*BAR_WIDTH}"
            )

            field_rows = []

            for noise_level in noise_levels:
                if noise_level == 0:
                    # Clean baseline — no corruption
                    f_recall, f_mrr, f_hit1 = base_recall, base_mrr, base_hit1
                else:
                    def _field_query(canonical, identity_idx, _field=field, _seed=seed, _nl=noise_level):
                        rng = random.Random(_seed + hash(_field) + identity_idx + _nl)
                        return _inject_single_field_noise(canonical, _field, rng)

                    f_recall, f_mrr, f_hit1 = _compute_metrics(
                        canonical_persons,
                        identity_to_milvus_ids,
                        milvus_id_to_identity,
                        top_k,
                        test_collection,
                        _field_query,
                    )

                field_rows.append({
                    "noise_level": noise_level,
                    "recall_at_k": round(f_recall, 6),
                    "mrr":         round(f_mrr, 6),
                    "hit_at_1":    round(f_hit1, 6),
                })

                c = _recall_color(f_recall)
                print(
                    f"  {noise_level:>6}%  "
                    f"{c}{f_recall:>9.1%}{RESET}  "
                    f"{f_mrr:>7.3f}  "
                    f"{c}{f_hit1:>7.1%}{RESET}  "
                    f"{c}{_recall_bar(f_recall)}{RESET}"
                )

            print("=" * 85)
            sweep_results[field] = field_rows

        # --- 6. Save JSON report ---
        output_path = save_report("per_field_sweep", mode, {
            "mode": mode,
            "config": {
                "fields":                fields,
                "noise_levels":          noise_levels,
                "n_identities":          n_identities,
                "variants_per_identity": variants_per_identity,
                "top_k":                 top_k,
                "hdim":                  HDC_DIM,
                "seed":                  seed,
            },
            "results": sweep_results,
        })
        print(f"\n[PER_FIELD_SWEEP] Results saved to {output_path.name}")
