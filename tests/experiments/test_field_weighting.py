"""
Experiment 3 — Field Weighting Ablation for HDC data reconciliation.

Measures how different field weighting / exclusion schemes affect deduplication
recall@K when multiple noisy variants of each identity are stored in Milvus.

Setup per variant
-----------------
1. Generate N synthetic canonical identities.
2. For each identity, produce V noisy variants using inject_noise().
3. Insert all N×V records into an ephemeral Milvus collection using the
   weighting scheme under test.
4. For each inserted record, query top-(K+1), exclude self, check whether any
   of the remaining top-K results belongs to the same identity.
5. Report recall@K = hits / (N×V).

Weighting variants
------------------
- baseline         : all weights = 1 (default behaviour)
- name_heavy       : name=3, lastname=3
- date_heavy       : dob=3
- name_and_date    : name=2, lastname=2, dob=2
- leave_out_name   : excluded_fields={"name", "lastname"}
- leave_out_dob    : excluded_fields={"dob"}
- leave_out_gender : excluded_fields={"gender"}
- leave_out_race   : excluded_fields={"race"}
- leave_out_marital: excluded_fields={"marital_status"}
- leave_out_phone  : excluded_fields={"mobile_number"}
- leave_out_address: excluded_fields={"address", "akas", "landlines"}

Run
---
    pytest tests/experiments/test_field_weighting.py -v -s
"""

import json
import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import database_utils.milvus_db_connection as milvus_conn
from configs.settings import DEFAULT_SEED, HDC_DIM
from database_utils.milvus_db_connection import ensure_people_collection
from dummy_data.generacion_base_de_datos import generate_data_chunk
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from utils.person_data_normalization import normalize_person_data
from tests.experiments.noise_injection import inject_noise
from tests.experiments.conftest import dataframe_row_to_person_dict


# ---------------------------------------------------------------------------
# Weighting variant definitions
# ---------------------------------------------------------------------------

WEIGHTING_VARIANTS = [
    {
        "name": "baseline",
        "field_weights": None,
        "excluded_fields": None,
    },
    {
        "name": "name_heavy",
        "field_weights": {"name": 3, "lastname": 3},
        "excluded_fields": None,
    },
    {
        "name": "date_heavy",
        "field_weights": {"dob": 3},
        "excluded_fields": None,
    },
    {
        "name": "name_and_date",
        "field_weights": {"name": 2, "lastname": 2, "dob": 2},
        "excluded_fields": None,
    },
    {
        "name": "leave_out_name",
        "field_weights": None,
        "excluded_fields": {"name", "lastname"},
    },
    {
        "name": "leave_out_dob",
        "field_weights": None,
        "excluded_fields": {"dob"},
    },
    {
        "name": "leave_out_gender",
        "field_weights": None,
        "excluded_fields": {"gender"},
    },
    {
        "name": "leave_out_race",
        "field_weights": None,
        "excluded_fields": {"race"},
    },
    {
        "name": "leave_out_marital",
        "field_weights": None,
        "excluded_fields": {"marital_status"},
    },
    {
        "name": "leave_out_phone",
        "field_weights": None,
        "excluded_fields": {"mobile_number"},
    },
    {
        "name": "leave_out_address",
        "field_weights": None,
        "excluded_fields": {"address", "akas", "landlines"},
    },
]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestFieldWeighting:

    def test_field_weighting_ablation(self):
        n_identities          = 200
        variants_per_identity = 3
        noise_fraction        = 0.3
        top_k                 = 5
        seed                  = DEFAULT_SEED
        total_records         = n_identities * variants_per_identity

        # --- Pre-generate canonical identities once (shared across variants) ---
        df = generate_data_chunk(n_identities)
        canonical_persons = []
        for _, row in df.iterrows():
            raw = dataframe_row_to_person_dict(row)
            canonical_persons.append(normalize_person_data(raw))

        config = {
            "n_identities":          n_identities,
            "variants_per_identity": variants_per_identity,
            "noise_fraction":        noise_fraction,
            "top_k":                 top_k,
            "hdim":                  HDC_DIM,
            "seed":                  seed,
        }

        project_root = Path(__file__).resolve().parents[2]
        output_dir   = project_root / "test_results"
        output_dir.mkdir(exist_ok=True)

        for mode in ["binary", "float"]:
            # Switch vector mode
            original_mode = milvus_conn.VECTOR_MODE
            milvus_conn.VECTOR_MODE = mode

            try:
                mode_results = []

                for variant_cfg in WEIGHTING_VARIANTS:
                    variant_name    = variant_cfg["name"]
                    field_weights   = variant_cfg["field_weights"]
                    excluded_fields = variant_cfg["excluded_fields"]

                    # Create a fresh ephemeral collection for this variant
                    col_name = f"fw_{uuid.uuid4().hex[:10]}"
                    col = ensure_people_collection(col_name, include_embedding=False)

                    print(
                        f"\n[FW] mode={mode}  variant={variant_name}  "
                        f"collection={col_name}"
                    )

                    try:
                        # --- Insert all noisy variants ---
                        identity_to_milvus_ids: list = [[] for _ in range(n_identities)]
                        milvus_id_to_identity:  dict = {}

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
                                    field_weights=field_weights,
                                    excluded_fields=excluded_fields,
                                )
                                identity_to_milvus_ids[identity_idx].append(milvus_id)
                                milvus_id_to_identity[milvus_id] = identity_idx

                        col.flush()
                        print(
                            f"[FW] Inserted & flushed {total_records} records "
                            f"for variant={variant_name}"
                        )

                        # --- Evaluate recall@K ---
                        hits  = 0
                        total = 0

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

                                hit = any(
                                    milvus_id_to_identity.get(m["id"]) == identity_idx
                                    for m in neighbours
                                )
                                if hit:
                                    hits += 1
                                total += 1

                        recall_at_k = hits / total if total > 0 else 0.0
                        print(
                            f"[FW] mode={mode}  variant={variant_name}  "
                            f"recall@{top_k}={recall_at_k:.3f}  "
                            f"({hits}/{total})"
                        )
                        mode_results.append(
                            {
                                "variant":     variant_name,
                                "recall_at_k": round(recall_at_k, 6),
                                "hits":        hits,
                                "total":       total,
                            }
                        )

                    finally:
                        # Drop ephemeral collection regardless of success/failure
                        try:
                            col.drop()
                        except Exception as drop_err:
                            print(
                                f"[FW] Warning: could not drop collection "
                                f"{col_name}: {drop_err}"
                            )

                # --- Save JSON report ---
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"field_weighting_{mode}_{timestamp}.json"
                report = {
                    "mode":    mode,
                    "config":  config,
                    "results": mode_results,
                }
                output_path.write_text(json.dumps(report, indent=2))
                print(f"\n[FW] Results saved to {output_path.name}")

                # --- Print summary table ---
                col_width = 22
                print(f"\nMode: {mode}")
                print(f"{'variant':<{col_width}}  {'recall@' + str(top_k)}")
                print(f"{'-' * col_width}  {'-' * 8}")
                for row in mode_results:
                    print(f"{row['variant']:<{col_width}}  {row['recall_at_k']:.3f}")

            finally:
                milvus_conn.VECTOR_MODE = original_mode
