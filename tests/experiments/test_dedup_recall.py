"""
Deduplication recall experiment for HDC-based data reconciliation.

Measures how effectively the system surfaces same-person candidates when
multiple noisy variants of each identity are stored in Milvus.

Unlike recall-under-noise (which checks whether the query retrieves the *exact
same record*), this experiment models the real deduplication use-case: given a
record already stored in the database, do its top-K neighbours include at least
one other variant of the *same identity*?

Setup
-----
1. Generate N synthetic canonical identities.
2. For each identity, produce V noisy variants using inject_noise().
3. Insert all N×V records into Milvus, each obtaining a unique Milvus ID.
4. Build ground truth: identity_idx → [milvus_id_v0, milvus_id_v1, ...].
5. For each inserted record, query top-(K+1), exclude self, check whether any
   of the remaining top-K results belongs to the same identity.
6. Report recall@K = hits / (N×V).

Run
---
    pytest tests/experiments/test_dedup_recall.py -v -s

Environment variables
---------------------
    DEDUP_N_IDENTITIES          Number of canonical identities (default: 200)
    DEDUP_VARIANTS_PER_IDENTITY Noisy variants per identity (default: 3)
                                Total records inserted = N × V.
    DEDUP_NOISE_FRACTION        Noise fraction passed to inject_noise (default: 0.3)
    DEDUP_TOP_K                 K for recall@K (default: 5)
    DEDUP_SEED                  RNG seed (default: DEFAULT_SEED from settings)
"""

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import DEFAULT_SEED, HDC_DIM
from dummy_data.generacion_base_de_datos import generate_data_chunk
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from utils.person_data_normalization import normalize_person_data
from tests.experiments.noise_injection import inject_noise
from tests.experiments.conftest import dataframe_row_to_person_dict


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("with_vector_mode")
class TestDedupRecall:

    def test_dedup_recall(self, with_vector_mode, test_collection):
        # --- Config from env ---
        n_identities = int(os.environ.get("DEDUP_N_IDENTITIES", 200))
        variants_per_identity = int(os.environ.get("DEDUP_VARIANTS_PER_IDENTITY", 3))
        noise_fraction = float(os.environ.get("DEDUP_NOISE_FRACTION", 0.3))
        top_k = int(os.environ.get("DEDUP_TOP_K", 5))
        seed = int(os.environ.get("DEDUP_SEED", DEFAULT_SEED))

        mode = with_vector_mode
        total_records = n_identities * variants_per_identity

        print(
            f"\n[DEDUP] mode={mode}, n_identities={n_identities}, "
            f"variants_per_identity={variants_per_identity}, "
            f"noise_fraction={noise_fraction}, top_k={top_k}, seed={seed}"
        )
        print(f"[DEDUP] Total records to insert: {total_records}")

        # --- 1. Generate canonical identities ---
        df = generate_data_chunk(n_identities)
        canonical_persons = []
        for _, row in df.iterrows():
            raw = dataframe_row_to_person_dict(row)
            canonical_persons.append(normalize_person_data(raw))

        # --- 2. Generate variants and insert into Milvus ---
        # identity_to_milvus_ids[i] = list of Milvus IDs for identity i
        identity_to_milvus_ids: list = [[] for _ in range(n_identities)]

        for identity_idx, canonical in enumerate(canonical_persons):
            for variant_idx in range(variants_per_identity):
                variant_rng = random.Random(
                    seed + identity_idx * variants_per_identity + variant_idx
                )
                noisy = inject_noise(canonical, noise_fraction, variant_rng)
                milvus_id = store_person(noisy, collection_name=test_collection)
                identity_to_milvus_ids[identity_idx].append(milvus_id)

        # --- 3. Flush to make all records searchable ---
        from database_utils.milvus_db_connection import ensure_people_collection
        col = ensure_people_collection(test_collection)
        col.flush()
        print(f"[DEDUP] Inserted & flushed {total_records} records.")

        # Build reverse lookup: milvus_id -> identity_idx
        milvus_id_to_identity: dict = {}
        for identity_idx, milvus_ids in enumerate(identity_to_milvus_ids):
            for mid in milvus_ids:
                milvus_id_to_identity[mid] = identity_idx

        # --- 4. Evaluate recall@K ---
        hits = 0
        total = 0

        for identity_idx, milvus_ids in enumerate(identity_to_milvus_ids):
            for variant_idx, query_milvus_id in enumerate(milvus_ids):
                # Re-generate the stored variant with the same seed to get the query vector
                query_rng = random.Random(
                    seed + identity_idx * variants_per_identity + variant_idx
                )
                query_person = inject_noise(
                    canonical_persons[identity_idx], noise_fraction, query_rng
                )

                # Request top-(K+1) to account for self appearing in results
                matches = find_closest_match_db(
                    query_person,
                    threshold=0.0,
                    limit=top_k + 1,
                    collection_name=test_collection,
                )

                # Exclude self, then take top-K
                neighbours = [m for m in matches if m["id"] != query_milvus_id][:top_k]

                # Hit if any neighbour belongs to the same identity
                hit = any(
                    milvus_id_to_identity.get(m["id"]) == identity_idx
                    for m in neighbours
                )
                if hit:
                    hits += 1
                total += 1

        recall_at_k = hits / total if total > 0 else 0.0
        print(
            f"[DEDUP] recall@{top_k} = {recall_at_k:.3f}  "
            f"({hits}/{total})  noise_fraction={noise_fraction}"
        )

        # --- 5. Save JSON report ---
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "test_results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dedup_recall_{mode}_{timestamp}.json"
        output_path = output_dir / filename

        report = {
            "config": {
                "vector_mode": mode,
                "n_identities": n_identities,
                "variants_per_identity": variants_per_identity,
                "total_records": total_records,
                "noise_fraction": noise_fraction,
                "top_k": top_k,
                "hdim": HDC_DIM,
                "seed": seed,
            },
            "results": {
                f"recall_at_{top_k}": recall_at_k,
                "hits": hits,
                "total": total,
            },
        }
        output_path.write_text(json.dumps(report, indent=2))
        print(f"[DEDUP] Results saved to {filename}")

        # --- 6. Assertion ---
        assert recall_at_k >= 0.5, (
            f"recall@{top_k} below 0.5 at noise_fraction={noise_fraction}: "
            f"{recall_at_k:.3f} ({hits}/{total})"
        )
