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
7. Sample DEDUP_N_SAMPLES records and display their top-K search results in
   full detail for visual inspection.

Run
---
    pytest tests/experiments/test_dedup_recall.py -v -s

Environment variables
---------------------
    DEDUP_N_IDENTITIES          Number of canonical identities (default: 200)
    DEDUP_VARIANTS_PER_IDENTITY Noisy variants per identity (default: 3)
    DEDUP_NOISE_FRACTION        Noise fraction passed to inject_noise (default: 0.3)
    DEDUP_TOP_K                 K for recall@K (default: 5)
    DEDUP_N_SAMPLES             Records to display in showcase section (default: 5)
    DEDUP_SEED                  RNG seed (default: DEFAULT_SEED from settings)
"""

import os
import random
import sys
from datetime import date

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import DEFAULT_SEED, HDC_DIM
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from tests.experiments.noise_injection import inject_noise
from tests.experiments.experiment_utils import generate_canonical_persons, run_dedup_recall, save_report


def _serialize_person(p: dict) -> dict:
    """Flatten a normalized person dict into a JSON-serializable display dict."""
    dob = p.get("dob")
    attrs = p.get("attrs", {})
    return {
        "name":           p.get("name", ""),
        "lastname":       p.get("lastname", ""),
        "dob":            dob.isoformat() if isinstance(dob, date) else str(dob or ""),
        "gender":         p.get("gender", ""),
        "race":           p.get("race", ""),
        "marital_status": p.get("marital_status", ""),
        "mobile_number":  p.get("mobile_number", ""),
        "addresses":      attrs.get("address", []),
        "akas":           attrs.get("akas", []),
        "landlines":      attrs.get("landlines", []),
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("with_vector_mode")
class TestDedupRecall:

    def test_dedup_recall(self, with_vector_mode, test_collection):
        # --- Config from env ---
        n_identities          = int(os.environ.get("DEDUP_N_IDENTITIES", 1000))
        variants_per_identity = int(os.environ.get("DEDUP_VARIANTS_PER_IDENTITY", 3))
        noise_fraction        = float(os.environ.get("DEDUP_NOISE_FRACTION", 0.3))
        top_k                 = int(os.environ.get("DEDUP_TOP_K", 3))
        n_samples             = int(os.environ.get("DEDUP_N_SAMPLES", 5))
        seed                  = int(os.environ.get("DEDUP_SEED", DEFAULT_SEED))

        mode = with_vector_mode
        total_records = n_identities * variants_per_identity

        print(
            f"\n[DEDUP] mode={mode}, n_identities={n_identities}, "
            f"variants_per_identity={variants_per_identity}, "
            f"noise_fraction={noise_fraction}, top_k={top_k}, seed={seed}"
        )
        print(f"[DEDUP] Total records to insert: {total_records}")

        # --- 1. Generate canonical identities ---
        canonical_persons = generate_canonical_persons(n_identities)

        # --- 2. Generate variants and insert into Milvus ---
        identity_to_milvus_ids: list = [[] for _ in range(n_identities)]
        milvus_id_to_person:    dict = {}
        milvus_id_to_identity:  dict = {}

        for identity_idx, canonical in enumerate(canonical_persons):
            for variant_idx in range(variants_per_identity):
                variant_rng = random.Random(
                    seed + identity_idx * variants_per_identity + variant_idx
                )
                noisy = inject_noise(canonical, noise_fraction, variant_rng)
                milvus_id = store_person(noisy, collection_name=test_collection)
                identity_to_milvus_ids[identity_idx].append(milvus_id)
                milvus_id_to_person[milvus_id] = noisy
                milvus_id_to_identity[milvus_id] = identity_idx

        # --- 3. Flush to make all records searchable ---
        from database_utils.milvus_db_connection import ensure_people_collection
        col = ensure_people_collection(test_collection)
        col.flush()
        print(f"[DEDUP] Inserted & flushed {total_records} records.")

        # --- 4. Evaluate recall@K ---
        recall_at_k, _, _, hits, total = run_dedup_recall(
            canonical_persons,
            identity_to_milvus_ids,
            milvus_id_to_identity,
            variants_per_identity,
            noise_fraction,
            seed,
            top_k,
            test_collection,
        )
        print(
            f"[DEDUP] recall@{top_k} = {recall_at_k:.3f}  "
            f"({hits}/{total})  noise_fraction={noise_fraction}"
        )

        # --- 5. Showcase: sample records and display full search results ---
        sample_rng = random.Random(seed + 9999)
        all_ids = list(milvus_id_to_person.keys())
        sample_ids = sample_rng.sample(all_ids, min(n_samples, len(all_ids)))

        queries_report = []

        for i, query_milvus_id in enumerate(sample_ids, 1):
            query_person  = milvus_id_to_person[query_milvus_id]
            identity_idx  = milvus_id_to_identity[query_milvus_id]
            query_display = _serialize_person(query_person)

            matches = find_closest_match_db(
                query_person,
                threshold=0.0,
                limit=top_k + 1,
                collection_name=test_collection,
            )
            neighbours = [m for m in matches if m["id"] != query_milvus_id][:top_k]

            results_report = []
            for rank, m in enumerate(neighbours, 1):
                result_identity = milvus_id_to_identity.get(m["id"])
                is_match = result_identity == identity_idx

                result_person  = milvus_id_to_person.get(m["id"], {})
                result_display = _serialize_person(result_person)
                result_display["similarity"] = m.get("similarity")

                results_report.append({
                    "rank":         rank,
                    "milvus_id":    m["id"],
                    "identity_idx": result_identity,
                    "is_match":     is_match,
                    "record":       result_display,
                })

            queries_report.append({
                "query_num":       i,
                "query_milvus_id": query_milvus_id,
                "identity_idx":    identity_idx,
                "query_record":    query_display,
                "results":         results_report,
            })

        # --- 6. Save JSON report ---
        report = {
            "config": {
                "vector_mode":           mode,
                "n_identities":          n_identities,
                "variants_per_identity": variants_per_identity,
                "total_records":         total_records,
                "noise_fraction":        noise_fraction,
                "top_k":                 top_k,
                "n_samples":             n_samples,
                "hdim":                  HDC_DIM,
                "seed":                  seed,
            },
            "results": {
                f"recall_at_{top_k}": recall_at_k,
                "hits":               hits,
                "total":              total,
            },
            "queries": queries_report,
        }
        output_path = save_report("dedup_recall", mode, report)
        print(f"[DEDUP] Results saved to {output_path.name}")

        # --- 7. Assertion ---
        assert recall_at_k >= 0.5, (
            f"recall@{top_k} below 0.5 at noise_fraction={noise_fraction}: "
            f"{recall_at_k:.3f} ({hits}/{total})"
        )
