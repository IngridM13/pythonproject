"""
Experiment 5 — Ranking Metrics for HDC-based data reconciliation.

Measures ranking quality beyond simple recall by computing MRR, Precision@K,
and Hit@1 for the HDC vector search, modelling the real deduplication use-case:
given a record already stored in the database, how well-ranked are same-identity
neighbours in the top-K results?

Setup
-----
1. Generate N synthetic canonical identities.
2. For each identity, produce V noisy variants using inject_noise().
3. Insert all N×V records into Milvus, each obtaining a unique Milvus ID.
4. Build ground truth: identity_idx → [milvus_id_v0, milvus_id_v1, ...].
5. For each inserted record, query top-(K+1), exclude self, collect ordered
   top-K neighbours.
6. Compute Recall@K, MRR, Precision@K, and Hit@1.

Run
---
    pytest tests/experiments/test_ranking_metrics.py -v -s

Environment variables
---------------------
    RANKING_N_IDENTITIES          Number of canonical identities (default: 200)
    RANKING_VARIANTS_PER_IDENTITY Noisy variants per identity (default: 3)
    RANKING_NOISE_FRACTION        Noise fraction passed to inject_noise (default: 0.3)
    RANKING_TOP_K                 K for ranking metrics (default: 5)
    RANKING_SEED                  RNG seed (default: RANKING_SEED from settings)
"""

import os
import random
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import HDC_DIM, RANKING_N, RANKING_V, RANKING_NOISE, RANKING_K, RANKING_SEED
from encoding_methods.encoding_and_search_milvus import find_closest_match_db
from tests.experiments.noise_injection import inject_noise
from tests.experiments.experiment_utils import generate_canonical_persons, insert_noisy_variants, save_report


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("with_vector_mode")
class TestRankingMetrics:

    def test_ranking_metrics(self, with_vector_mode, test_collection):
        # --- Config from env ---
        n_identities          = int(os.environ.get("RANKING_N_IDENTITIES", RANKING_N))
        variants_per_identity = int(os.environ.get("RANKING_VARIANTS_PER_IDENTITY", RANKING_V))
        noise_fraction        = float(os.environ.get("RANKING_NOISE_FRACTION", RANKING_NOISE))
        top_k                 = int(os.environ.get("RANKING_TOP_K", RANKING_K))
        seed                  = int(os.environ.get("RANKING_SEED", RANKING_SEED))

        mode = with_vector_mode
        total_records = n_identities * variants_per_identity

        print(
            f"\n[RANKING] mode={mode}, n_identities={n_identities}, "
            f"variants_per_identity={variants_per_identity}, "
            f"noise_fraction={noise_fraction}, top_k={top_k}, seed={seed}"
        )
        print(f"[RANKING] Total records to insert: {total_records}")

        # --- 1. Generate canonical identities ---
        canonical_persons = generate_canonical_persons(n_identities)

        # --- 2. Generate variants and insert into Milvus ---
        identity_to_milvus_ids, milvus_id_to_identity = insert_noisy_variants(
            canonical_persons, variants_per_identity, noise_fraction, seed, test_collection
        )

        # --- 3. Flush to make all records searchable ---
        from database_utils.milvus_db_connection import ensure_people_collection
        col = ensure_people_collection(test_collection)
        col.flush()
        print(f"[RANKING] Inserted & flushed {total_records} records.")

        # --- 4. Evaluate ranking metrics ---
        recall_hits = 0
        hit_at_1_hits = 0
        reciprocal_ranks = []
        precisions = []
        total = 0

        for identity_idx, milvus_ids in enumerate(identity_to_milvus_ids):
            for variant_idx, query_milvus_id in enumerate(milvus_ids):
                query_rng = random.Random(
                    seed + identity_idx * variants_per_identity + variant_idx
                )
                query_person = inject_noise(
                    canonical_persons[identity_idx], noise_fraction, query_rng
                )

                matches = find_closest_match_db(
                    query_person,
                    threshold=0.0,
                    limit=top_k + 1,
                    collection_name=test_collection,
                )

                neighbours = [m for m in matches if m["id"] != query_milvus_id][:top_k]

                # Recall@K: at least one top-K neighbour shares the same identity
                hit = any(
                    milvus_id_to_identity.get(m["id"]) == identity_idx
                    for m in neighbours
                )
                if hit:
                    recall_hits += 1

                # Hit@1: rank-1 result is a true match
                if neighbours and milvus_id_to_identity.get(neighbours[0]["id"]) == identity_idx:
                    hit_at_1_hits += 1

                # MRR: reciprocal rank of first true match
                rr = 0.0
                for rank, m in enumerate(neighbours, 1):
                    if milvus_id_to_identity.get(m["id"]) == identity_idx:
                        rr = 1.0 / rank
                        break
                reciprocal_ranks.append(rr)

                # Precision@K: fraction of top-K results belonging to the same identity
                match_count = sum(
                    1 for m in neighbours
                    if milvus_id_to_identity.get(m["id"]) == identity_idx
                )
                precision = match_count / len(neighbours) if neighbours else 0.0
                precisions.append(precision)

                total += 1

        recall_at_k   = recall_hits / total if total > 0 else 0.0
        mrr           = sum(reciprocal_ranks) / total if total > 0 else 0.0
        precision_at_k = sum(precisions) / total if total > 0 else 0.0
        hit_at_1      = hit_at_1_hits / total if total > 0 else 0.0

        print(f"[RANKING] recall@{top_k}    = {recall_at_k:.3f}  ({recall_hits}/{total})")
        print(f"[RANKING] MRR           = {mrr:.3f}")
        print(f"[RANKING] precision@{top_k}  = {precision_at_k:.3f}")
        print(f"[RANKING] hit@1         = {hit_at_1:.3f}  ({hit_at_1_hits}/{total})")

        # --- 5. Save JSON report ---
        report = {
            "mode": mode,
            "config": {
                "n_identities":          n_identities,
                "variants_per_identity": variants_per_identity,
                "noise_fraction":        noise_fraction,
                "top_k":                 top_k,
                "hdim":                  HDC_DIM,
                "seed":                  seed,
            },
            "results": {
                "recall_at_k":   recall_at_k,
                "mrr":           mrr,
                "precision_at_k": precision_at_k,
                "hit_at_1":      hit_at_1,
                "total_queries": total,
            },
        }
        output_path = save_report("ranking_metrics", mode, report)
        print(f"[RANKING] Results saved to {output_path.name}")

        # --- 6. Assertion ---
        assert recall_at_k >= 0.5, (
            f"recall@{top_k} below 0.5 at noise_fraction={noise_fraction}: "
            f"{recall_at_k:.3f} ({recall_hits}/{total})"
        )
