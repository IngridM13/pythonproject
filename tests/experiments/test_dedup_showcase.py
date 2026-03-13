"""
Deduplication showcase experiment for HDC-based data reconciliation.

Creates N synthetic identities with V noisy variants each, picks a random
sample of records, searches for each one, and displays the top-K most similar
records retrieved — including the full record content — so results can be
evaluated visually.

Run:
    pytest tests/experiments/test_dedup_showcase.py -v -s

Environment variables:
    SHOWCASE_N_IDENTITIES          Number of canonical identities (default: 100)
    SHOWCASE_VARIANTS_PER_IDENTITY Noisy variants per identity (default: 3)
    SHOWCASE_NOISE_FRACTION        Noise fraction for variants (default: 0.3)
    SHOWCASE_N_SAMPLES             Number of records to query (default: 5)
    SHOWCASE_TOP_K                 Top results to retrieve per query (default: 2)
    SHOWCASE_SEED                  RNG seed (default: DEFAULT_SEED from settings)
"""

import json
import os
import random
import sys
from datetime import date, datetime
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
# Display helpers
# ---------------------------------------------------------------------------

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


def _print_record(d: dict, indent: int = 4, similarity: float = None):
    pad = " " * indent
    sim_str = f"  (similarity: {similarity:.4f})" if similarity is not None else ""
    print(f"{pad}Name     : {d['name']} {d['lastname']}{sim_str}")
    print(f"{pad}DOB      : {d['dob']}")
    print(f"{pad}Gender   : {d['gender']}  |  Race: {d['race']}  |  Marital: {d['marital_status']}")
    print(f"{pad}Phone    : {d['mobile_number']}")
    if d["akas"]:
        print(f"{pad}AKAs     : {', '.join(d['akas'])}")
    if d["addresses"]:
        first = d["addresses"][0]
        extra = f" (+{len(d['addresses']) - 1} more)" if len(d["addresses"]) > 1 else ""
        print(f"{pad}Address  : {first}{extra}")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("with_vector_mode")
class TestDedupShowcase:

    def test_dedup_showcase(self, with_vector_mode, test_collection):
        # --- Config from env ---
        n_identities          = int(os.environ.get("SHOWCASE_N_IDENTITIES", 100))
        variants_per_identity = int(os.environ.get("SHOWCASE_VARIANTS_PER_IDENTITY", 3))
        noise_fraction        = float(os.environ.get("SHOWCASE_NOISE_FRACTION", 0.3))
        n_samples             = int(os.environ.get("SHOWCASE_N_SAMPLES", 5))
        top_k                 = int(os.environ.get("SHOWCASE_TOP_K", 2))
        seed                  = int(os.environ.get("SHOWCASE_SEED", DEFAULT_SEED))

        mode = with_vector_mode
        total_records = n_identities * variants_per_identity

        print(
            f"\n[SHOWCASE] mode={mode}, n_identities={n_identities}, "
            f"variants_per_identity={variants_per_identity}, "
            f"noise_fraction={noise_fraction}, n_samples={n_samples}, top_k={top_k}"
        )

        # --- 1. Generate canonical identities ---
        df = generate_data_chunk(n_identities)
        canonical_persons = []
        for _, row in df.iterrows():
            raw = dataframe_row_to_person_dict(row)
            canonical_persons.append(normalize_person_data(raw))

        # --- 2. Insert all variants ---
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

        # --- 3. Flush ---
        from database_utils.milvus_db_connection import ensure_people_collection
        col = ensure_people_collection(test_collection)
        col.flush()
        print(f"[SHOWCASE] Inserted & flushed {total_records} records.")

        # --- 4. Sample records to query ---
        sample_rng = random.Random(seed + 9999)
        all_ids = list(milvus_id_to_person.keys())
        sample_ids = sample_rng.sample(all_ids, min(n_samples, len(all_ids)))

        # --- 5. Query each sampled record and display results ---
        queries_report = []

        for i, query_milvus_id in enumerate(sample_ids, 1):
            query_person   = milvus_id_to_person[query_milvus_id]
            identity_idx   = milvus_id_to_identity[query_milvus_id]
            query_display  = _serialize_person(query_person)

            matches = find_closest_match_db(
                query_person,
                threshold=0.0,
                limit=top_k + 1,
                collection_name=test_collection,
            )
            neighbours = [m for m in matches if m["id"] != query_milvus_id][:top_k]

            print(f"\n{'=' * 60}")
            print(f" Query {i}/{len(sample_ids)}  [identity={identity_idx}]")
            print(f"{'=' * 60}")
            print("  SEARCHED FOR:")
            _print_record(query_display)

            results_report = []
            for rank, m in enumerate(neighbours, 1):
                result_identity = milvus_id_to_identity.get(m["id"])
                is_match = result_identity == identity_idx
                label = "[MATCH]" if is_match else "[DIFF] "

                result_person  = milvus_id_to_person.get(m["id"], {})
                result_display = _serialize_person(result_person)
                result_display["similarity"] = m.get("similarity")

                print(f"\n  Result {rank}  {label}  identity={result_identity}")
                _print_record(result_display, similarity=m.get("similarity"))

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

        print(f"\n{'=' * 60}\n")

        # --- 6. Save JSON report ---
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "test_results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dedup_showcase_{mode}_{timestamp}.json"
        output_path = output_dir / filename

        report = {
            "config": {
                "vector_mode":           mode,
                "n_identities":          n_identities,
                "variants_per_identity": variants_per_identity,
                "total_records":         total_records,
                "noise_fraction":        noise_fraction,
                "n_samples":             n_samples,
                "top_k":                 top_k,
                "hdim":                  HDC_DIM,
                "seed":                  seed,
            },
            "queries": queries_report,
        }
        output_path.write_text(json.dumps(report, indent=2))
        print(f"[SHOWCASE] Results saved to {filename}")
