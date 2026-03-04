"""
Recall-under-noise experiment for HDC-based data reconciliation.

Measures recall@1 at increasing noise levels by storing N persons in Milvus,
then querying with noisy versions and checking whether the top-1 result is the
original record.

Run:
    pytest tests/experiments/test_recall_under_noise.py -v -s

Environment variables:
    RECALL_N_PEOPLE      Number of persons to insert (default: 100)
    RECALL_NOISE_LEVELS  Comma-separated floats (default: 0,0.1,...,1.0)
    RECALL_THRESHOLD     find_closest_match_db threshold (default: 0.0)
    RECALL_SEED          RNG seed (default: DEFAULT_SEED from settings)
"""

import json
import math
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_noise_levels(env_value: str) -> list:
    return [float(x.strip()) for x in env_value.split(",") if x.strip()]


def dataframe_row_to_person_dict(row) -> dict:
    """
    Convert a pandas Series (row from generate_data_chunk DataFrame) to the
    person dict format expected by normalize_person_data / store_person.

    generate_data_chunk columns:
        name, lastname, dob, addresses (JSON str), addresses_count,
        marital_status, akas (JSON str), akas_count, landlines (JSON str),
        landlines_count, mobile_number, gender, race
    """
    def _parse_json_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    return {
        "name": row.get("name", ""),
        "lastname": row.get("lastname", ""),
        "dob": row.get("dob", None),
        "marital_status": row.get("marital_status", ""),
        "mobile_number": row.get("mobile_number", ""),
        "gender": row.get("gender", ""),
        "race": row.get("race", ""),
        "attrs": {
            "address": _parse_json_list(row.get("addresses")),
            "akas": _parse_json_list(row.get("akas")),
            "landlines": _parse_json_list(row.get("landlines")),
        },
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("with_vector_mode")
class TestRecallUnderNoise:

    def test_recall_under_noise(self, with_vector_mode, test_collection):
        # --- Config from env ---
        n_people = int(os.environ.get("RECALL_N_PEOPLE", 1000))
        noise_levels_str = os.environ.get(
            "RECALL_NOISE_LEVELS",
            "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        )
        threshold = float(os.environ.get("RECALL_THRESHOLD", 0.0))
        seed = int(os.environ.get("RECALL_SEED", DEFAULT_SEED))
        noise_levels = _parse_noise_levels(noise_levels_str)

        mode = with_vector_mode
        print(f"\n[EXPERIMENT] mode={mode}, n_people={n_people}, seed={seed}")
        print(f"[EXPERIMENT] noise_levels={noise_levels}, threshold={threshold}")

        # --- 1. Generate & insert persons ---
        df = generate_data_chunk(n_people)

        id_to_person: dict = {}  # milvus_id -> normalized person dict

        for _, row in df.iterrows():
            raw = dataframe_row_to_person_dict(row)
            normalized = normalize_person_data(raw)
            milvus_id = store_person(normalized, collection_name=test_collection)
            id_to_person[milvus_id] = normalized

        # --- 2. Flush to make records searchable ---
        from database_utils.milvus_db_connection import ensure_people_collection
        col = ensure_people_collection(test_collection)
        col.flush()
        print(f"[EXPERIMENT] Inserted & flushed {len(id_to_person)} persons.")

        # --- 3. Evaluate recall@1 per noise level ---
        results = []

        for noise_level in noise_levels:
            rng = random.Random(seed)  # fresh RNG per noise level for reproducibility
            hits = 0

            for milvus_id, person in id_to_person.items():
                noisy = inject_noise(person, noise_level, rng)
                matches = find_closest_match_db(
                    noisy,
                    threshold=threshold,
                    limit=1,
                    collection_name=test_collection,
                )
                if matches and matches[0]["id"] == milvus_id:
                    hits += 1

            total = len(id_to_person)
            recall = hits / total if total > 0 else 0.0
            results.append({
                "noise_level": noise_level,
                "recall_at_1": recall,
                "hits": hits,
                "total": total,
            })
            print(
                f"[EXPERIMENT] noise={noise_level:.1f}  recall@1={recall:.3f}"
                f"  ({hits}/{total})"
            )

        # --- 4. Save JSON ---
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "test_results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recall_under_noise_{mode}_{timestamp}.json"
        output_path = output_dir / filename

        report = {
            "config": {
                "vector_mode": mode,
                "n_people": n_people,
                "hdim": HDC_DIM,
                "seed": seed,
                "noise_levels": noise_levels,
                "threshold": threshold,
            },
            "results": results,
        }
        output_path.write_text(json.dumps(report, indent=2))
        print(f"[EXPERIMENT] Results saved to {filename}")

        # --- 5. Assertions ---
        recall_by_level = {r["noise_level"]: r["recall_at_1"] for r in results}

        assert recall_by_level.get(0.0, 0.0) == 1.0, (
            f"Expected perfect recall at noise=0.0, got {recall_by_level.get(0.0)}"
        )

        mid_noise = 0.5
        if mid_noise in recall_by_level:
            assert recall_by_level[mid_noise] >= 0.6, (
                f"Recall at noise=0.5 below 0.6: {recall_by_level[mid_noise]:.3f}"
            )
