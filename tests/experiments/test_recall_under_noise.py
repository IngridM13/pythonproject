"""
Recall-under-noise experiment for HDC-based data reconciliation.

Measures recall@1 at increasing noise levels by storing N persons in Milvus,
then querying with noisy versions and checking whether the top-1 result is the
original record.

Run:
    pytest tests/experiments/test_recall_under_noise.py -v -s

Environment variables:
    RECALL_N_PEOPLE           Number of persons to insert (default: 1000)
    RECALL_NOISE_LEVELS       Comma-separated floats (default: 0,0.1,...,1.0)
    RECALL_THRESHOLD          find_closest_match_db threshold (default: 0.0)
    RECALL_SEED               RNG seed (default: DEFAULT_SEED from settings)
    RECALL_NEAR_DUPE_FRACTION Fraction of extra confuser records to insert as
                              near-duplicates (default: 0.0). For example, 0.2
                              adds floor(n_people * 0.2) near-duplicate records
                              to the collection. These records are NOT query
                              targets — they exist solely to make the search
                              problem harder and more realistic.
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
from tests.experiments.near_duplicates import generate_near_duplicates
from tests.experiments.conftest import dataframe_row_to_person_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_noise_levels(env_value: str) -> list:
    return [float(x.strip()) for x in env_value.split(",") if x.strip()]


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
        near_dupe_fraction = float(os.environ.get("RECALL_NEAR_DUPE_FRACTION", 0.0))

        mode = with_vector_mode
        print(f"\n[EXPERIMENT] mode={mode}, n_people={n_people}, seed={seed}, near_dupe_fraction={near_dupe_fraction}")
        print(f"[EXPERIMENT] noise_levels={noise_levels}, threshold={threshold}")

        # --- 1. Generate & insert persons ---
        df = generate_data_chunk(n_people)

        id_to_person: dict = {}  # milvus_id -> normalized person dict

        for _, row in df.iterrows():
            raw = dataframe_row_to_person_dict(row)
            normalized = normalize_person_data(raw)
            milvus_id = store_person(normalized, collection_name=test_collection)
            id_to_person[milvus_id] = normalized

        # --- 1b. Generate & insert near-duplicate confusers (optional) ---
        n_near_dupes = math.floor(n_people * near_dupe_fraction)
        if n_near_dupes > 0:
            dupe_rng = random.Random(seed + 1)  # isolated RNG stream
            near_dupes = generate_near_duplicates(
                list(id_to_person.values()), n_near_dupes, dupe_rng
            )
            for nd in near_dupes:
                store_person(nd, collection_name=test_collection)
            print(f"[EXPERIMENT] Inserted {n_near_dupes} near-duplicate confusers.")

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
                "near_dupe_fraction": near_dupe_fraction,
                "n_near_dupes": n_near_dupes,
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
            assert recall_by_level[mid_noise] >= 0.5, (
                f"Recall at noise=0.5 below 0.5: {recall_by_level[mid_noise]:.3f}"
            )
