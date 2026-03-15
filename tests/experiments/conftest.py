import json
import os
import uuid
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database_utils.milvus_db_connection import ensure_people_collection, _collection_cache


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def dataframe_row_to_person_dict(row) -> dict:
    """
    Convert a pandas Series (row from generate_data_chunk DataFrame) to the
    person dict format expected by normalize_person_data / store_person.
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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class", params=["binary", "float"])
def with_vector_mode(request):
    import database_utils.milvus_db_connection as milvus_conn
    original_mode = milvus_conn.VECTOR_MODE

    mode = request.param
    milvus_conn.VECTOR_MODE = mode

    yield mode

    milvus_conn.VECTOR_MODE = original_mode


@pytest.fixture(scope="class")
def test_collection(with_vector_mode):
    name = f"people_test_{uuid.uuid4().hex[:8]}"
    print(f"\n[FIXTURE] Creating collection: {name}")

    col = ensure_people_collection(name)
    print(f"[FIXTURE] Collection '{col.name}' ready. Entities before: {col.num_entities}")

    yield name

    if os.environ.get("KEEP_COLLECTION", "").lower() in ("1", "true", "yes"):
        print(f"\n[FIXTURE] KEEP_COLLECTION set — skipping teardown for '{name}'.")
        print(f"[FIXTURE] Collection has {col.num_entities} entities. Inspect it, then drop manually.")
        return

    print(f"\n[FIXTURE] Teardown: dropping {name}. Entities after: {col.num_entities}")
    _collection_cache.pop(f"{name}_{with_vector_mode}", None)
    try:
        col.drop()
        print(f"[FIXTURE] Collection {name} dropped.")
    except Exception as e:
        print(f"[FIXTURE] col.drop() failed: {e}. Attempting manual cleanup...")
        try:
            rows = col.query(expr="id >= 0", output_fields=["id"], limit=100000)
            if rows:
                col.delete(ids=[int(r["id"]) for r in rows])
                col.flush()
        except Exception as e2:
            print(f"[FIXTURE] Manual cleanup failed: {e2}")
