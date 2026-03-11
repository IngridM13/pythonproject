import os
import uuid
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database_utils.milvus_db_connection import ensure_people_collection, _collection_cache


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

    print(f"\n[FIXTURE] Teardown: dropping {name}. Entities after: {col.num_entities}")
    _collection_cache.pop(name, None)
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
