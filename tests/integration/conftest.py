import uuid
import pytest
from pymilvus import MilvusClient
from database_utils.milvus_db_connection import ensure_people_collection

@pytest.fixture(scope="function")
def test_collection():
    """
    Crea una colección única para este test y la destruye al final.
    """
    name = f"people_test_{uuid.uuid4().hex[:8]}"
    col = ensure_people_collection(name)
    yield name
    # Tear down: intentar dropear; si falla, vaciar
    try:
        col.drop()
    except Exception:
        try:
            rows = col.query(expr="id >= 0", output_fields=["id"], limit=100000)
            if rows:
                col.delete(ids=[int(r["id"]) for r in rows])
                col.flush()
                try:
                    col.compact()
                except Exception:
                    pass
        except Exception:
            pass

@pytest.fixture(scope="function")
def milvus_client():
    """Cliente Milvus (por si lo necesitás en tests)."""
    return MilvusClient(uri="http://localhost:19530")

@pytest.fixture(scope="function")
def test_people():
    """Datos de prueba en formato DB-style con attrs."""
    return [
        {
            "name": "John",
            "lastname": "Doe",
            "dob": "1990-05-15",
            "marital_status": "Single",
            "mobile_number": "123456789",
            "gender": "Male",
            "race": "Caucasian",
            "attrs": {
                "address": ["123 Main St, City"],
                "akas": ["Johnny"],
                "landlines": ["987654321"]
            }
        },
        {
            "name": "Jane",
            "lastname": "Smith",
            "dob": "1985-10-20",
            "marital_status": "Married",
            "mobile_number": "987654321",
            "gender": "Female",
            "race": "African",
            "attrs": {
                "address": ["456 Oak St, Town"],
                "akas": ["Janie"],
                "landlines": ["123456789"]
            }
        },
        {
            "name": "Juan",
            "lastname": "Doe",
            "dob": "1992-05-17",
            "marital_status": "Single",
            "mobile_number": "567891234",
            "gender": "Male",
            "race": "Hispanic",
            "attrs": {
                "address": ["789 Pine St, Village"],
                "akas": ["Juanito"],
                "landlines": ["432156789"]
            }
        }
    ]
