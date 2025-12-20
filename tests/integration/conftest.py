import uuid
import pytest
from pymilvus import MilvusClient
from database_utils.milvus_db_connection import ensure_people_collection

@pytest.fixture(scope="function")
def test_collection():
    """
    Crea una colección única para este test y la destruye al final.
    """
    # --- DEBUG SETUP ---
    print("\n[DEBUG-FIXTURE] ----------------------------------")
    name = f"people_test_{uuid.uuid4().hex[:8]}"
    print(f"[DEBUG-FIXTURE] Creando colección: {name}")

    col = ensure_people_collection(name)

    print(f"[DEBUG-FIXTURE] Colección '{col.name}' asegurada.")
    print(f"[DEBUG-FIXTURE] Entidades ANTES del test: {col.num_entities}")
    print("[DEBUG-FIXTURE] Entregando nombre al test...")
    # --- FIN DEBUG ---

    yield name  # <--- AQUÍ CORRE TU TEST

    # --- DEBUG TEARDOWN ---
    print(f"\n[DEBUG-FIXTURE] Test finalizado. Iniciando Teardown para: {name}")
    print(f"[DEBUG-FIXTURE] Entidades DESPUÉS del test: {col.num_entities}")

    try:
        col.drop()
        print(f"[DEBUG-FIXTURE] Colección {name} dropeada exitosamente.")
    except Exception as e:
        print(f"[DEBUG-FIXTURE] Falló col.drop(): {e}. Intentando vaciar...")
        try:
            rows = col.query(expr="id >= 0", output_fields=["id"], limit=100000)
            if rows:
                print(f"[DEBUG-FIXTURE] Encontradas {len(rows)} filas para borrar.")
                col.delete(ids=[int(r["id"]) for r in rows])
                col.flush()
                print("[DEBUG-FIXTURE] col.delete() y col.flush() llamados.")
                try:
                    col.compact()
                    print("[DEBUG-FIXTURE] col.compact() exitoso.")
                except Exception as e_compact:
                    print(f"[DEBUG-FIXTURE] col.compact() falló: {e_compact}")
            else:
                print("[DEBUG-FIXTURE] No se encontraron filas para borrar.")
        except Exception as e_query:
            print(f"[DEBUG-FIXTURE] Falló el vaciado (query/delete): {e_query}")

    print("[DEBUG-FIXTURE] ----------------------------------\n")
    # --- FIN DEBUG ---

@pytest.fixture(scope="function")
def milvus_client():
    """Cliente Milvus (por si lo necesito en tests)."""
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
