import sys
import os
import uuid
from pathlib import Path
from datetime import datetime

import pytest
from pymilvus import MilvusClient, MilvusException
from database_utils.milvus_db_connection import ensure_people_collection

# Asegurar path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.metrics.TestMetricsCollector import metrics_collector


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

    yield name

    # --- DEBUG TEARDOWN ---
    print(f"\n[DEBUG-FIXTURE] Test finalizado. Iniciando Teardown para: {name}")
    print(f"[DEBUG-FIXTURE] Entidades DESPUÉS del test: {col.num_entities}")

    try:
        col.drop()
        print(f"[DEBUG-FIXTURE] Colección {name} dropeada exitosamente.")
    except MilvusException as e:
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
                except MilvusException as e_compact:
                    print(f"[DEBUG-FIXTURE] col.compact() falló: {e_compact}")
            else:
                print("[DEBUG-FIXTURE] No se encontraron filas para borrar.")
        except MilvusException as e_query:
            print(f"[DEBUG-FIXTURE] Falló el vaciado (query/delete): {e_query}")

    print("[DEBUG-FIXTURE] ----------------------------------\n")
    # --- FIN DEBUG ---


@pytest.fixture(scope="function")
def milvus_client():
    """Cliente Milvus (por si lo necesito en tests)."""
    return MilvusClient(uri="http://localhost:19530")

@pytest.fixture
def with_vector_mode(request):
    """Fixture para cambiar temporalmente el modo de vector durante una prueba."""
    import database_utils.milvus_db_connection as milvus_conn
    original_mode = milvus_conn.VECTOR_MODE

    # Establecer el modo solicitado
    milvus_conn.VECTOR_MODE = request.param

    yield request.param

    # Restaurar el modo original
    milvus_conn.VECTOR_MODE = original_mode

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


@pytest.fixture(scope="function")
def test_metrics(with_vector_mode):
    """
    Prepara el collector para cada modo y guarda el JSON al terminar.
    """
    metrics_collector.reset()

    yield metrics_collector

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "test_results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Este genera: test_metrics_binary_...json y test_metrics_bipolar_...json
    filename = f"test_metrics_{with_vector_mode}_{timestamp}.json"
    output_path = output_dir / filename

    metrics_collector.save_metrics(str(output_path))
    print(f"\n[FIXTURE-INFO] Métricas de modo '{with_vector_mode}' guardadas en {filename}")


