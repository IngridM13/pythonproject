import os
import uuid
import sys
from pathlib import Path
from datetime import datetime
import pytest

# Ensure path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database_utils.milvus_db_connection import ensure_people_collection


@pytest.fixture(scope="class")
def with_vector_mode(request):
    """
    Configura el modo de vector para las pruebas.
    Valores aceptados (en parametrize): 'binary', 'float'
    """
    import database_utils.milvus_db_connection as milvus_conn
    original_mode = milvus_conn.VECTOR_MODE
    
    # Set vector mode for test
    mode = request.param
    milvus_conn.VECTOR_MODE = mode
    
    yield mode
    
    # Restore original
    milvus_conn.VECTOR_MODE = original_mode


@pytest.fixture(scope="function")
def test_metrics(with_vector_mode):
    """
    Fixture that creates a TestMetricsCollector for benchmarking
    and saves metrics to a file after test completes.
    """
    from tests.metrics.TestMetricsCollector import TestMetricsCollector
    metrics = TestMetricsCollector()
    yield metrics
    
    # Generate output path for metrics
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_metrics_{with_vector_mode}_{timestamp}.json"
    output_path = str(output_dir / filename)
    
    # Save metrics to the generated path
    metrics.save_metrics(output_path)
    print(f"\n[FIXTURE-INFO] Métricas de modo '{with_vector_mode}' guardadas en {filename}")


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