import os
import uuid
import sys
import re
import pytest
from pathlib import Path
from datetime import datetime

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

def _sanitize_filename_part(s: str, max_len: int = 80) -> str:
    # deja letras, números, guión y guión bajo; el resto a "_"
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s

def _infer_metrics_tag(node) -> str:
    """
    Inferí un tag estable (search/insert/encoding/...) a partir del nombre/nodeid
    sin tener que modificar cada test.
    """
    haystack = f"{node.name} {node.nodeid}".lower()

    # Ajustá estos heurísticos a tus nombres reales
    if "search" in haystack:
        return "search"
    if "insert" in haystack and "batch" in haystack:
        return "insert_batch"
    if "insert" in haystack:
        return "insert"
    if "encode" in haystack :
        return "encoding"

    # fallback: usa el nombre del test (más específico)
    return _sanitize_filename_part(node.name)

@pytest.fixture(scope="function")
def test_metrics(with_vector_mode, request):
    """
    Fixture que crea un TestMetricsCollector y guarda métricas a un archivo
    cuyo nombre incluye el "tipo" de test (search/insert/encoding/...).
    """
    from tests.metrics.TestMetricsCollector import TestMetricsCollector

    metrics = TestMetricsCollector()
    yield metrics

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "test_results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    test_tag = _infer_metrics_tag(request.node)
    test_tag = _sanitize_filename_part(test_tag)

    filename = f"test_metrics_{with_vector_mode}_{test_tag}_{timestamp}.json"
    output_path = str(output_dir / filename)

    metrics.save_metrics(output_path)
    print(f"\n[FIXTURE-INFO] Métricas modo='{with_vector_mode}' tag='{test_tag}' guardadas en {filename}")



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