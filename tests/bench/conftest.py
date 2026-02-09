import pytest
from tests.metrics.TestMetricsCollector import metrics_collector
from datetime import datetime
from pathlib import Path

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
