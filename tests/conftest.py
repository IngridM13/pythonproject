import pytest
from pathlib import Path
from datetime import datetime

from tests.metrics.TestMetricsCollector import metrics_collector


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


@pytest.fixture(scope="session")
def test_metrics():
    """Fixture que proporciona el recolector de métricas para las pruebas"""
    return metrics_collector


@pytest.fixture(scope="session", autouse=True)
def save_test_metrics(request):
    """
    Fixture que se ejecuta automáticamente al final de la sesión de prueba
    para guardar las métricas recopiladas.
    """
    # Código que se ejecuta antes de las pruebas
    yield

    # Código que se ejecuta después de todas las pruebas
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"test_metrics_{timestamp}.json"

    metrics_collector.save_metrics(str(output_path))



