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



