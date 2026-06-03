import pytest



@pytest.fixture
def with_vector_mode(request, monkeypatch):
    """Fixture para cambiar temporalmente el modo de vector durante una prueba."""
    monkeypatch.setenv("MILVUS_VECTOR_MODE", request.param)
    yield request.param



