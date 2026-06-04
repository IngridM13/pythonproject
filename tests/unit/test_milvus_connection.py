import pytest
from unittest.mock import patch, MagicMock
import torch  # Add torch import
from database_utils.milvus_db_connection import connect, ensure_people_collection

@pytest.fixture
def mock_connections():
    with patch('database_utils.milvus_db_connection.connections') as mock_conn:
        mock_conn.has_connection.return_value = False
        yield mock_conn

def test_connect_establishes_connection(mock_connections):
    connect()
    mock_connections.connect.assert_called_once()

def test_connect_skips_if_connection_exists(mock_connections):
    mock_connections.has_connection.return_value = True
    connect()
    mock_connections.connect.assert_not_called()

def test_ensure_collection_returns_cached_object(monkeypatch):
    """Cache hit must return the stored collection without connecting to Milvus."""
    import database_utils.milvus_db_connection as db

    mock_col = MagicMock()
    monkeypatch.setenv("MILVUS_VECTOR_MODE", "binary")
    monkeypatch.setitem(db._collection_cache, "people_binary", mock_col)

    result = ensure_people_collection()
    assert result is mock_col

# PyTorch-specific test
def test_torch_tensor_creation():
    """Verify basic torch tensor operations are working correctly."""
    # Test tensor creation
    t = torch.tensor([1, 2, 3])
    assert isinstance(t, torch.Tensor)
    
    # Test tensor type and shape
    assert t.dtype == torch.int64
    assert t.shape == (3,)
    
    # Test tensor operations
    result = t * 2
    assert torch.equal(result, torch.tensor([2, 4, 6]))