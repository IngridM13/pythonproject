
import pytest
from unittest.mock import patch, MagicMock
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

@patch('database_utils.milvus_db_connection.utility')
@patch('database_utils.milvus_db_connection.Collection')
def test_ensure_collection_creates_when_missing(mock_collection, mock_utility):
    mock_utility.has_collection.return_value = False
    ensure_people_collection()
    mock_collection.assert_called_once()