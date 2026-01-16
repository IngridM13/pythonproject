import pytest
import torch
from unittest.mock import patch, MagicMock
from database_utils.milvus_db_connection import connect, ensure_people_collection
import os
from pymilvus import MilvusClient

from encoding_methods.encoding_and_search_milvus import get_person_details, store_person

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

@pytest.mark.skipif(os.getenv('SKIP_MILVUS_TESTS', 'True') == 'True',
                    reason="Requiere Milvus en ejecución")
class TestMilvusIntegration:

    @pytest.fixture(scope="class")
    def milvus_client(self):
        milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
        client = MilvusClient(uri=milvus_uri)
        # Limpiar colección para tests - ensure we drop 'people' which is the default
        # used by store_person if not specified
        try:
            client.drop_collection("people")
        except:
            pass
        try:
            client.drop_collection("test_people")
        except:
            pass
        yield client

    def test_store_and_retrieve_person(self, milvus_client):
        # Crear una persona de prueba
        test_person = {
            "name": "Test",
            "lastname": "Person",
            "dob": "1990-01-01",
            "marital_status": "Single",
            "mobile_number": "123456789",
            "gender": "Other",
            "race": "Mixed",
            "address": ["Test Address"],
            "akas": ["Testy"],
            "landlines": ["987654321"]
        }

        # Almacenar la persona y obtener el ID
        person_id = store_person(test_person)

        # Verificar que se haya generado un ID
        assert person_id > 0

        # Recuperar los detalles de la persona
        retrieved = get_person_details(person_id)

        # Verificar que los datos recuperados coincidan con los originales
        assert retrieved["name"] == "Test"
        assert retrieved["lastname"] == "Person"
        assert retrieved["dob"] == "1990-01-01"
        assert "Test Address" in retrieved["address"]

    def test_person_data_to_torch_tensor(self, milvus_client):
        """
        Test converting person data to PyTorch tensor for potential ML preprocessing
        """
        test_person = {
            "name": "Test",
            "lastname": "Person",
            "dob": "1990-01-01",
            "mobile_number": "123456789",
            "gender": "Other",
            "race": "Mixed"
        }

        # Store person to ensure database interaction
        person_id = store_person(test_person)
        
        # Retrieve person details
        retrieved = get_person_details(person_id)
        
        # Convert selected features to PyTorch tensor
        # Use abs() to ensure positive values since hash() can be negative
        tensor_features = torch.tensor([
            abs(hash(retrieved['name'])),
            abs(hash(retrieved['lastname'])),
            int(retrieved['mobile_number'] or 0),
            abs(hash(retrieved['gender'])),
            abs(hash(retrieved['race']))
        ], dtype=torch.float32)
        
        # PyTorch-specific assertions
        assert isinstance(tensor_features, torch.Tensor)
        assert tensor_features.dtype == torch.float32
        assert tensor_features.shape == (5,)
        
        # Optional: compute some basic tensor stats
        assert torch.all(torch.isfinite(tensor_features))
        
        # Verify non-zero tensor
        assert tensor_features.sum() > 0

    def test_torch_tensor_to_milvus_embedding(self, milvus_client):
        """
        Demonstrate converting a PyTorch tensor to a Milvus-compatible embedding
        """
        # Create a sample PyTorch tensor
        sample_tensor = torch.rand(128)  # Example 128-dimensional embedding
        
        # Convert to numpy for Milvus (if needed)
        milvus_embedding = sample_tensor.numpy().tolist()
        
        # Store the embedding
        test_person = {
            "name": "TensorTest",
            "embedding": milvus_embedding
        }
        
        # Store person with tensor-derived embedding
        person_id = store_person(test_person)
        
        # Verify storage
        assert person_id > 0
        
        # Retrieve and validate
        retrieved = get_person_details(person_id)
        assert "embedding" in retrieved
        assert len(retrieved["embedding"]) == 128