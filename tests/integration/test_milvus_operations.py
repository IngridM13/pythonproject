import pytest
import os
from pymilvus import MilvusClient
from database_utils.milvus_db_connection import ensure_people_collection
from encoding_methods.encoding_and_search_milvus import store_person, get_person_details


# Esta prueba requiere una instancia de Milvus en ejecución
# Puedes saltarla si no tienes el entorno configurado
@pytest.mark.skipif(os.getenv('SKIP_MILVUS_TESTS', 'True') == 'True',
                    reason="Requiere Milvus en ejecución")
class TestMilvusIntegration:

    @pytest.fixture(scope="class")
    def milvus_client(self):
        client = MilvusClient(uri="http://localhost:19530")
        # Limpiar colección para tests
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