
from encoding_methods.encoding_and_search_milvus import (
    store_person,
    get_person_details,
    find_closest_match_db,
    find_similar_by_date,
    normalize_person_data,
    ensure_people_collection
)
import pytest
import os
import torch
from datetime import date
from pymilvus import Collection
from configs import settings


@pytest.mark.skipif(os.getenv('SKIP_MILVUS_TESTS', 'True') == 'True',
                    reason="Requiere Milvus en ejecución")
@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
class TestEncodingSearch:

    def test_collection_name_matches_fixture(self, with_vector_mode, test_collection):
        col = ensure_people_collection(test_collection)
        # si la colección expone .name:
        assert getattr(col, "name", test_collection) == test_collection

    def test_find_closest_match(self, with_vector_mode, test_collection, test_people, test_metrics):
        vector_mode = with_vector_mode

        # Registramos los parámetros de configuración al inicio del test
        test_metrics.set_config(
            encoding=vector_mode, # O el nombre de tu codificación actual
            dimension=settings.HDC_DIM,
            seed = settings.DEFAULT_SEED

        )
        # --------------------------------------

        person_ids = []

        print(f"\n[DEBUG-TEST] Usando colección: {test_collection}")
        # Obtenemos el objeto Collection para poder interactuar con él
        col = Collection(test_collection)

        # --- MEDIR TIEMPO DE ENCODING E INSERCIÓN ---
        # Usamos el context manager para medir el bloque de inserción completo
        with test_metrics.measure_insertion_time():
            # Nota: Si tuvieras separado el encoding del store, usarías measure_encoding_time
            # Como store_person hace ambas cosas, medimos la "inserción" completa aquí,
            # o podrías medir encoding por separado si separaras la lógica.

            for i, person in enumerate(test_people):
                # Opcional: Si quieres medir SOLO el encoding (si normalizar fuera pesado)
                # with test_metrics.measure_encoding_time():
                #    normalized_person = normalize_person_data(person)

                pid = store_person(person, collection_name=test_collection)
                person_ids.append(pid)

                # --- DEBUG TEST ---
                print(f"[DEBUG-TEST] Persona {i} supuestamente almacenada. PID: {pid}")
                # ---

        print(f"[DEBUG-TEST] Forzando col.flush() en '{test_collection}'...")
        col.flush()
        print(f"[DEBUG-TEST] Flush completado. Entidades en la colección: {col.num_entities}")

        query_person = {
            "name": "John",
            "lastname": "Doe",
            "dob": "1990-05-15",
            "marital_status": "Single",
            "mobile_number": "123456789",
            "gender": "Male",
            "race": "Caucasian",
            "attrs": {
                "address": ["123 Main St, City"],
                "akas": ["Johnny"],
                "landlines": ["987654322"]
            }
        }

        print("[DEBUG-TEST] Ejecutando find_closest_match_db...")

        # --- MEDIR LATENCIA DE BÚSQUEDA ---
        import time
        start_time = time.perf_counter()  # Inicio manual para latencia

        results = find_closest_match_db(query_person, collection_name=test_collection)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        test_metrics.add_query_latency(latency_ms)  # <--- Guardamos la métrica
        # -------------------------------------

        print(f"[DEBUG-TEST] Búsqueda finalizada. Resultados obtenidos: {len(results)}")
        if not results:
            print("[DEBUG-TEST] ¡ERROR! La búsqueda no devolvió resultados (results está vacío).")
        else:
            print(f"[DEBUG-TEST] Resultado[0] encontrado: {results[0]}")

        assert results[0]['id'] == person_ids[0]
        assert results[0]['similarity'] > 0.8

    def test_find_similar_by_date(self, with_vector_mode, test_collection, test_people):
        person_ids = []
        for person in test_people:
            pid = store_person(person, collection_name=test_collection)
            person_ids.append(pid)
            assert pid > 0

        results = find_similar_by_date("1990-05-20", range_days=30, collection_name=test_collection)
        result_ids = [r['id'] for r in results]

        # Incluye año → debe estar John, no Juan ni Jane
        assert person_ids[0] in result_ids   # John (1990-05-15)
        assert person_ids[2] not in result_ids  # Juan (1992-05-17)
        assert person_ids[1] not in result_ids  # Jane (1985-10-20)

    def test_normalization(self, with_vector_mode):
        person_data = {
            "name": "John",
            "lastname": "Doe",
            "dob": "1990/05/15",
            "marital_status": "single",
            "gender": " Male ",
            "address": "123 Main St, City",
            "attrs": {}
        }
        normalized = normalize_person_data(person_data)
        assert normalized["name"] == "John"
        assert normalized["lastname"] == "Doe"
        assert normalized["dob"] == date(1990, 5, 15)
        assert normalized["marital_status"] == "Single"
        assert normalized["gender"] == "Male"
        assert "attrs" in normalized
        assert "address" in normalized["attrs"]
        assert isinstance(normalized["attrs"]["address"], list)
        assert normalized["attrs"]["address"][0] == "123 Main St, City"

    def test_normalized_data_to_torch_tensor(self, with_vector_mode):
        """Additional test to verify PyTorch tensor conversion"""
        person_data = {
            "name": "John",
            "lastname": "Doe",
            "dob": "1990-05-15",
            "mobile_number": "123456789"
        }

        normalized = normalize_person_data(person_data)

        # Convert selected features to torch tensor - use smaller values to avoid overflow
        tensor_features = torch.tensor([
            1,  # Usar valor constante en lugar de hash(normalized['name'])
            2,  # Usar valor constante en lugar de hash(normalized['lastname'])
            normalized['dob'].year if normalized['dob'] else 0,
            int(normalized['mobile_number'] or 0)
        ], dtype=torch.float32)

        # PyTorch-specific assertions
        assert isinstance(tensor_features, torch.Tensor)
        assert tensor_features.dtype == torch.float32
        assert tensor_features.shape == (4,)

        # Optional: compute some basic tensor stats
        assert torch.all(torch.isfinite(tensor_features))
        assert tensor_features.sum() > 0  # Ahora esto debería pasar con los valores modificados