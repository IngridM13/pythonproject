from datetime import date

import pytest
import os
from encoding_methods.encoding_and_search_milvus import (
    store_person,
    get_person_details,
    find_closest_match_db,
    find_similar_by_date,
    normalize_person_data,
    ensure_people_collection
)

@pytest.mark.skipif(os.getenv('SKIP_MILVUS_TESTS', 'True') == 'True',
                    reason="Requiere Milvus en ejecución")
class TestEncodingSearch:

    def test_collection_name_matches_fixture(self, test_collection):
        col = ensure_people_collection(test_collection)
        # si la colección expone .name:
        assert getattr(col, "name", test_collection) == test_collection

    def test_find_closest_match(self, test_collection, test_people):
        person_ids = []
        for person in test_people:
            pid = store_person(person, collection_name=test_collection)
            person_ids.append(pid)
            assert pid > 0

        query_person = {
            "name": "Jon",
            "lastname": "Doe",
            "dob": "1990-05-16",
            "marital_status": "Single",
            "gender": "Male",
            "attrs": {}
        }
        normalized_query = normalize_person_data(query_person)
        results = find_closest_match_db(normalized_query, collection_name=test_collection)

        assert results[0]['id'] == person_ids[0]
        assert results[0]['similarity'] > 0.8

    def test_find_similar_by_date(self, test_collection, test_people):
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

    def test_normalization(self):
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
