import pytest
from datetime import date
from encoding_methods.encoding_and_search_milvus import normalize_person_data

def test_normalize_person_data_standardizes_keys():
    input_data = {
        "Name": "John",
        "LastName": "Doe"
    }
    normalized = normalize_person_data(input_data)
    assert "name" in normalized
    assert "lastname" in normalized
    assert normalized["name"] == "John"
    assert normalized["lastname"] == "Doe"

def test_normalize_person_data_parses_list_strings_db_schema():
    input_data = {"address": ["123 Main St", "456 Broadway"]}
    normalized = normalize_person_data(input_data)

    expected_top = {
        "name", "lastname", "mobile_number", "race", "attrs",
        "dob", "gender", "marital_status"
    }
    assert set(normalized.keys()) == expected_top

    assert {"address", "akas", "landlines"}.issubset(set(normalized["attrs"].keys()))
    assert normalized["attrs"]["address"] == ["123 Main St", "456 Broadway"]

    # Defaults coherentes con la implementación actual
    assert normalized["name"] == ""
    assert normalized["lastname"] == ""
    assert normalized["mobile_number"] == ""
    assert normalized["race"] == ""
    assert normalized["attrs"]["akas"] == []
    assert normalized["attrs"]["landlines"] == []
    assert normalized["dob"] is None
    assert normalized["gender"] == ""
    assert normalized["marital_status"] == ""

def test_normalize_person_data_converts_date_strings():
    input_data = {
        "dob": "1990-05-15"
    }
    normalized = normalize_person_data(input_data)
    assert isinstance(normalized["dob"], date)
    assert normalized["dob"].year == 1990
    assert normalized["dob"].month == 5
    assert normalized["dob"].day == 15
