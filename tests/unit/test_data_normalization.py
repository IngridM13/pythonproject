import pytest
import torch
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

def test_normalized_data_torch_conversion():
    """Verify that normalized data can be converted to PyTorch tensor"""
    input_data = {
        "name": "John",
        "lastname": "Doe",
        "mobile_number": "123456789",
        "dob": "1990-05-15"
    }
    
    normalized = normalize_person_data(input_data)
    
    # Convert some numeric/hashable features to tensor
    numeric_features = [
        hash(normalized['name']),
        hash(normalized['lastname']),
        int(normalized['mobile_number'] or 0),
        normalized['dob'].year if normalized['dob'] else 0
    ]
    
    # Convert to torch tensor
    tensor_features = torch.tensor(numeric_features, dtype=torch.float32)
    
    # Verify torch tensor properties
    assert isinstance(tensor_features, torch.Tensor)
    assert tensor_features.dtype == torch.float32
    assert tensor_features.shape == (4,)