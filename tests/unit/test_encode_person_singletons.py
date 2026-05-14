"""
Covers:
  Deterministic output across repeated calls (both modes).
"""
import pytest
import torch
from datetime import date
from unittest.mock import patch

import encoding_methods.encoding_and_search_milvus as em
from encoding_methods.encoding_and_search_milvus import encode_date, encode_person
from hdc.binary_hdc import HyperDimensionalComputingBinary
from hdc.bipolar_hdc import HyperDimensionalComputingBipolar
from utils.person_data_normalization import normalize_person_data
from configs.settings import HDC_DIM


@pytest.fixture
def sample_person():
    return normalize_person_data({
        "name": "John",
        "lastname": "Doe",
        "dob": "1990-05-15",
        "gender": "Male",
        "marital_status": "Single",
        "mobile_number": "123456789",
        "race": "Caucasian",
        "attrs": {"address": ["123 Main St"], "akas": [], "landlines": []},
    })


SAMPLE_DATE = date(1990, 5, 15)

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Repeated calls with the same input must return identical tensors."""

    def test_encode_person_binary_is_deterministic(self, sample_person):
        hv1 = encode_person(sample_person)
        hv2 = encode_person(sample_person)
        assert torch.equal(hv1, hv2)

    @pytest.mark.parametrize("with_vector_mode", ["float"], indirect=True)
    def test_encode_person_bipolar_is_deterministic(self, with_vector_mode, sample_person):
        hv1 = encode_person(sample_person)
        hv2 = encode_person(sample_person)
        assert torch.equal(hv1, hv2)

    def test_encode_date_binary_is_deterministic(self):
        hv1 = encode_date(SAMPLE_DATE)
        hv2 = encode_date(SAMPLE_DATE)
        assert torch.equal(hv1, hv2)

    @pytest.mark.parametrize("with_vector_mode", ["float"], indirect=True)
    def test_encode_date_bipolar_is_deterministic(self, with_vector_mode):
        hv1 = encode_date(SAMPLE_DATE)
        hv2 = encode_date(SAMPLE_DATE)
        assert torch.equal(hv1, hv2)


