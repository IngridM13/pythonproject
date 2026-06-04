"""
Covers:
  Deterministic output across repeated calls (both modes).
  Mode dispatch: switching MILVUS_VECTOR_MODE changes encoding output.
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

    @pytest.mark.parametrize("with_vector_mode", ["binary"], indirect=True)
    def test_encode_person_binary_is_deterministic(self, with_vector_mode, sample_person):
        hv1 = encode_person(sample_person)
        hv2 = encode_person(sample_person)
        assert torch.equal(hv1, hv2)

    @pytest.mark.parametrize("with_vector_mode", ["float"], indirect=True)
    def test_encode_person_bipolar_is_deterministic(self, with_vector_mode, sample_person):
        hv1 = encode_person(sample_person)
        hv2 = encode_person(sample_person)
        assert torch.equal(hv1, hv2)

    @pytest.mark.parametrize("with_vector_mode", ["binary"], indirect=True)
    def test_encode_date_binary_is_deterministic(self, with_vector_mode):
        hv1 = encode_date(SAMPLE_DATE)
        hv2 = encode_date(SAMPLE_DATE)
        assert torch.equal(hv1, hv2)

    @pytest.mark.parametrize("with_vector_mode", ["float"], indirect=True)
    def test_encode_date_bipolar_is_deterministic(self, with_vector_mode):
        hv1 = encode_date(SAMPLE_DATE)
        hv2 = encode_date(SAMPLE_DATE)
        assert torch.equal(hv1, hv2)


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

class TestModeDispatch:
    """Switching MILVUS_VECTOR_MODE must change which HDC singleton is used."""

    def test_binary_mode_returns_binary_vector(self, sample_person):
        with patch.dict("os.environ", {"MILVUS_VECTOR_MODE": "binary"}):
            hv = encode_person(sample_person)
        assert hv.shape == (HDC_DIM,)
        assert torch.all((hv == 0) | (hv == 1)), "binary mode debe retornar valores en {0, 1}"

    def test_float_mode_returns_bipolar_vector(self, sample_person):
        with patch.dict("os.environ", {"MILVUS_VECTOR_MODE": "float"}):
            hv = encode_person(sample_person)
        assert hv.shape == (HDC_DIM,)
        assert torch.all((hv == 1) | (hv == -1)), "float mode debe retornar valores en {-1, +1}"

    def test_binary_and_float_modes_produce_different_vectors(self, sample_person):
        with patch.dict("os.environ", {"MILVUS_VECTOR_MODE": "binary"}):
            hv_binary = encode_person(sample_person)
        with patch.dict("os.environ", {"MILVUS_VECTOR_MODE": "float"}):
            hv_float = encode_person(sample_person)
        assert not torch.equal(hv_binary.float(), hv_float.float()), \
            "binary y float mode deben producir vectores distintos para la misma persona"


