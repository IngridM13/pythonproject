"""
Unit tests for field_weights and excluded_fields parameters in
encode_person_binary and encode_person_bipolar.
"""
import torch
import pytest

from hdc.binary_hdc import HyperDimensionalComputingBinary
from hdc.bipolar_hdc import HyperDimensionalComputingBipolar

DIM = 1000
SEED = 42

PERSON = {
    "name": "John",
    "lastname": "Doe",
    "dob": "1990-05-15",
    "gender": "Male",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def hdc_binary():
    return HyperDimensionalComputingBinary(dim=DIM, seed=SEED)


@pytest.fixture
def hdc_bipolar():
    return HyperDimensionalComputingBipolar(dim=DIM, seed=SEED)


# ---------------------------------------------------------------------------
# field_weights
# ---------------------------------------------------------------------------

class TestFieldWeights:

    def test_binary_weight_changes_output(self, hdc_binary):
        hv_default = hdc_binary.encode_person_binary(PERSON)
        hv_weighted = hdc_binary.encode_person_binary(PERSON, field_weights={"name": 5})
        assert not torch.equal(hv_default, hv_weighted)

    def test_bipolar_weight_changes_output(self, hdc_bipolar):
        hv_default = hdc_bipolar.encode_person_bipolar(PERSON)
        hv_weighted = hdc_bipolar.encode_person_bipolar(PERSON, field_weights={"name": 5})
        assert not torch.equal(hv_default, hv_weighted)

    def test_binary_weight_one_equals_default(self, hdc_binary):
        hv_default = hdc_binary.encode_person_binary(PERSON)
        hv_explicit = hdc_binary.encode_person_binary(
            PERSON, field_weights={"name": 1, "lastname": 1, "dob": 1, "gender": 1}
        )
        assert torch.equal(hv_default, hv_explicit)

    def test_bipolar_weight_one_equals_default(self, hdc_bipolar):
        hv_default = hdc_bipolar.encode_person_bipolar(PERSON)
        hv_explicit = hdc_bipolar.encode_person_bipolar(
            PERSON, field_weights={"name": 1, "lastname": 1, "dob": 1, "gender": 1}
        )
        assert torch.equal(hv_default, hv_explicit)

    def test_binary_unknown_field_in_weights_ignored(self, hdc_binary):
        hv = hdc_binary.encode_person_binary(PERSON, field_weights={"nonexistent": 5})
        assert hv.shape == (DIM,)

    def test_bipolar_unknown_field_in_weights_ignored(self, hdc_bipolar):
        hv = hdc_bipolar.encode_person_bipolar(PERSON, field_weights={"nonexistent": 5})
        assert hv.shape == (DIM,)


# ---------------------------------------------------------------------------
# excluded_fields
# ---------------------------------------------------------------------------

class TestExcludedFields:

    def test_binary_excluded_field_changes_output(self, hdc_binary):
        hv_full = hdc_binary.encode_person_binary(PERSON)
        hv_no_name = hdc_binary.encode_person_binary(PERSON, excluded_fields={"name"})
        assert not torch.equal(hv_full, hv_no_name)

    def test_bipolar_excluded_field_changes_output(self, hdc_bipolar):
        hv_full = hdc_bipolar.encode_person_bipolar(PERSON)
        hv_no_name = hdc_bipolar.encode_person_bipolar(PERSON, excluded_fields={"name"})
        assert not torch.equal(hv_full, hv_no_name)

    def test_binary_empty_set_equals_none(self, hdc_binary):
        hv_none = hdc_binary.encode_person_binary(PERSON, excluded_fields=None)
        hv_empty = hdc_binary.encode_person_binary(PERSON, excluded_fields=set())
        assert torch.equal(hv_none, hv_empty)

    def test_bipolar_empty_set_equals_none(self, hdc_bipolar):
        hv_none = hdc_bipolar.encode_person_bipolar(PERSON, excluded_fields=None)
        hv_empty = hdc_bipolar.encode_person_bipolar(PERSON, excluded_fields=set())
        assert torch.equal(hv_none, hv_empty)

    def test_binary_unknown_excluded_field_ignored(self, hdc_binary):
        hv = hdc_binary.encode_person_binary(PERSON, excluded_fields={"nonexistent_field"})
        hv_default = hdc_binary.encode_person_binary(PERSON)
        assert torch.equal(hv, hv_default)

    def test_bipolar_unknown_excluded_field_ignored(self, hdc_bipolar):
        hv = hdc_bipolar.encode_person_bipolar(PERSON, excluded_fields={"nonexistent_field"})
        hv_default = hdc_bipolar.encode_person_bipolar(PERSON)
        assert torch.equal(hv, hv_default)

    def test_binary_all_fields_excluded_returns_zeros(self, hdc_binary):
        all_fields = {"name", "lastname", "dob", "gender", "marital_status",
                      "mobile_number", "race", "attrs"}
        hv = hdc_binary.encode_person_binary(PERSON, excluded_fields=all_fields)
        assert hv.shape == (DIM,)
        assert torch.all(hv == 0)

    def test_bipolar_all_fields_excluded_returns_correct_shape(self, hdc_bipolar):
        all_fields = {"name", "lastname", "dob", "gender", "marital_status",
                      "mobile_number", "race", "attrs"}
        hv = hdc_bipolar.encode_person_bipolar(PERSON, excluded_fields=all_fields)
        assert hv.shape == (DIM,)
