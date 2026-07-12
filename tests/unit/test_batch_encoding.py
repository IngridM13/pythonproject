import torch
import pytest
from hdc.binary_hdc import HyperDimensionalComputingBinary
from hdc.bipolar_hdc import HyperDimensionalComputingBipolar

DIM = 1000
SEED = 42

RAW_PEOPLE = [
    {"name": "Alice", "lastname": "Smith", "dob": "1990-01-01", "gender": "Female"},
    {"name": "Bob", "lastname": "Jones", "dob": "1985-06-15", "gender": "Male"},
    {"name": "Carol", "lastname": "White", "dob": "2000-12-31", "gender": "Female"},
]


class TestBinaryEncodeBatch:
    def test_output_shape(self):
        hdc = HyperDimensionalComputingBinary(dim=DIM, seed=SEED)
        batch = hdc.encode_batch(RAW_PEOPLE)
        assert batch.shape == (len(RAW_PEOPLE), DIM)

    def test_empty_returns_zero_rows(self):
        hdc = HyperDimensionalComputingBinary(dim=DIM, seed=SEED)
        batch = hdc.encode_batch([])
        assert batch.shape == (0, DIM)

    def test_single_element_shape(self):
        hdc = HyperDimensionalComputingBinary(dim=DIM, seed=SEED)
        batch = hdc.encode_batch([RAW_PEOPLE[0]])
        assert batch.shape == (1, DIM)

    def test_values_in_binary_range(self):
        hdc = HyperDimensionalComputingBinary(dim=DIM, seed=SEED)
        batch = hdc.encode_batch(RAW_PEOPLE)
        assert torch.all((batch == 0) | (batch == 1))

    def test_deterministic(self):
        hdc = HyperDimensionalComputingBinary(dim=DIM, seed=SEED)
        batch1 = hdc.encode_batch(RAW_PEOPLE)
        batch2 = hdc.encode_batch(RAW_PEOPLE)
        assert torch.equal(batch1, batch2)

    def test_different_people_produce_different_vectors(self):
        hdc = HyperDimensionalComputingBinary(dim=DIM, seed=SEED)
        batch = hdc.encode_batch(RAW_PEOPLE)
        assert not torch.equal(batch[0], batch[1])
        assert not torch.equal(batch[0], batch[2])
        assert not torch.equal(batch[1], batch[2])


class TestBipolarEncodeBatch:
    def test_output_shape(self):
        hdc = HyperDimensionalComputingBipolar(dim=DIM, seed=SEED)
        batch = hdc.encode_batch(RAW_PEOPLE)
        assert batch.shape == (len(RAW_PEOPLE), DIM)

    def test_empty_returns_zero_rows(self):
        hdc = HyperDimensionalComputingBipolar(dim=DIM, seed=SEED)
        batch = hdc.encode_batch([])
        assert batch.shape == (0, DIM)

    def test_single_element_shape(self):
        hdc = HyperDimensionalComputingBipolar(dim=DIM, seed=SEED)
        batch = hdc.encode_batch([RAW_PEOPLE[0]])
        assert batch.shape == (1, DIM)

    def test_values_in_bipolar_range(self):
        hdc = HyperDimensionalComputingBipolar(dim=DIM, seed=SEED)
        batch = hdc.encode_batch(RAW_PEOPLE)
        assert torch.all((batch == -1.0) | (batch == 1.0))

    def test_deterministic(self):
        hdc = HyperDimensionalComputingBipolar(dim=DIM, seed=SEED)
        batch1 = hdc.encode_batch(RAW_PEOPLE)
        batch2 = hdc.encode_batch(RAW_PEOPLE)
        assert torch.equal(batch1, batch2)

    def test_different_people_produce_different_vectors(self):
        hdc = HyperDimensionalComputingBipolar(dim=DIM, seed=SEED)
        batch = hdc.encode_batch(RAW_PEOPLE)
        assert not torch.equal(batch[0], batch[1])
        assert not torch.equal(batch[0], batch[2])
        assert not torch.equal(batch[1], batch[2])
