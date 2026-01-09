import pytest
import torch
from encoding_methods.by_data_type.bool import BoolEncoding
from encoding_methods.by_data_type.numbers import IntegerEncoding

def test_bool_encoding_produces_expected_vectors():
    D_TEST = 1000  # Define la dimensión
    enc = BoolEncoding(D=D_TEST, flips_per_step=5, seed=42)

    h_true = enc.encode(True)
    h_false = enc.encode(False)

    # Verify shape and uniqueness
    assert h_true.shape == (D_TEST,)
    assert h_false.shape == (D_TEST,)
    assert not torch.equal(h_true, h_false)

    # Similarity verification
    sim_true_true = torch.dot(h_true, h_true) / D_TEST
    sim_false_false = torch.dot(h_false, h_false) / D_TEST
    sim_true_false = torch.dot(h_true, h_false) / D_TEST

    # Use torch.isclose for float comparisons
    assert torch.isclose(sim_true_true, torch.tensor(1.0))
    assert torch.isclose(sim_false_false, torch.tensor(1.0))
    assert sim_true_false < 1.0

def test_bool_similarity_helper_method():
    enc = BoolEncoding(D=1000, flips_per_step=5, seed=42)

    assert torch.isclose(enc.similarity(True, True), torch.tensor(1.0))
    assert torch.isclose(enc.similarity(False, False), torch.tensor(1.0))
    assert enc.similarity(True, False) < 1.0

def test_integer_encoding_preserves_ordinal_relationships():
    D_TEST = 1000
    enc = IntegerEncoding(D=D_TEST, flips_per_step=5, n0=0, seed=42)

    h0 = enc.encode(0)
    h1 = enc.encode(1)
    h2 = enc.encode(2)
    h10 = enc.encode(10)

    # Calculate similarities using torch
    sim_0_1 = torch.dot(h0, h1) / D_TEST
    sim_0_2 = torch.dot(h0, h2) / D_TEST
    sim_0_10 = torch.dot(h0, h10) / D_TEST

    print(f"Sim(0,1): {sim_0_1}")
    print(f"Sim(0,2): {sim_0_2}")
    print(f"Sim(0,10): {sim_0_10}")

    # Verify ordinal relationships
    assert sim_0_1 > sim_0_2
    assert sim_0_2 > sim_0_10