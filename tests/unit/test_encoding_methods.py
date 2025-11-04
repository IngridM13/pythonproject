import pytest
import numpy as np
from encoding_methods.by_data_type.bool import BoolEncoding
from encoding_methods.by_data_type.numbers import IntegerEncoding

# test que falla. no estoy segura de qué hacer aca porque IA me dice que me haga otra version de similarity pero
# no me parece sensato.
def test_bool_encoding_produces_expected_vectors():
    enc = BoolEncoding(D=1000, flips_per_step=5, seed=42)

    h_true = enc.encode(True)
    h_false = enc.encode(False)

    # Verificar dimensionalidad
    assert h_true.shape == (1000,)
    assert h_false.shape == (1000,)

    # Verificar que sean diferentes
    assert not np.array_equal(h_true, h_false)

    # Verificar similitud (propia = 1.0, cruzada < 1.0)
    assert enc.similarity(h_true, h_true) == 1.0
    assert enc.similarity(h_false, h_false) == 1.0
    assert enc.similarity(h_true, h_false) < 1.0

# test que falla. no estoy segura de qué hacer aca porque IA me dice que me haga otra version de similarity pero
# no me parece sensato.
def test_integer_encoding_preserves_ordinal_relationships():
    enc = IntegerEncoding(D=1000, flips_per_step=5, n0=0, seed=42)

    h0 = enc.encode(0)
    h1 = enc.encode(1)
    h2 = enc.encode(2)
    h10 = enc.encode(10)

    # Verificar que números cercanos tienen mayor similitud
    sim_0_1 = enc.similarity(h0, h1)
    sim_0_2 = enc.similarity(h0, h2)
    sim_0_10 = enc.similarity(h0, h10)

    assert sim_0_1 > sim_0_2 > sim_0_10