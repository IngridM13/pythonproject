import pytest
import numpy as np
from encoding_methods.by_data_type.bool import BoolEncoding
from encoding_methods.by_data_type.numbers import IntegerEncoding


def test_bool_encoding_produces_expected_vectors():
    D_TEST = 1000  # Define la dimensión
    enc = BoolEncoding(D=D_TEST, flips_per_step=5, seed=42)

    h_true = enc.encode(True)
    h_false = enc.encode(False)

    # ... (verificaciones de shape y array_equal) ...
    assert h_true.shape == (D_TEST,)
    assert h_false.shape == (D_TEST,)
    assert not np.array_equal(h_true, h_false)

    # --- Verificación de similaridad directa ---

    # Similitud de un vector consigo mismo: dot(v, v) / D
    # np.dot(h_true, h_true) debe ser igual a D (1000)
    sim_true_true = np.dot(h_true, h_true) / D_TEST
    sim_false_false = np.dot(h_false, h_false) / D_TEST

    # Similitud cruzada
    sim_true_false = np.dot(h_true, h_false) / D_TEST

    # Usamos np.isclose() para comparar floats
    assert np.isclose(sim_true_true, 1.0)
    assert np.isclose(sim_false_false, 1.0)
    assert sim_true_false < 1.0


def test_bool_similarity_helper_method():
    enc = BoolEncoding(D=1000, flips_per_step=5, seed=42)

    # Este test SÍ usa enc.similarity() porque prueba el method en sí mismo,
    # pasándole los tipos de datos que espera (booleanos).

    assert np.isclose(enc.similarity(True, True), 1.0)
    assert np.isclose(enc.similarity(False, False), 1.0)
    assert enc.similarity(True, False) < 1.0

import numpy as np  # Asegúrate de importar numpy


def test_integer_encoding_preserves_ordinal_relationships():
    D_TEST = 1000  # Es buena práctica definir la D como una variable
    enc = IntegerEncoding(D=D_TEST, flips_per_step=5, n0=0, seed=42)

    h0 = enc.encode(0)
    h1 = enc.encode(1)
    h2 = enc.encode(2)
    h10 = enc.encode(10)


    # Verificar que números cercanos tienen mayor similitud
    # Debemos calcular la similaridad de coseno manualmente sobre los vectores.

    # Para vectores bipolares de norma constante (D), sim = dot(a, b) / D
    sim_0_1 = np.dot(h0, h1) / D_TEST
    sim_0_2 = np.dot(h0, h2) / D_TEST
    sim_0_10 = np.dot(h0, h10) / D_TEST
    # --- FIN DE LA CORRECCIÓN ---

    print(f"Sim(0,1): {sim_0_1}")
    print(f"Sim(0,2): {sim_0_2}")
    print(f"Sim(0,10): {sim_0_10}")

    assert sim_0_1 > sim_0_2 > sim_0_10