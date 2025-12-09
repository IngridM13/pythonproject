import os
import sys
from datetime import date
import pytest

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Make project root importable (same as before)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Your code (unchanged) ---
from encoding_methods.encoding_and_search_milvus import (
    encode_person, encode_date, DIMENSION, store_person,
    normalize_person_data, find_closest_match_db, parse_date,
    find_similar_by_date
)

# --- Milvus glue (new) ---
from database_utils.milvus_db_connection import ensure_people_collection
try:
    # If you exposed this in your schema module
    from database_utils.milvus_db_connection import VECTOR_MODE
except Exception:
    VECTOR_MODE = os.getenv("MILVUS_VECTOR_MODE", "binary").lower()

# Global for deterministic symbol HVs during a test run
hv_dict = {}

# --- Helpers to read/compare hv from Milvus ---

def _unpack_binary_to_bipolar(payload, dim: int) -> np.ndarray:
    """
    Accepts:
      - bytes/bytearray/memoryview of packed bits (0/1)
      - list with a single bytes object (common from Milvus)
      - list/ndarray of uint8 values (either packed bytes 0..255 or already-unpacked bits 0/1)
    Returns bipolar {-1,+1} of length dim.
    """
    # If Milvus gave us a list wrapper, unwrap common cases
    if isinstance(payload, list):
        if len(payload) == 0:
            raise ValueError("Empty hv payload")
        first = payload[0]
        # Case: [b'\x12\x34...']  -> unwrap to bytes
        if isinstance(first, (bytes, bytearray, memoryview)):
            payload = first
        else:
            # Treat as numeric uint8 array (could be bits or packed bytes)
            arr = np.asarray(payload, dtype=np.uint8).ravel()
            # If already bits of length dim, use directly
            if arr.size == dim and np.all((arr == 0) | (arr == 1)):
                bits = arr
            else:
                # Assume packed bytes 0..255
                bits = np.unpackbits(arr, bitorder="big")
            bits = bits[:dim]
            return np.where(bits == 1, 1, -1).astype(np.int8)

    # Bytes-like path
    if isinstance(payload, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(payload, dtype=np.uint8)
    else:
        # Fallback (handles ndarray, lists of ints, etc.)
        arr = np.asarray(payload, dtype=np.uint8).ravel()

    bits = np.unpackbits(arr, bitorder="big")
    bits = bits[:dim]
    return np.where(bits == 1, 1, -1).astype(np.int8)

def _unpack_binary(payload, dim: int) -> np.ndarray:
    """
    Similar to _unpack_binary_to_bipolar but returns binary {0,1} values.
    Accepts:
      - bytes/bytearray/memoryview of packed bits (0/1)
      - list with a single bytes object (common from Milvus)
      - list/ndarray of uint8 values (either packed bytes 0..255 or already-unpacked bits 0/1)
    Returns binary {0,1} of length dim.
    """
    # If Milvus gave us a list wrapper, unwrap common cases
    if isinstance(payload, list):
        if len(payload) == 0:
            raise ValueError("Empty hv payload")
        first = payload[0]
        # Case: [b'\x12\x34...']  -> unwrap to bytes
        if isinstance(first, (bytes, bytearray, memoryview)):
            payload = first
        else:
            # Treat as numeric uint8 array (could be bits or packed bytes)
            arr = np.asarray(payload, dtype=np.uint8).ravel()
            # If already bits of length dim, use directly
            if arr.size == dim and np.all((arr == 0) | (arr == 1)):
                return arr.astype(np.uint8)  # Already binary
            else:
                # Assume packed bytes 0..255
                bits = np.unpackbits(arr, bitorder="big")
                return bits[:dim].astype(np.uint8)  # Return binary bits

    # Bytes-like path
    if isinstance(payload, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(payload, dtype=np.uint8)
    else:
        # Fallback (handles ndarray, lists of ints, etc.)
        arr = np.asarray(payload, dtype=np.uint8).ravel()

    # Now arr is properly defined in all code paths
    bits = np.unpackbits(arr, bitorder="big")
    bits = bits[:dim]
    return bits.astype(np.uint8)  # Returns 0/1

def _load_person_from_milvus(person_id: int, with_hv: bool = True):
    """
    Query Milvus by primary key and return a dict with scalar fields and optionally 'hv'.
    """
    col = ensure_people_collection()
    out_fields = ["name", "lastname", "dob", "marital_status", "mobile_number", "gender", "race", "attrs"]
    if with_hv:
        out_fields.append("hv")

    res = col.query(expr=f"id == {person_id}", output_fields=out_fields, consistency_level="Strong")
    # query returns list[dict]
    return res[0] if res else None

def _delete_people_from_milvus(ids):
    col = ensure_people_collection()
    if not ids:
        # When called with empty list, delete all records instead of returning early
        col.delete(expr="id >= 0")  # Delete all records
        return
    id_list = ",".join(str(i) for i in ids)
    col.delete(expr=f"id in [{id_list}]")

# -------------------- Tests --------------------

@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
def test_encoding_consistency(with_vector_mode):
    """Verificamos que el encoding es consistente con enfoque determinista"""

    # Verifica que el modo sea el esperado
    from database_utils.milvus_db_connection import get_vector_mode
    current_mode = get_vector_mode()
    assert current_mode == with_vector_mode, f"Expected mode {with_vector_mode}, got {current_mode}"

    print(f"\n--- Verificación de Consistencia del Encoding (modo: {with_vector_mode}) ---")

    test_person = {
        "name": "John",
        "lastname": "Doe",
        "dob": "1990-05-15",  # normalize() lo va a parsear
        "marital_status": "Married",
        "mobile_number": "555-1111",
        "gender": "Male",
        "race": "Caucasian",
        "attrs": {  # <-- los arrays viven en attrs en Milvus
            "address": ["456 Main St", "Apt 789"],
            "akas": ["Johnny", "J.D."],
            "landlines": ["555-9876"]
        }
    }

    global hv_dict
    hv_dict = {}
    encoding1 = encode_person(test_person)

    hv_dict = {}
    encoding2 = encode_person(test_person)

    are_equal = np.array_equal(encoding1, encoding2)
    print(f"La misma persona codificada dos veces con el diccionario reiniciado - Los vectores son iguales: {are_equal}")

    if not are_equal:
        diff_count = np.sum(encoding1 != encoding2)
        print(f"Número de elementos diferentes: {diff_count} de {DIMENSION}")
    else:
        print("La codificación determinista funciona correctamente!")

    test_person2 = test_person.copy()
    test_person2["name"] = "Different"

    hv_dict = {}
    encoding3 = encode_person(test_person2)

    are_different = not np.array_equal(encoding1, encoding3)
    print(f"Diferentes personas producen diferentes encodings: {are_different}")

    return are_equal, are_different


def test_db_encoding_preservation():
    """Los datos personales se puedan almacenar y recuperar de Milvus con el encoding preservado."""
    print("\n--- Testing Milvus Encoding Preservation ---")

    # Test person (las listas las guardo en 'attrs')
    test_person = {
        "name": "John",
        "lastname": "Doe",
        "dob": "1990-05-15",  # ISO string is fine; normalize() will parse it
        "marital_status": "Married",
        "mobile_number": "555-1111",
        "gender": "Male",
        "race": "Caucasian",
        "attrs": {  # <-- acá viven las listas
            "address": ["456 Main St", "Apt 789"],
            "akas": ["Johnny", "J.D."],
            "landlines": ["555-9876"]
        }
    }

    # Normalización ( el string de 'dob' a date, etc.)
    test_person = normalize_person_data(test_person)

    print("Datos originales de la persona:")
    for k, v in test_person.items():
        print(f"  {k}: {v}")

    global hv_dict
    hv_dict = {}

    # Encode de manera local (ésta es la referencia)
    original_encoding = encode_person(test_person)
    print(f"\nEncoding original (primeros 5 elementos): {original_encoding[:5]}")

    # Insertar en Milvus (devuelve PK id)
    person_id = store_person(test_person)
    print(f"\nPersona guardada en Milvus con ID: {person_id}")

    # Leer de Milvus (incluyendo hv) la persona que recién guardé
    stored = _load_person_from_milvus(person_id, with_hv=True)
    if not stored:
        print("ERROR: Persona no se encuentra en Milvus!")
        return False

    print("\nDatos de persona recuperados (escalares):")
    for key in ["name", "lastname", "dob", "marital_status", "mobile_number", "gender", "race", "attrs"]:
        print(f"  {key}: {stored.get(key)}")

    # Reconstrucción del diccionario de personas exactamente como encode_person() espera
    attrs = stored.get("attrs") or {}
    normalized_retrieved = {
        "name": stored.get("name", ""),
        "lastname": stored.get("lastname", ""),
        "dob": parse_date(stored.get("dob")),
        "marital_status": stored.get("marital_status", ""),
        "mobile_number": stored.get("mobile_number", ""),
        "gender": stored.get("gender", ""),
        "race": stored.get("race", ""),
        "attrs": attrs,  # pass through as-is; do NOT explode to top-level
    }

    print("\nNormalized retrieved person data:")
    for k, v in normalized_retrieved.items():
        print(f"  {k}: {v}")

    # Recalcular encoding a partir del diccionario recuperado (hacer esto ANTES de las comparaciones)
    hv_dict = {}
    recomputed_encoding = encode_person(normalized_retrieved)

    if VECTOR_MODE.lower() == "binary":
        stored_encoding = _unpack_binary(stored["hv"], DIMENSION).astype(np.uint8)
        # chequeos binarios
        assert np.all((original_encoding == 0) | (original_encoding == 1))
        assert np.all((recomputed_encoding == 0) | (recomputed_encoding == 1))
        assert np.all((stored_encoding == 0) | (stored_encoding == 1))
    else:  # "bipolar"
        stored_encoding = np.asarray(stored["hv"]).astype(np.int8)
        stored_encoding = np.where(stored_encoding > 0, 1, -1)
        # chequeos bipolares
        assert np.all((original_encoding == -1) | (original_encoding == 1))
        assert np.all((recomputed_encoding == -1) | (recomputed_encoding == 1))
        assert np.all((stored_encoding == -1) | (stored_encoding == 1))

    # Ahora stored_encoding está en binario (0/1), directamente comparable a original_encoding
    stored_vs_original = np.array_equal(stored_encoding, original_encoding)
    recomputed_vs_stored = np.array_equal(stored_encoding, recomputed_encoding)


    # Ahora original_encoding y recomputed_encoding deberían ser binarios (0/1)
    # y stored_encoding de Milvus se descomprmió en binario también

    if VECTOR_MODE.lower() == "binary":
        # Milvus devuelve hv empaquetado o uint8 -> desempaquetar a {0,1}
        stored_encoding = _unpack_binary(stored["hv"], DIMENSION).astype(np.uint8)

        # Chequeos binarios
        assert np.all((original_encoding == 0) | (original_encoding == 1)), "Original encoding is not binary"
        assert np.all((recomputed_encoding == 0) | (recomputed_encoding == 1)), "Recomputed encoding is not binary"
        assert np.all((stored_encoding == 0) | (stored_encoding == 1)), "Stored encoding is not binary"

    else:  # "bipolar"
        # Milvus devolvió lista/array con -1/+1 (floats o ints)
        stored_encoding = np.asarray(stored["hv"])
        # Normalizamos a -1/+1 enteros de forma robusta (por si vino como float +-1.0)
        stored_encoding = np.where(stored_encoding > 0, 1, -1).astype(np.int8)

        # Chequeos bipolares
        assert np.all((original_encoding == -1) | (original_encoding == 1)), "Original encoding is not bipolar"
        assert np.all((recomputed_encoding == -1) | (recomputed_encoding == 1)), "Recomputed encoding is not bipolar"
        assert np.all((stored_encoding == -1) | (stored_encoding == 1)), "Stored encoding is not bipolar"

    # Ahora podemos compararlos directamente
    stored_vs_original = np.array_equal(stored_encoding, original_encoding)
    recomputed_vs_stored = np.array_equal(stored_encoding, recomputed_encoding)

    print(f"\nEncoding del almacenado (primeros 5): {stored_encoding[:5]}")
    print(f"Encoding original (primeros 5): {original_encoding[:5]}")
    print(f"Encoding recalculado (primeros 5): {recomputed_encoding[:5]}")
    print(f"\nEncoding almacenado coincide con el original: {stored_vs_original}")
    print(f"Encoding recalculado coincide con el almacenado: {recomputed_vs_stored}")

    if not stored_vs_original:
        diff_count = np.sum(stored_encoding != original_encoding)
        print(f"Hay diferencias entre el almacenado y el original: {diff_count} / {DIMENSION}")

    if not recomputed_vs_stored:
        diff_count = np.sum(stored_encoding != recomputed_encoding)
        print(f"Hay diferencias entre el recalculado y el almacenado: {diff_count} / {DIMENSION}")

    # Limpiamos la fila insertada
    _delete_people_from_milvus([person_id])

    return stored_vs_original and recomputed_vs_stored


def test_search_with_encoded_vector():
    """Búsqueda de persona usando un vector codificado con pequeñas variaciones(Milvus backend)"""
    print("\n--- Búsqueda con Vector Codificado (Milvus) ---")

    original_person = {
        "name": "Jane",
        "lastname": "Smith",
        "dob": "1985-08-23",
        "marital_status": "Single",
        "mobile_number": "555-8765",
        "gender": "Female",
        "race": "Asian",
        "attrs": {
            "address": ["789 Oak Rd", "Suite 456"],
            "akas": ["J. Smith", "Janie"],
            "landlines": ["555-4321"]
        }
    }

    global hv_dict
    hv_dict = {}
    person_id = store_person(original_person)
    print(f"Persona original almacenada en Milvus con ID: {person_id}")
    for k, v in original_person.items():
        print(f"  {k}: {v}")

    query_person = original_person.copy()
    query_person["lastname"] = "Smyth"             # typo
    query_person["dob"] = "1985-8-25"        # 2 days off
    query_person["address"] = ["789 Oak Road"]     # slight change
    query_person["akas"] = ["J. Smith"]            # subset

    print("\nPersona utilizada para la búsqueda (con pequeñas variaciones:")
    for k, v in query_person.items():
        print(f"  {k}: {v}")

    # find_closest_match_db ahora debería buscar en Milvus
    matches = find_closest_match_db(query_person, threshold=0.5) # experimentar con este threshold para rsultados mas exactos

    print("\nResultados de búsqueda de persona con vector codificado (Milvus):")
    for match in matches:
        print(f"Coincidencia encontrada: {match}")

    is_correct_match = any(match.get("id") == person_id for match in matches)
    print(f"Persona correcta encontrada: {is_correct_match}")

    different_person = {
        "name": "Bob",
        "lastname": "Johnson",
        "dob": "1970-01-01",
        "address": ["123 Different St"],
        "marital_status": "Married",
        "akas": ["Robert"],
        "landlines": ["555-9999"],
        "mobile_number": "555-0000",
        "gender": "Male",
        "race": "Caucasian"
    }

    print("\nBúsqueda con persona completamente distinta:")
    for k, v in different_person.items():
        print(f"  {k}: {v}")

    different_matches = find_closest_match_db(different_person, threshold=0.5)

    print("\nResultados para la persona distinta:")
    if different_matches:
        for match in different_matches:
            print(f"Coincidencia encontrada: {match}")

        original_similarity = next((m["similarity"] for m in matches if m.get("id") == person_id), 0)
        different_similarity = next((m["similarity"] for m in different_matches if m.get("id") == person_id), 0)
        lower_similarity = different_similarity < original_similarity
        print(f"La persona diferente tiene menos similaridad que la original (esperado): {lower_similarity}")
    else:
        print("No se encontraron coincidencias para la persona distinta (esperado)")
        lower_similarity = True

    # Limpiamos DB
    _delete_people_from_milvus([person_id])

    return is_correct_match and lower_similarity

def test_date_encoding_and_search():
    """Encoding de fecha y funcionalidad de búsqueda basada en fecha (Milvus backend)"""
    print("\n--- Encoding y búsqueda de fechas (Milvus) ---")
    # Limpiamos la base de datos antes de empezar
    _delete_people_from_milvus([])

    test_people = [
        {"name": "Person", "lastname": "One", "dob": "1990-05-15", "gender": "Male"},
        {"name": "Person", "lastname": "Two", "dob": "1990-05-20", "gender": "Female"},
        {"name": "Person", "lastname": "Three", "dob": "1990-06-15", "gender": "Other"},
        {"name": "Person", "lastname": "Four", "dob": "1991-05-15", "gender": "Male"},
    ]

    ids = []
    for p in test_people:
        pid = store_person(p)
        ids.append(pid)
        print(f"Guardando {p['name']} {p['lastname']} (DOB: {p['dob']}) con ID: {pid}")

    print("\nBuscando personas nacidas en mayo de 1990 (dentro de los 15 días del 15 de mayo):")
    matches = find_similar_by_date(date(1990, 5, 15), range_days=15)
    for match in matches:
        print(f"  Encontrado: {match['name']} {match['lastname']} (DOB: {match['dob']})")

    found_ids = [m["id"] for m in matches]
    expected_ids = [ids[0], ids[1]]
    correct_date_matches = all(i in found_ids for i in expected_ids) and len(found_ids) == len(expected_ids)
    print(f"Se encontraron coincidencias de fecha correctas: {correct_date_matches}")

    print("\nPrueba de similitud de encoding de fechas:")
    mode = VECTOR_MODE.lower()

    enc1 = encode_date(date(1990, 5, 15), mode=mode)
    enc2 = encode_date(date(1990, 5, 16), mode=mode)  # 1 day
    enc3 = encode_date(date(1990, 6, 15), mode=mode)  # 1 month
    enc4 = encode_date(date(1991, 5, 15), mode=mode)  # 1 year

    sim_1_day = cosine_similarity([enc1], [enc2])[0][0]
    sim_1_month = cosine_similarity([enc1], [enc3])[0][0]
    sim_1_year = cosine_similarity([enc1], [enc4])[0][0]

    print(f"Similaridad con 1 día de diferencia: {sim_1_day:.4f}")
    print(f"Similaridad con 1 mes de diferencia: {sim_1_month:.4f}")
    print(f"Similaridad con 1 año de diferencia: {sim_1_year:.4f}")

    correct_similarity_order = sim_1_day > sim_1_month > sim_1_year
    print(f"Orden de similitud correcto (las fechas más cercanas tienen mayor similitud): {correct_similarity_order}")

    # Clean up
    _delete_people_from_milvus(ids)

    return correct_date_matches and correct_similarity_order


if __name__ == "__main__":
    consistency_result, differentiation_result = test_encoding_consistency()
    db_preservation_result = test_db_encoding_preservation()
    search_result = test_search_with_encoded_vector()
    date_result = test_date_encoding_and_search()

    print("\n\033[94m--- Resultados de ejecución de tests: ---\033[0m")

    if consistency_result and differentiation_result:
        print("\033[92m ✓ Test de consistencia de Encoding: PASS\033[0m")
    else:
        print("\033[91m✗ Algunos tests de consistencia de encoding fallaron: \033[0m")
        if not consistency_result:
            print("  - Test de consistencia FALLÓ: Los mismos datos produjeron distintos encodings")
        if not differentiation_result:
            print("  - Test de diferenciación FALLÓ: Datos diferentes produjeron el mismo encoding")

    if db_preservation_result:
        print("\033[92m ✓ Test de preservación de encoding en la BD: PASS\033[0m")
    else:
        print("\033[91m✗ Test de preservación de encoding en la BD: FALLÓ\033[0m")

    if search_result:
        print("\033[92m ✓ Búsqueda basada en vector: PASS\033[0m")
    else:
        print("\033[91m✗ Búsqueda basada en vector: FALLÓ\033[0m")


    if date_result:
        print("\033[92m ✓ Tests de Encoding y búsqueda de fechas: PASS\033[0m")
    else:
        print("\033[91m✗ Tests de Encoding y búsqueda de fechas: FALLÓ\033[0m")

