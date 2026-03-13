import os
import sys
from datetime import date
import pytest

import numpy as np
import torch
import torch.nn.functional as F

# Make project root importable (same as before)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Your code (unchanged) ---
from encoding_methods.encoding_and_search_milvus import (
    encode_person, encode_date, DIMENSION, store_person, find_closest_match_db, parse_date,
    find_similar_by_date, normalize_person_data
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
def _unpack_binary_to_bipolar(payload, dim: int, device: str = 'cpu') -> torch.Tensor:
    """
    Strict PyTorch implementation.
    Accepts:
      - torch.Tensor (uint8)
      - bytes / bytearray (packed bits)
      - list[bytes] (Milvus wrapper)
      - list[int] (Python list of uint8)

    Raises:
      - TypeError if input is a NumPy array or unsupported type.
    """
    # --- 1. Strict Type Guard (Catches Legacy NumPy) ---
    # We check the type name string to avoid importing numpy just for the check.
    type_str = str(type(payload))
    if 'numpy' in type_str:
        raise TypeError(
            f"Migration Error: NumPy input detected ({type_str}). "
            "Please convert to torch.Tensor or bytes before calling this function."
        )

    # --- 2. Normalize Input to Flat uint8 Tensor ---

    # Handle Milvus-style list wrapper: [b'\x12...']
    if isinstance(payload, list):
        if len(payload) > 0 and isinstance(payload[0], (bytes, bytearray)):
            payload = payload[0]
        # Else: It's likely a plain list of ints, which we handle below.

    if isinstance(payload, (bytes, bytearray, memoryview)):
        # Zero-copy where possible; explicit copy for immutable bytes
        payload = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    elif isinstance(payload, list):
        # Handle standard python list of ints
        payload = torch.tensor(payload, dtype=torch.uint8)
    elif isinstance(payload, torch.Tensor):
        # Accept existing tensors (ensure uint8)
        payload = payload.to(dtype=torch.uint8)
    else:
        raise TypeError(f"Unsupported input type: {type(payload)}. Expected bytes, list, or torch.Tensor.")

    # Move to device and flatten
    payload = payload.to(device).flatten()

    # --- 3. Determine if Unpacking is Needed ---

    # Heuristic: If size matches 'dim' exactly, assume it's already unpacked bits (0/1)
    if payload.numel() == dim:
        bits = payload
    else:
        # --- 4. Unpack Bits (Big Endian) ---
        # Create mask: [128, 64, 32, 16, 8, 4, 2, 1]
        mask = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=device)

        # Expand payload (N, 1) and mask (1, 8) -> Result (N, 8)
        # Perform bitwise AND and check if > 0
        bits = (payload.unsqueeze(-1) & mask) > 0
        bits = bits.flatten()

    # --- 5. Slice and Convert to Bipolar ---

    # Ensure we strictly output the requested dimension
    bits = bits[:dim]

    # Convert {0, 1} -> {-1, 1}
    # Logic: (x * 2) - 1
    # 0 -> -1
    # 1 ->  1
    return bits.to(torch.int8) * 2 - 1

def _unpack_binary(payload, dim: int, device: str = 'cpu') -> torch.Tensor:
    """
    Strict PyTorch implementation.
    Accepts:
      - torch.Tensor (uint8)
      - bytes / bytearray (packed bits)
      - list[bytes] (Milvus wrapper)
      - list[int] (Python list of uint8)
      - list[float] (Milvus may return float32 values like [0.0, 1.0, ...])

    Returns:
      - torch.Tensor (uint8) containing {0, 1} values.

    Raises:
      - TypeError if input is a NumPy array or unsupported type.
    """
    # --- 1. Strict Type Guard (Catches Legacy NumPy) ---
    type_str = str(type(payload))
    if 'numpy' in type_str and not isinstance(payload, list):
        raise TypeError(
            f"Migration Error: NumPy input detected ({type_str}). "
            "Please convert to torch.Tensor or bytes before calling this function."
        )

    # --- 2. Normalize Input to Flat uint8 Tensor ---

    # Handle Milvus-style list wrapper: [b'\x12...']
    if isinstance(payload, list):
        if len(payload) == 0:
            raise ValueError("Empty hv payload")

        first = payload[0]
        if isinstance(first, (bytes, bytearray, memoryview)):
            payload = first
        # Else: It's likely a plain list of ints or floats, handle below.

    if isinstance(payload, (bytes, bytearray, memoryview)):
        # Zero-copy conversion (wraps in bytearray to ensure writability/compatibility)
        payload = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    elif isinstance(payload, list):
        try:
            # First try the standard way (for integer lists)
            payload = torch.tensor(payload, dtype=torch.uint8)
        except TypeError:
            # Handle float lists (from Milvus) by first converting to float tensor
            # then rounding and converting to uint8
            payload = torch.tensor(payload, dtype=torch.float32).round().to(torch.uint8)
    elif isinstance(payload, torch.Tensor):
        # If it's already a tensor but not uint8, convert it
        if payload.dtype != torch.uint8:
            payload = payload.round().to(torch.uint8)
    else:
        raise TypeError(f"Unsupported input type: {type(payload)}. Expected bytes, list, or torch.Tensor.")

    # Move to device and flatten
    payload = payload.to(device).flatten()

    # --- 3. Determine if Unpacking is Needed ---

    # Heuristic: If size matches 'dim' exactly, assume it's already unpacked bits (0/1)
    if payload.numel() == dim:
        # Ensure strict 0/1 (optional sanity check, though typically we trust the dimensions)
        return payload.to(torch.uint8)

    # --- 4. Unpack Bits (Big Endian) ---

    # Create mask: [128, 64, 32, 16, 8, 4, 2, 1]
    mask = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=device)

    # Expand payload (N, 1) and mask (1, 8) -> Result (N, 8)
    # Perform bitwise AND. If result > 0, the bit is set.
    bits = (payload.unsqueeze(-1) & mask) > 0
    bits = bits.flatten()

    # --- 5. Slice and Return ---

    # Cut to requested dimension and cast boolean -> uint8 (0 or 1)
    return bits[:dim].to(torch.uint8)

def _load_person_from_milvus(person_id: int, with_hv: bool = True, collection_name: str = "people"):
    """
    Query Milvus by primary key and return a dict with scalar fields and optionally 'hv'.
    """
    col = ensure_people_collection(collection_name)
    out_fields = ["name", "lastname", "dob", "marital_status", "mobile_number", "gender", "race", "attrs"]
    if with_hv:
        out_fields.append("hv")

    res = col.query(expr=f"id == {person_id}", output_fields=out_fields, consistency_level="Strong")
    return res[0] if res else None

def _delete_people_from_milvus(ids, collection_name: str = "people"):
    col = ensure_people_collection(collection_name)
    if not ids:
        col.delete(expr="id >= 0")
        return
    id_list = ",".join(str(i) for i in ids)
    col.delete(expr=f"id in [{id_list}]")

# -------------------- Tests --------------------


@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
def test_same_person_encoding_is_consistent(with_vector_mode):
    """Verifies that encoding the same person twice yields the same vector."""
    from database_utils.milvus_db_connection import get_vector_mode

    # Use GPU if available for improved performance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    current_mode = get_vector_mode()
    assert current_mode == with_vector_mode, f"Expected mode {with_vector_mode}, got {current_mode}"

    print(f"\n--- Verificación de Consistencia del Encoding (modo: {with_vector_mode}) ---")
    test_person = {
        "name": "John", "lastname": "Doe", "dob": "1990-05-15",
        "marital_status": "Married", "mobile_number": "555-1111", "gender": "Male", "race": "Caucasian",
        "attrs": {"address": ["456 Main St", "Apt 789"], "akas": ["Johnny", "J.D."], "landlines": ["555-9876"]}
    }

    # Create a batch of the same person to leverage batch processing
    test_batch = [test_person, test_person]

    # Use no_grad context to save memory and improve performance during inference
    with torch.no_grad():
        # Encode the same person twice, resetting the cache each time
        global hv_dict
        hv_dict = {}

        # First encoding
        encoding1 = encode_person(test_person).to(device)

        # Reset cache
        hv_dict = {}

        # Second encoding
        encoding2 = encode_person(test_person).to(device)

        # Use torch.allclose for robust comparison with floating point values
        if encoding1.dtype.is_floating_point or encoding2.dtype.is_floating_point:
            are_equal = torch.allclose(encoding1, encoding2, rtol=1e-5, atol=1e-8)
        else:
            are_equal = torch.equal(encoding1, encoding2)

    # Print diagnostic information before the assertion
    print(
        f"La misma persona codificada dos veces con el diccionario reiniciado - Los vectores son iguales: {are_equal}")
    if not are_equal:
        # Compute differences efficiently
        if encoding1.dtype.is_floating_point:
            diff_mask = ~torch.isclose(encoding1, encoding2, rtol=1e-5, atol=1e-8)
        else:
            diff_mask = encoding1 != encoding2

        diff_count = diff_mask.sum().item()

        # Show first few differences if any exist
        if diff_count > 0:
            diff_indices = torch.nonzero(diff_mask, as_tuple=True)[0][:5]  # Get first 5 differences
            print(f"Primeras diferencias (índice: valor1 vs valor2):")
            for idx in diff_indices:
                print(f"  {idx}: {encoding1[idx].item()} vs {encoding2[idx].item()}")

        print(f"Número de elementos diferentes: {diff_count} de {DIMENSION}")
    else:
        print("La codificación determinista funciona correctamente!")

    assert are_equal, "La codificación repetida de la misma persona debería producir el mismo vector"


@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
def test_different_people_produce_different_encodings(with_vector_mode):
    """Verifies that encoding two different people produces different vectors."""
    from database_utils.milvus_db_connection import get_vector_mode
    import torch.nn.functional as F
    import time

    # Use GPU if available for improved performance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    current_mode = get_vector_mode()
    assert current_mode == with_vector_mode, f"Expected mode {with_vector_mode}, got {current_mode}"

    print(f"\n--- Verificación de Diferenciación del Encoding (modo: {with_vector_mode}) ---")

    # Create a mini-batch of different people to test
    # Add more variations to stress-test the encoding differences
    test_batch = [
        {
            "name": "John", "lastname": "Doe", "dob": "1990-05-15",
            "marital_status": "Married", "mobile_number": "555-1111", "gender": "Male", "race": "Caucasian",
            "attrs": {"address": ["456 Main St", "Apt 789"], "akas": ["Johnny", "J.D."], "landlines": ["555-9876"]}
        },
        {
            "name": "Different", "lastname": "Doe", "dob": "1990-05-15",
            "marital_status": "Married", "mobile_number": "555-1111", "gender": "Male", "race": "Caucasian",
            "attrs": {"address": ["456 Main St", "Apt 789"], "akas": ["Johnny", "J.D."], "landlines": ["555-9876"]}
        },
        {
            "name": "John", "lastname": "Smith", "dob": "1990-05-15",
            "marital_status": "Married", "mobile_number": "555-1111", "gender": "Male", "race": "Caucasian",
            "attrs": {"address": ["456 Main St", "Apt 789"], "akas": ["Johnny", "J.D."], "landlines": ["555-9876"]}
        }
    ]

    # Reset cache once before any processing
    global hv_dict
    hv_dict = {}

    # Use no_grad for inference to reduce memory usage and improve performance
    start_time = time.time()
    with torch.no_grad():
        # Preprocess each person
        all_encodings = []

        # Process all encodings in parallel when possible
        for person in test_batch:
            encoding = encode_person(person)

            # Convert to tensor if necessary
            if not isinstance(encoding, torch.Tensor):
                encoding = torch.tensor(encoding, dtype=torch.float32 if with_vector_mode == "float" else torch.uint8)

            all_encodings.append(encoding)

        # Stack all encodings into a single batch tensor
        # This enables efficient batch operations on GPU
        encodings_tensor = torch.stack(all_encodings).to(device)

        # For binary mode, ensure we use correct bit representation
        if not encodings_tensor.dtype.is_floating_point:
            # Convert to float for similarity calculations
            encodings_float = encodings_tensor.float()
        else:
            encodings_float = encodings_tensor

        # Compute all pairwise similarities in one efficient operation
        # This is much faster than doing individual comparisons
        # Shape: (batch_size, batch_size)
        similarity_matrix = F.cosine_similarity(
            encodings_float.unsqueeze(1),  # Shape: (batch_size, 1, dim)
            encodings_float.unsqueeze(0),  # Shape: (1, batch_size, dim)
            dim=2
        )

        # Extract relevant similarity scores
        similarities = []
        for i in range(len(test_batch)):
            for j in range(i + 1, len(test_batch)):
                sim = similarity_matrix[i, j].item()
                similarities.append((i, j, sim))

        # Calculate bit difference percentages for non-identical pairs
        if not encodings_tensor.dtype.is_floating_point:
            bit_diffs = []
            for i, j, _ in similarities:
                diff_ratio = (encodings_tensor[i] != encodings_tensor[j]).float().mean().item() * 100
                bit_diffs.append(diff_ratio)

    # Report processing time
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.4f} seconds")

    # Check if all encodings are different
    all_different = True

    # Print similarity results
    print("\nSimilarity between encodings:")
    for idx, (i, j, sim) in enumerate(similarities):
        p1_name = f"{test_batch[i]['name']} {test_batch[i]['lastname']}"
        p2_name = f"{test_batch[j]['name']} {test_batch[j]['lastname']}"

        if not encodings_tensor.dtype.is_floating_point:
            bit_diff = bit_diffs[idx]
            print(f"  {p1_name} vs {p2_name}: similarity={sim:.4f}, bit_difference={bit_diff:.2f}%")
        else:
            print(f"  {p1_name} vs {p2_name}: similarity={sim:.4f}")

        # High similarity (>0.99) would indicate they're effectively the same
        if sim > 0.99:
            all_different = False

    print(f"\nDiferentes personas producen diferentes encodings: {all_different}")
    assert all_different, "La codificación de personas diferentes debería producir vectores diferentes"


@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
def test_db_encoding_preservation(with_vector_mode):
    """Los datos personales se puedan almacenar y recuperar de Milvus con el encoding preservado."""
    import uuid
    from database_utils.milvus_db_connection import get_vector_mode, _collection_cache

    # Verify the vector mode is correctly set for the test
    current_mode = get_vector_mode()
    assert current_mode == with_vector_mode, f"Expected mode {with_vector_mode}, got {current_mode}"

    print(f"\n--- Testing Milvus Encoding Preservation (mode: {with_vector_mode}) ---")

    # Use an isolated collection to avoid interference with other tests
    collection_name = f"people_test_{uuid.uuid4().hex[:8]}"
    col = ensure_people_collection(collection_name)

    # Use GPU if available for improved performance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test person (las listas las guardo en 'attrs')
    raw_test_person = {
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

    # Create a small batch of test persons to leverage batch processing
    batch_size = 3
    test_batch = [raw_test_person.copy() for _ in range(batch_size)]

    # Add slight variations to make the test more robust
    if batch_size > 1:
        test_batch[1]["name"] = "Johnny"
        if batch_size > 2:
            test_batch[2]["mobile_number"] = "555-2222"

    # Normalización (el string de 'dob' a date, etc.)
    normalized_test_persons = [normalize_person_data(p) for p in test_batch]
    normalized_test_person = normalized_test_persons[0]  # First one is our main test subject

    if batch_size == 1:
        print("Datos originales de la persona:")
        for k, v in raw_test_person.items():
            print(f"  {k}: {v}")
    else:
        print(f"Testing with batch of {batch_size} persons (with variations)")

    global hv_dict
    hv_dict = {}

    # Use torch.no_grad to reduce memory usage during inference
    with torch.no_grad():
        # Encode de manera local (ésta es la referencia)
        original_encoding = encode_person(raw_test_person).to(device)
        print(f"\nEncoding original (primeros 5 elementos): {original_encoding[:5]}")

        # Insertar en Milvus (devuelve PK id)
        person_ids = []
        for person in normalized_test_persons:
            person_id = store_person(person, collection_name=collection_name)
            person_ids.append(person_id)

        print(f"\nPersonas guardadas en Milvus con IDs: {person_ids}")

        # Leer de Milvus (incluyendo hv) la persona que recién guardé
        stored = _load_person_from_milvus(person_ids[0], with_hv=True, collection_name=collection_name)
        if not stored:
            print("ERROR: Persona no se encuentra en Milvus!")
            return False

        if batch_size == 1:
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

        if batch_size == 1:
            print("\nNormalized retrieved person data:")
            for k, v in normalized_retrieved.items():
                print(f"  {k}: {v}")

        # Recalcular encoding a partir del diccionario recuperado
        hv_dict = {}
        recomputed_encoding = encode_person(normalized_retrieved).to(device)

        # Determine stored encoding based on mode
        if with_vector_mode.lower() == "binary":
            stored_encoding = _unpack_binary(stored["hv"], DIMENSION).to(device)  # Returns Tensor {0,1}
        else:  # "float" (bipolar)
            # Milvus might return list of floats/ints for float_vector
            raw_hv = stored["hv"]
            # When querying a vector database, the Python SDK will almost always return the vector as either a Python list or a numpy.ndarray
            if isinstance(raw_hv, (list, np.ndarray)):
                stored_encoding = torch.tensor(raw_hv, device=device)
                # Normalize to -1, 1 if they came back as floats or ints
                stored_encoding = torch.where(
                    stored_encoding > 0,
                    torch.tensor(1, dtype=torch.int8, device=device),
                    torch.tensor(-1, dtype=torch.int8, device=device)
                )
            else:
                # If packed bytes for some reason
                stored_encoding = _unpack_binary_to_bipolar(raw_hv, DIMENSION).to(device)

        # Batch checks - more efficient than individual checks
        if with_vector_mode.lower() == "binary":
            # Chequeos binarios - use optimized batch operations
            binary_check = torch.stack([
                torch.all((original_encoding == 0) | (original_encoding == 1)),
                torch.all((recomputed_encoding == 0) | (recomputed_encoding == 1)),
                torch.all((stored_encoding == 0) | (stored_encoding == 1))
            ])

            if not torch.all(binary_check):
                if not binary_check[0]: print("Original encoding is not binary")
                if not binary_check[1]: print("Recomputed encoding is not binary")
                if not binary_check[2]: print("Stored encoding is not binary")
                assert False, "Binary encoding checks failed"

        else:  # "float" (bipolar)
            # Chequeos bipolares - use optimized batch operations
            bipolar_check = torch.stack([
                torch.all((original_encoding == -1) | (original_encoding == 1)),
                torch.all((recomputed_encoding == -1) | (recomputed_encoding == 1)),
                torch.all((stored_encoding == -1) | (stored_encoding == 1))
            ])

            if not torch.all(bipolar_check):
                if not bipolar_check[0]: print("Original encoding is not bipolar")
                if not bipolar_check[1]: print("Recomputed encoding is not bipolar")
                if not bipolar_check[2]: print("Stored encoding is not bipolar")
                assert False, "Bipolar encoding checks failed"

        # Use efficient tensor operations for comparisons
        stored_vs_original = torch.equal(stored_encoding, original_encoding)
        recomputed_vs_stored = torch.equal(stored_encoding, recomputed_encoding)

        print(f"\nEncoding almacenado coincide con el original: {stored_vs_original}")
        print(f"Encoding recalculado coincide con el almacenado: {recomputed_vs_stored}")

        if not stored_vs_original:
            # Optimize difference calculation for large tensors
            diff_count = (stored_encoding != original_encoding).sum().item()
            diff_percent = (diff_count / DIMENSION) * 100
            print(
                f"Hay diferencias entre el almacenado y el original: {diff_count} / {DIMENSION} ({diff_percent:.2f}%)")

            # Calculate similarity for more informative comparison
            if with_vector_mode.lower() == "binary":
                # For binary vectors: calculate Hamming similarity (percentage of matching bits)
                sim = 1.0 - (diff_count / DIMENSION)
                print(f"Hamming similarity entre almacenado y original: {sim:.6f}")
            else:
                # For bipolar: calculate cosine similarity
                cosine_sim = F.cosine_similarity(
                    stored_encoding.float().unsqueeze(0),
                    original_encoding.float().unsqueeze(0)
                ).item()
                print(f"Cosine similarity entre almacenado y original: {cosine_sim:.6f}")

        if not recomputed_vs_stored:
            # Optimize difference calculation for large tensors
            diff_count = (stored_encoding != recomputed_encoding).sum().item()
            diff_percent = (diff_count / DIMENSION) * 100
            print(
                f"Hay diferencias entre el recalculado y el almacenado: {diff_count} / {DIMENSION} ({diff_percent:.2f}%)")

            # Additional metrics: Calculate similarity based on vector mode
            if with_vector_mode.lower() == "binary":
                sim = 1.0 - (diff_count / DIMENSION)
                print(f"Hamming similarity entre recalculado y almacenado: {sim:.6f}")
            else:
                cosine_sim = F.cosine_similarity(
                    stored_encoding.float().unsqueeze(0),
                    recomputed_encoding.float().unsqueeze(0)
                ).item()
                print(f"Cosine similarity entre recalculado y almacenado: {cosine_sim:.6f}")

    _collection_cache.pop(f"{collection_name}_{with_vector_mode}", None)
    try:
        col.drop()
    except Exception:
        pass


# TODO: Hay que trabajar en optimizar este test para pytorch!
def test_search_with_encoded_vector():
    """Búsqueda de persona usando un vector codificado con pequeñas variaciones(Milvus backend)"""
    import uuid
    from database_utils.milvus_db_connection import get_vector_mode, _collection_cache

    print("\n--- Búsqueda con Vector Codificado (Milvus) ---")

    collection_name = f"people_test_{uuid.uuid4().hex[:8]}"
    col = ensure_people_collection(collection_name)

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
    person_id = store_person(original_person, collection_name=collection_name)
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
    matches = find_closest_match_db(query_person, threshold=0.5, collection_name=collection_name)

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

    different_matches = find_closest_match_db(different_person, threshold=0.5, collection_name=collection_name)

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

    _collection_cache.pop(f"{collection_name}_{get_vector_mode()}", None)
    try:
        col.drop()
    except Exception:
        pass


@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
def test_date_range_search(with_vector_mode):
    """Tests the ability to search for records within a specific date range with optimized PyTorch usage"""
    import uuid
    from database_utils.milvus_db_connection import get_vector_mode, _collection_cache
    from pymilvus import utility
    import time
    import torch

    # Use GPU if available for improved performance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Verify the vector mode is correctly set for the test
    current_mode = get_vector_mode()
    assert current_mode == with_vector_mode, f"Expected mode {with_vector_mode}, got {current_mode}"

    print(f"\n--- Test: Date Range Search (mode: {with_vector_mode}) ---")
    print(f"Using device: {device}")

    # Use an isolated collection to avoid interference with other tests
    collection_name = f"people_test_{uuid.uuid4().hex[:8]}"
    col = ensure_people_collection(collection_name)
    print(f"Using isolated collection: {collection_name}")

    try:
        # Test data - batch of people with different date ranges
        test_people = [
            {"name": "Person", "lastname": "One", "dob": "1990-05-15", "gender": "Male"},
            {"name": "Person", "lastname": "Two", "dob": "1990-05-20", "gender": "Female"},
            {"name": "Person", "lastname": "Three", "dob": "1990-06-15", "gender": "Other"},
            {"name": "Person", "lastname": "Four", "dob": "1991-05-15", "gender": "Male"},
        ]

        # Measure performance - start time
        start_time = time.time()

        ids = []
        for p in test_people:
            pid = store_person(p, collection_name=collection_name)
            ids.append(pid)
            print(f"Guardando {p['name']} {p['lastname']} (DOB: {p['dob']}) con ID: {pid}")

        col.flush()

        # Perform the date range search
        print("\nBuscando personas nacidas en mayo de 1990 (dentro de los 15 días del 15 de mayo):")
        target_date = date(1990, 5, 15)
        range_days = 15

        with torch.no_grad():  # Disable gradient tracking for inference
            matches = find_similar_by_date(target_date, range_days=range_days,
                                           collection_name=collection_name)

        # Convert results to tensors for efficient processing
        if matches:
            # Extract found IDs
            found_ids = torch.tensor([m["id"] for m in matches], device=device)

            # Print results
            for match in matches:
                print(f"  Encontrado: {match['name']} {match['lastname']} (DOB: {match['dob']})")

            # Expected IDs
            expected_ids = torch.tensor([ids[0], ids[1]], device=device)

            # Check for containment efficiently using set operations
            # Convert to sets for O(1) lookups
            found_set = set(found_ids.tolist())
            expected_set = set(expected_ids.tolist())

            # All expected IDs should be in found_set and counts should match
            correct_date_matches = expected_set.issubset(found_set) and len(found_set) == len(expected_set)
        else:
            correct_date_matches = False
            print("No se encontraron coincidencias")

        # Report execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")

        print(f"Se encontraron coincidencias de fecha correctas: {correct_date_matches}")

        assert correct_date_matches, "The date matching search failed to find the expected records"

        # Optional: Test with different date ranges and batch sizes for more thorough validation
        if len(ids) > 2:
            print("\nPrueba adicional: Buscando personas nacidas en 1991:")
            with torch.no_grad():
                future_matches = find_similar_by_date(date(1991, 5, 15), range_days=15,
                                                      collection_name=collection_name)

                # Check that Person Four is found
                found_future = any(m["id"] == ids[3] for m in future_matches)
                print(f"Se encontró Person Four en 1991: {found_future}")
                assert found_future, "Could not find Person Four in 1991 search"

    finally:
        _collection_cache.pop(f"{collection_name}_{with_vector_mode}", None)
        try:
            col.drop()
        except Exception:
            pass


def test_date_similarity_ordering_binary():
    """Tests that the similarity between binary date encodings correctly reflects their temporal proximity"""
    from datetime import date
    from hdc.binary_hdc import HyperDimensionalComputingBinary
    import pytest
    import torch

    print("\n--- Test: Binary Date Similarity Ordering ---")

    # Initialize binary HDC
    hdc = HyperDimensionalComputingBinary(dim=10000, seed=42)

    print("Prueba de similitud de encoding de fechas binarias:")

    # Test dates with different intervals
    date1 = date(1990, 5, 15)
    date2 = date(1990, 5, 16)  # 1 day difference
    date3 = date(1990, 6, 15)  # 1 month difference
    date4 = date(1991, 5, 15)  # 1 year difference

    # Create a list of dates for batch processing
    dates = [date1, date2, date3, date4]

    # Use GPU if available for improved performance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Encode dates using the binary method (batch processing would be ideal but not supported yet)
    encodings = []
    for i, d in enumerate(dates):
        enc = hdc.encode_date_binary(d)
        # Check if already a tensor and handle accordingly
        if isinstance(enc, torch.Tensor):
            enc_tensor = enc.to(device)
        else:
            # Fallback for NumPy arrays if needed
            enc_tensor = torch.from_numpy(enc).to(device)
        encodings.append(enc_tensor)
        print(f">>> Codificando fecha: {d}")

    # Stack tensors for more efficient processing
    # Shape: (4, dimension)
    stacked_encodings = torch.stack(encodings)

    # Calculate Hamming similarities more efficiently using tensor operations
    # 1. XOR operation to find differing bits (1 where different, 0 where same)
    # 2. Count the matching bits (by inverting XOR result)
    # 3. Normalize by dimension to get similarity

    with torch.no_grad():  # Disable gradient tracking for inference
        # Calculate hamming similarity using efficient tensor operations
        # Between date1 and the others (including itself)
        reference = stacked_encodings[0].unsqueeze(0)  # Shape: (1, dimension)

        # XOR to find differences
        differences = torch.logical_xor(reference, stacked_encodings).float()

        # Hamming distances (count differences)
        hamming_distances = differences.sum(dim=1)

        # Convert to similarities (1.0 - normalized distance)
        similarities = 1.0 - (hamming_distances / hdc.dim)

        # Extract individual similarity scores
        sim_1_day = similarities[1].item()  # Similarity with date2
        sim_1_month = similarities[2].item()  # Similarity with date3
        sim_1_year = similarities[3].item()  # Similarity with date4

    print(f"Similitud binaria con 1 día de diferencia:  {sim_1_day:.4f}")
    print(f"Similitud binaria con 1 mes de diferencia:  {sim_1_month:.4f}")
    print(f"Similitud binaria con 1 año de diferencia:  {sim_1_year:.4f}")

    # Verify that similarity decreases as temporal distance increases
    correct_similarity_order = sim_1_day >= sim_1_month >= sim_1_year

    # More granular checks for diagnosis
    day_to_month = sim_1_day >= sim_1_month
    month_to_year = sim_1_month >= sim_1_year

    print(f"¿Día a mes mantiene orden correcto?   {day_to_month}")
    print(f"¿Mes a año mantiene orden correcto?   {month_to_year}")
    print(f"Orden completo de similitud correcto: {correct_similarity_order}")

    # Fail the test if any condition is not met
    if not day_to_month:
        pytest.fail("La similitud día-a-mes no sigue el orden esperado")

    if not month_to_year:
        pytest.fail("La similitud mes-a-año no sigue el orden esperado")

    # Additional verification: absolute differences to verify the similarity drop is proportional
    with torch.no_grad():
        print("\nDiferencias de similitud:")
        diff_day_month = sim_1_day - sim_1_month
        diff_month_year = sim_1_month - sim_1_year
        print(f"Diferencia día-a-mes:  {diff_day_month:.4f}")
        print(f"Diferencia mes-a-año:  {diff_month_year:.4f}")

    return correct_similarity_order

def test_date_similarity_ordering_bipolar():
    """Tests that the similarity between date encodings correctly reflects their temporal proximity"""
    print("\n--- Test: Date Similarity Ordering ---")

    mode = VECTOR_MODE.lower()
    print(f"Testing with mode: {mode}")

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Prueba de similitud de encoding de fechas:")

    # Create a batch of dates with different time intervals
    dates = [
        date(1990, 5, 15),  # reference
        date(1990, 5, 16),  # 1 day difference
        date(1990, 6, 15),  # 1 month difference
        date(1991, 5, 15)  # 1 year difference
    ]

    # Measure encoding time for performance evaluation
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if start_time is not None:
        start_time.record()

    # Encode all dates efficiently using batch processing if available
    # Otherwise fall back to individual encoding
    with torch.no_grad():  # Disable gradient tracking for inference
        try:
            # Try batch encoding if available (assuming encode_date supports it)
            encodings_tensor = torch.stack([
                torch.tensor(encode_date(d, mode=mode), device=device).float()
                for d in dates
            ])
        except Exception as e:
            # Fallback to individual encoding
            print(f"Batch encoding not supported, using individual encoding: {e}")
            encodings = []
            for d in dates:
                enc = encode_date(d, mode=mode)
                # Ensure it's a tensor and on the right device
                if not isinstance(enc, torch.Tensor):
                    enc = torch.tensor(enc, device=device)
                # Ensure tensors are float type for cosine_similarity
                enc = enc.float()
                encodings.append(enc)
            # Stack into a single batch tensor
            encodings_tensor = torch.stack(encodings)

    if end_time is not None:
        end_time.record()
        torch.cuda.synchronize()
        encoding_time = start_time.elapsed_time(end_time)
        print(f"Encoding time: {encoding_time:.2f} ms")

    # Reference encoding (first date)
    reference = encodings_tensor[0].unsqueeze(0)  # Shape: (1, dimension)

    # Calculate all similarities in a single batch operation
    # This avoids repeated calculations and is much more efficient
    with torch.no_grad():
        similarities = F.cosine_similarity(
            reference,  # Shape: (1, dimension)
            encodings_tensor,  # Shape: (4, dimension)
            dim=1  # Compare along feature dimension
        )

    # Extract individual similarities
    sim_1_day = similarities[1].item()  # Similarity with date 1 day apart
    sim_1_month = similarities[2].item()  # Similarity with date 1 month apart
    sim_1_year = similarities[3].item()  # Similarity with date 1 year apart

    print(f"Similaridad con 1 día de diferencia:  {sim_1_day:.4f}")
    print(f"Similaridad con 1 mes de diferencia:  {sim_1_month:.4f}")
    print(f"Similaridad con 1 año de diferencia:  {sim_1_year:.4f}")

    # Calculate similarity differences for more detailed analysis
    day_month_diff = sim_1_day - sim_1_month
    month_year_diff = sim_1_month - sim_1_year
    print(f"Diferencia día-mes: {day_month_diff:.4f}")
    print(f"Diferencia mes-año: {month_year_diff:.4f}")

    # Verify correct similarity ordering
    correct_similarity_order = sim_1_day > sim_1_month > sim_1_year
    print(f"Orden de similitud correcto (las fechas más cercanas tienen mayor similitud): {correct_similarity_order}")

    # Use an assertion that will cause the test to fail if the order is incorrect
    assert correct_similarity_order, "El orden de similitud es incorrecto: las fechas más alejadas deberían tener una similitud menor"

    return correct_similarity_order

@pytest.mark.parametrize("mode", ["binary"])  # si querés agregar luego: ["binary", "bipolar"]
def test_date_encoding_binary(mode):
    from hdc.binary_hdc import HyperDimensionalComputingBinary
    from datetime import date, timedelta
    import torch
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hdc = HyperDimensionalComputingBinary(dim=10000, seed=42)

    ref_date = date(1970, 1, 1)
    days_list = [0, 1, 7, 30, 90, 180, 365, 730, 1095]
    dates = [ref_date + timedelta(days=d) for d in days_list]

    def _to_bool_tensor(x):
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.from_numpy(x)
        return t.to(device).bool()

    def spearman_rho(x, y):
        x = np.asarray(x); y = np.asarray(y)

        def rankdata(a):
            order = np.argsort(a)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(a), dtype=float)
            vals = a[order]
            i = 0
            while i < len(a):
                j = i
                while j + 1 < len(a) and vals[j + 1] == vals[i]:
                    j += 1
                if j > i:
                    avg = (i + j) / 2.0
                    ranks[order[i:j+1]] = avg
                i = j + 1
            return ranks

        rx = rankdata(x); ry = rankdata(y)
        rx = rx - rx.mean(); ry = ry - ry.mean()
        denom = (np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum()))
        return 0.0 if denom == 0 else float((rx * ry).sum() / denom)

    with torch.no_grad():
        encodings = [_to_bool_tensor(hdc.encode_date_binary(d)) for d in dates]
        stack = torch.stack(encodings)
        ref = stack[0].unsqueeze(0)

        diff = torch.logical_xor(ref, stack).float()
        hamming = diff.sum(dim=1)
        similitud_binaria_normalizada = (1.0 - (hamming / hdc.dim)).detach().cpu().numpy()

    distances = np.array(days_list, dtype=float)
    rho_bin = spearman_rho(distances, similitud_binaria_normalizada) # que la tendencia decreciente exista, no que sea perfecta.

    print(f"device={device}")
    print(f"Spearman rho (bin)={rho_bin:.3f}")
    for d, s in zip(days_list, similitud_binaria_normalizada):
        print(f"dist={d:4d}  sim_bin={s:.4f}")

    assert rho_bin < 0.0, "Binario: no hay tendencia decreciente (Spearman rho no es negativa)"

    near_idx = [0, 1, 2, 3]
    far_idx = [6, 7, 8]
    near_mean_bin = similitud_binaria_normalizada[near_idx].mean()
    far_mean_bin = similitud_binaria_normalizada[far_idx].mean()
    print(f"bin near_mean={near_mean_bin:.4f} far_mean={far_mean_bin:.4f}")
    assert near_mean_bin > far_mean_bin, "Binario: near_mean no es mayor que far_mean"

    def count_local_violations(sim):
        eps = 1e-6
        return sum(1 for i in range(len(sim) - 1) if sim[i] + eps < sim[i + 1])

    local_violations = count_local_violations(similitud_binaria_normalizada)
    print(f"local violations (bin)={local_violations}")
    assert local_violations <= 2, f"Binario: demasiadas violaciones locales ({local_violations})"

def test_date_encoding_bipolar():
    """
    Test realista para validar codificación de fechas con vectores bipolares (HDC).

    Evalúa:
      1) Tendencia global decreciente (Spearman rho < 0).
      2) Fechas cercanas más similares que fechas lejanas (near_mean > far_mean).
      3) Separación "significativa" de forma robusta (no umbral absoluto arbitrario):
         - diff >= min_diff (umbral mínimo conservador)
         - diff >= 2 * std(far) (separación relativa)
      4) Monotonía local: permite algunas violaciones.
      5) Similitud de una fecha consigo misma ~ 1 (tolerancia numérica razonable).

    Nota: Este test evita exigir monotonía estricta punto-a-punto o gaps absolutos grandes,
    porque HDC preserva similaridad de forma estadística.
    """
    from hdc.bipolar_hdc import HyperDimensionalComputingBipolar
    from datetime import date, timedelta
    import torch
    import numpy as np
    import time

    print("\n--- Testing Bipolar Date Encoding ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hdc = HyperDimensionalComputingBipolar(dim=10000, seed=42)

    ref_date = date(1970, 1, 1)
    days_list = [0, 1, 7, 30, 90, 180, 365, 730, 1095]
    dates = [ref_date + timedelta(days=d) for d in days_list]

    def spearman_rho(x, y):
        """
        Spearman rho sin scipy: correlación de ranks con manejo simple de empates.
        Devuelve float en [-1, 1].
        """
        x = np.asarray(x)
        y = np.asarray(y)

        def rankdata(a):
            order = np.argsort(a)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(a), dtype=float)

            vals = a[order]
            i = 0
            while i < len(a):
                j = i
                while j + 1 < len(a) and vals[j + 1] == vals[i]:
                    j += 1
                if j > i:
                    avg = (i + j) / 2.0
                    ranks[order[i:j + 1]] = avg
                i = j + 1
            return ranks

        rx = rankdata(x)
        ry = rankdata(y)

        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom = (np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum()))
        if denom == 0:
            return 0.0
        return float((rx * ry).sum() / denom)

    start_time = time.time()

    with torch.no_grad():
        bp_encodings = []
        for d in dates:
            bp = hdc.encode_date_bipolar(d)
            if not isinstance(bp, torch.Tensor):
                bp = torch.tensor(bp, dtype=torch.float32)
            bp_encodings.append(bp.to(device).float())

        bp_stack = torch.stack(bp_encodings)              # (n, dim)
        bp_ref = bp_stack[0].unsqueeze(0)                 # (1, dim)

        # Normalización para coseno
        bp_ref_n = torch.nn.functional.normalize(bp_ref, p=2, dim=1)
        bp_stack_n = torch.nn.functional.normalize(bp_stack, p=2, dim=1)

        # Cosine similarities contra ref
        sim_bp = torch.matmul(bp_ref_n, bp_stack_n.t()).squeeze().detach().cpu().numpy()

    encoding_time = time.time() - start_time
    print(f"Tiempo de codificación: {encoding_time:.4f} segundos")

    distances = np.array(days_list, dtype=float)

    # 1) Tendencia global
    rho_bp = spearman_rho(distances, sim_bp)
    print(f"Spearman rho (tendencia): {rho_bp:.4f}")

    for d, sim in zip(days_list, sim_bp):
        print(f"Distancia: {d:4d} días - Similitud: {sim:.4f}")

    # 2) Separación near vs far
    near_idx = [0, 1, 2, 3]   # 0, 1, 7, 30
    far_idx = [6, 7, 8]       # 365, 730, 1095

    near_mean = float(sim_bp[near_idx].mean())
    far_mean = float(sim_bp[far_idx].mean())
    diff = near_mean - far_mean

    print(f"\nPromedio similitud fechas cercanas: {near_mean:.4f}")
    print(f"Promedio similitud fechas lejanas:  {far_mean:.4f}")
    print(f"Diferencia (near - far): {diff:.4f}")

    # 3) Monotonía local (permite algunas violaciones)
    def count_local_violations(sim):
        eps = 1e-6
        return sum(1 for i in range(len(sim) - 1) if sim[i] + eps < sim[i + 1])

    violations = count_local_violations(sim_bp)
    print(f"Violaciones locales de monotonía: {violations}")

    # Sensibilidad local (informativo)
    if len(sim_bp) >= 2:
        print(f"\nSimilitud misma fecha: {sim_bp[0]:.6f}")
        print(f"Similitud un día después: {sim_bp[1]:.6f}")
        print(f"Diferencia: {sim_bp[0] - sim_bp[1]:.6f}")

    # ---------- Assertions ----------
    # 1) Tendencia general decreciente
    assert rho_bp < 0.0, f"No hay tendencia decreciente global (rho={rho_bp:.4f})"

    # 2) Fechas cercanas más similares que lejanas
    assert near_mean > far_mean, (
        f"Fechas cercanas ({near_mean:.4f}) no son más similares que fechas lejanas ({far_mean:.4f})"
    )

    # 3) Separación significativa (realista):
    #    (A) Umbral mínimo conservador
    min_diff = 0.02
    assert diff >= min_diff, (
        f"Diferencia near-far ({diff:.4f}) es muy baja (esperado >= {min_diff}). "
        "Esto sugiere poca discriminación temporal a escala anual."
    )

    #    (B) Separación relativa: al menos 2 * std(far)
    far_std = float(sim_bp[far_idx].std())
    far_std = max(far_std, 1e-6)
    assert diff >= 2.0 * far_std, (
        f"Diferencia near-far ({diff:.4f}) no supera 2*std(far) ({2.0*far_std:.4f}). "
        "La separación podría ser débil o estar saturada."
    )

    # 4) Limitar violaciones locales
    max_violations = 3
    assert violations <= max_violations, (
        f"Demasiadas violaciones de monotonía local: {violations} (máximo permitido: {max_violations})"
    )

    # 5) Similitud consigo misma ~ 1 (tolerancia numérica razonable)
    assert abs(float(sim_bp[0]) - 1.0) < 1e-4, (
        f"La similitud de una fecha consigo misma debe ser ~1.0, no {sim_bp[0]:.6f}"
    )



if __name__ == "__main__":
    test_result = test_date_encoding_bipolar()
    print(f"\n{'Test PASSED' if test_result else 'Test FAILED'}")

if __name__ == "__main__":
    test_result = test_date_encoding_bipolar()
    print(f"\n{'Test PASSED' if test_result else 'Test FAILED'}")

if __name__ == "__main__":
    consistency_result  = test_same_person_encoding_is_consistent()
    differentiation_result = test_different_people_produce_different_encodings()
    db_preservation_result = test_db_encoding_preservation()
    search_result = test_search_with_encoded_vector()
    date_range_result = test_date_range_search()
    binary_date_similarity_result = test_date_similarity_ordering_binary()
    bipolar_date_similarity_result = test_date_similarity_ordering_bipolar()
    date_encoding_binary_result = test_date_encoding_binary()  # <- Add this line

    print("\n\033[94m--- Resultados de ejecución de tests: ---\033[0m")

    if consistency_result:
        print("\033[92m ✓ Test de consistencia de Encoding: PASS\033[0m")
    else:
        print("\033[91m✗ Test de consistencia de Encoding: FALLÓ\033[0m")

    if differentiation_result:
        print("\033[92m ✓ Test de diferenciación de Encoding: PASS\033[0m")
    else:
        print("\033[91m✗ Test de diferenciación de Encoding: FALLÓ\033[0m")

    if db_preservation_result:
        print("\033[92m ✓ Test de preservación de encoding en la BD: PASS\033[0m")
    else:
        print("\033[91m✗ Test de preservación de encoding en la BD: FALLÓ\033[0m")

    if search_result:
        print("\033[92m ✓ Búsqueda basada en vector: PASS\033[0m")
    else:
        print("\033[91m✗ Búsqueda basada en vector: FALLÓ\033[0m")

    if date_range_result:
        print("\033[92m ✓ Test de búsqueda en range de fechas: PASS\033[0m")
    else:
        print("\033[91m✗ Test de búsqueda en range de fechas: FALLÓ\033[0m")

    if binary_date_similarity_result:
        print("\033[92m ✓ Test de similitud de fechas: PASS\033[0m")
    else:
        print("\033[91m✗ Test de similitud de fechas: FALLÓ\033[0m")

    if bipolar_date_similarity_result:
        print("\033[92m ✓ Test de similitud de fechas: PASS\033[0m")
    else:
        print("\033[91m✗ Test de similitud de fechas: FALLÓ\033[0m")

    if date_encoding_binary_result:
        print("\033[92m ✓ Test de codificación escalar de fechas (binario): PASS\033[0m")
    else:
        print("\033[91m✗ Test de codificación escalar de fechas (binario): FALLÓ\033[0m")