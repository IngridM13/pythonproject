# --- Milvus-backed storage/search (replaces psycopg2 bits) ---
import os
from datetime import datetime, date
import datetime
from dateutil.parser import parse as dateutil_parse
import numpy as np
import hashlib
import re
import ast
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime, date as date_cls, timedelta
from configs.settings import HDC_DIM as DIMENSION, HDC_DIM
from database_utils.milvus_db_connection import ensure_people_collection, VECTOR_MODE
# from encoding_methods.encoding_and_search_typed import normalize_person_data
from hdc.bipolar_hdc import HyperDimensionalComputingBipolar
from hdc.binary_hdc import HyperDimensionalComputingBinary
from database_utils.milvus_db_connection import get_vector_mode



# Global dictionary to cache hypervectors
hv_dict = {}


# Helpers to pack/unpack bipolar HVs for Milvus
def _bipolar_to_binary_bytes(hv: np.ndarray) -> bytes:
    # hv in {-1,+1} -> bits 0/1 -> bytes
    bits = (hv > 0).astype(np.uint8)
    return np.packbits(bits, bitorder="big").tobytes()

def _binary_bytes_to_bipolar(b: bytes, dim: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="big")
    bits = bits[:dim]
    return np.where(bits == 1, 1, -1).astype(np.int8)
'''
def _encode_for_milvus(hv: np.ndarray):
    if VECTOR_MODE == "binary":
        # Already binary, just pack bits into bytes
        return np.packbits(hv, bitorder="big").tobytes()
    else:
        # float mode: convert to float32
        return hv.astype(np.float32)
'''


def _encode_for_milvus(hv: np.ndarray) -> bytes | list[float]:
    """
    Prepares vector for Milvus based on VECTOR_MODE:
    - For binary mode: packs bits into bytes
    - For float mode: converts to a list of float values
    """
    from database_utils.milvus_db_connection import get_vector_mode

    vector_mode = get_vector_mode()

    if vector_mode == "binary":
        # Use your existing function for bipolar to binary conversion
        return _bipolar_to_binary_bytes(hv)
    else:  # vector_mode == "float"
        # For float vectors, convert to list of floats
        return hv.astype(float).tolist()


def _decode_from_milvus(stored) -> np.ndarray:
    if VECTOR_MODE == "binary":
        bits = np.unpackbits(np.frombuffer(stored, dtype=np.uint8), bitorder="big")
        return bits[:DIMENSION].astype(np.uint8)  # Return as uint8 with 0/1 values
    else:
        # list[float] from Milvus -> np.float32
        return np.asarray(stored, dtype=np.float32)

def _split_attrs(attrs: Dict[str, Any] | None):
    attrs = attrs or {}
    return attrs.get("address", []), attrs.get("akas", []), attrs.get("landlines", [])
'''
def _merge_attrs(person: Dict[str, Any]) -> Dict[str, Any]:
    # move list-like fields into one JSON bag for Milvus
    return {
        "address": person.get("address", []) or [],
        "akas": person.get("akas", []) or [],
        "landlines": person.get("landlines", []) or [],
    }'''
def _merge_attrs(person: Dict[str, Any]) -> Dict[str, Any]:
    """Move list-like fields into one JSON bag for Milvus"""

    result = {}

    # If we already have an attrs dictionary, use it directly
    if "attrs" in person and isinstance(person["attrs"], dict):
        # Just make a copy of the existing attrs dictionary
        print("[_merge_attrs] Using existing attrs dictionary")
        result = {
            "address": person["attrs"].get("address", []) or [],
            "akas": person["attrs"].get("akas", []) or [],
            "landlines": person["attrs"].get("landlines", []) or [],
        }
    else:
        # deberia pasar siempre bien los datos y esto no sería necesario. checkear.
        result = {
            "address": person.get("address", []) or [],
            "akas": person.get("akas", []) or [],
            "landlines": person.get("landlines", []) or [],
        }

    # Add more debug info
    print("[_merge_attrs] Result:")
    for key, value in result.items():
        print(f"  {key}: {repr(value)} (type={type(value).__name__})")

    return result

# def deterministic_hash(key: str) -> int:
#     import hashlib
#     key_bytes = str(key).encode("utf-8")
#     h = hashlib.md5(key_bytes).digest()
#     return int.from_bytes(h[:8], "little")
#
#
# def get_hv(key) -> np.ndarray:
#     """Get a reproducible binary hypervector for a key based on its hash"""
#     key_str = str(key)
#     if key_str not in hv_dict:
#         rng = np.random.RandomState(deterministic_hash(key_str) % (2**32))
#         # Generate binary vectors with 0/1
#         hv_dict[key_str] = rng.choice([0, 1], DIMENSION).astype(np.uint8)
#     return hv_dict[key_str]

def encode_date(date_obj, mode="binary"):
    """Factory function that delegates to the appropriate encoding function."""
    hdc_bipolar = HyperDimensionalComputingBipolar()
    hdc_binary = HyperDimensionalComputingBinary()
    if mode == "binary":
        print(">>> [encode_date] encoding binary")
        return hdc_binary.encode_date_binary(date_obj)
    else:
        print(">>> [encode_date] encoding bipolar")
        return hdc_bipolar.encode_date_bipolar(date_obj)

def encode_date_binary(date_obj):
    """Special encoding for date objects using binary HDC that preserves semantic meaning"""

    hdc_binary = HyperDimensionalComputingBinary()
    if not date_obj:
        return np.zeros(DIMENSION, dtype=np.uint8)

    # Get basic encoding for the full date
    base_encoding = hdc_binary.get_binary_hv(str(date_obj))

    # Also encode year and month separately
    year_encoding = hdc_binary.get_binary_hv(f"year_{date_obj.year}")
    month_encoding = hdc_binary.get_binary_hv(f"month_{date_obj.month}") #we used to have weights here?

    # Combine using binary operations (logical OR)
    result = np.zeros(DIMENSION, dtype=np.uint8)
    for enc in [base_encoding, year_encoding, month_encoding]:
        result = np.logical_or(result, enc).astype(np.uint8)

    return result


def encode_person(person, mode="binary"):
    """Factory function that delegates to the appropriate encoding function."""
    from database_utils.milvus_db_connection import get_vector_mode

    hdc_bipolar = HyperDimensionalComputingBipolar(dim=HDC_DIM)
    hdc_binary = HyperDimensionalComputingBinary(dim=HDC_DIM)

    vector_mode = get_vector_mode()
    print(f">>>>> {vector_mode}")

    if vector_mode == "binary":
        print(">>> [encode_person] encoding binary")
        return hdc_binary.encode_person_binary(person)
    elif vector_mode == "float":
        print(">>> [encode_person] encoding bipolar")
        return hdc_bipolar.encode_person_bipolar(person)
    else:
        raise ValueError(f"Invalid vector mode: {vector_mode}")


'''
def encode_person_binary(person):
    """Encode a person's data into a binary hypervector (0/1 representation)."""
    components = []
    hdc_binary = HyperDimensionalComputingBinary()

    # Sort keys to ensure consistent order
    for key in sorted(person.keys()):
        value = person[key]

        # Handle different data types
        if isinstance(value, list):
            if not value:
                encoded_value = np.zeros(DIMENSION, dtype=np.uint8)
            else:
                # Binary bundling with logical OR
                encoded_value = np.zeros(DIMENSION, dtype=np.uint8)
                for v in value:
                    encoded_value = np.logical_or(encoded_value, hdc_binary.get_binary_hv(str(v))).astype(np.uint8)
        elif isinstance(value, (date, datetime)):
            encoded_value = encode_date_binary(value)
        elif value is None:
            encoded_value = np.zeros(DIMENSION, dtype=np.uint8)
        else:
            encoded_value = hdc_binary.get_binary_hv(str(value))

        # Binding with XOR (appropriate for binary vectors)
        bound = np.logical_xor(hdc_binary.get_binary_hv(key), encoded_value).astype(np.uint8)
        components.append(bound)

    if not components:
        return np.zeros(DIMENSION, dtype=np.uint8)

    # Bundle components with binary majority voting
    result = np.zeros((DIMENSION,), dtype=np.int32)  # Use int32 for accumulation
    for comp in components:
        result += comp

    # Threshold at half the number of components
    threshold = len(components) / 2
    return (result > threshold).astype(np.uint8)


def encode_person_bipolar(person):
    """Encode a person's data into a bipolar hypervector (-1/+1 representation)."""
    components = []
    hdc_bipolar = HyperDimensionalComputingBipolar()


    # Sort keys to ensure consistent order
    for key in sorted(person.keys()):
        value = person[key]

        # Handle different data types
        if isinstance(value, list):
            if not value:
                encoded_value = np.ones(DIMENSION, dtype=np.int8)  # Neutral for bipolar is +1
            else:
                # For bipolar, we add and then threshold (sign)
                encoded_values = [hdc_bipolar.get_bipolar_hv(str(v)) for v in value]
                encoded_value = np.sum(encoded_values, axis=0)
                encoded_value = np.sign(encoded_value)
                encoded_value[encoded_value == 0] = 1  # Replace zeros with ones
                encoded_value = encoded_value.astype(np.int8)
        elif isinstance(value, (date, datetime)):
            encoded_value = encode_date_bipolar(value)
        elif value is None:
            encoded_value = np.ones(DIMENSION, dtype=np.int8)  # Neutral for bipolar is +1
        else:
            encoded_value = hdc_bipolar.get_bipolar_hv(str(value))

        # Binding with element-wise multiplication (appropriate for bipolar)
        bound = hdc_bipolar.get_bipolar_hv(key) * encoded_value
        components.append(bound)

    if not components:
        return np.ones(DIMENSION, dtype=np.int8)

    # Bundle components with bipolar majority voting (sign of sum)
    result = np.sum(components, axis=0)
    result = np.sign(result)
    result[result == 0] = 1  # Handle zeros

    return result.astype(np.int8)
'''

def store_person(person, collection_name: str = "people") -> int:
    """Almacena una persona en Milvus con su HV; devuelve el ID generado automáticamente.

    Args:
        person: Datos DE LA PERSONA YA NORMALIZADOS a almacenar
        collection_name: Nombre opcional de la colección (si no se proporciona,
                         se usa la colección predeterminada)

    Returns:
        int: ID generado automáticamente de la persona almacenada
    """
    # Use the provided collection name or default collection
    col = ensure_people_collection(collection_name)

    # normalized = normalize_person_data(person)  <-- 1. REMOVE THIS LINE

    hv = encode_person(person)  # <-- 2. Use 'person' instead of 'normalized'

    attrs = _merge_attrs(person)  # <-- 3. Use 'person'
    dob_val = person.get("dob")  # <-- 4. Use 'person'

    # Store dob as ISO string for easy range filtering
    if isinstance(dob_val, (date, datetime)):
        dob_str = dob_val.strftime("%Y-%m-%d")
    else:
        # This branch will now likely handle None or ""
        dob_str = dob_val if dob_val else ""

    entity = {
        "name": person.get("name", ""),  # <-- 5. Use 'person'
        "lastname": person.get("lastname", ""),  # <-- 6. Use 'person'
        "dob": dob_str,
        "marital_status": person.get("marital_status", ""),  # <-- 7. Use 'person'
        "mobile_number": person.get("mobile_number", ""),  # <-- 8. Use 'person'
        "gender": person.get("gender", ""),  # <-- 9. Use 'person'
        "race": person.get("race", ""),  # <-- 10. Use 'person'
        "attrs": attrs,
        "hv": _encode_for_milvus(hv),
    }
    mr = col.insert(entity)
    col.flush()
    return int(mr.primary_keys[0])

def get_person_details(person_id: int, collection_name: str = "people"):
    """Get complete details for a person by ID (Milvus)."""
    col = ensure_people_collection(collection_name)
    res = col.query(
        expr=f"id == {person_id}",
        output_fields=["name","lastname","dob","marital_status","mobile_number","gender","race","attrs"],
        consistency_level="Strong",
    )
    if not res:
        return None
    r = res[0]
    address, akas, landlines = _split_attrs(r.get("attrs"))
    return {
        "id": person_id,
        "name": r.get("name",""),
        "lastname": r.get("lastname",""),
        "dob": r.get("dob",""),
        "address": address,
        "marital_status": r.get("marital_status",""),
        "akas": akas,
        "landlines": landlines,
        "mobile_number": r.get("mobile_number",""),
        "gender": r.get("gender",""),
        "race": r.get("race",""),
    }

def parse_list_string(list_str):
    """Parse list strings from CSV or text field, with debug prints."""

    # Case 1: empty or '[]'
    if not list_str or list_str == '[]':
        return []

    try:
        parsed = ast.literal_eval(list_str)
        return parsed
    except Exception as e:
        print(f"[parse_list_string] WARNING: literal_eval failed: {e!r}")
        print(f"[parse_list_string] Fallback → wrapping value in list: [{list_str!r}]")
        return [list_str]

def _is_iso_date(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s or ""))

def normalize_person_data_db(person: dict) -> dict:
    """
    Normalize to strict DB-style. Raises ValueError if shape is not DB-style.
    Required:
      - top-level scalars: name, lastname, dob (ISO or ""), marital_status, mobile_number, gender, race
      - 'attrs' dict containing arrays: address, akas, landlines (can be empty lists)
    """
    if not isinstance(person, dict):
        raise ValueError("person must be a dict")

    # Hard errors if legacy app-style fields are present
    for k in ("address", "akas", "landlines"):
        if k in person:
            raise ValueError(f"DB-style only: '{k}' must be inside 'attrs'")

    if "attrs" not in person or not isinstance(person["attrs"], dict):
        raise ValueError("DB-style only: missing 'attrs' dict")

    attrs = person["attrs"]
    for k in ("address", "akas", "landlines"):
        if k not in attrs:
            attrs[k] = []
        if not isinstance(attrs[k], list):
            raise ValueError(f"'attrs.{k}' must be a list")

    # dob must be ISO string or empty
    dob = person.get("dob", "")
    if dob and not (isinstance(dob, str) and _is_iso_date(dob)):
        raise ValueError("DB-style only: 'dob' must be ISO 'YYYY-MM-DD' or ''")

    # Copy only allowed fields to canonical dict
    allowed = {
        "name": str(person.get("name", "")),
        "lastname": str(person.get("lastname", "")),
        "dob": dob or "",
        "marital_status": str(person.get("marital_status", "")),
        "mobile_number": str(person.get("mobile_number", "")),
        "gender": str(person.get("gender", "")),
        "race": str(person.get("race", "")),
        "attrs": {
            "address": list(attrs["address"]),
            "akas": list(attrs["akas"]),
            "landlines": list(attrs["landlines"]),
        },
    }
    return allowed

'''
def parse_date(value: Any) -> Optional[date]:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)  # estrictamente 'YYYY-MM-DD'
        except ValueError:
            return None
    return None
'''
from typing import Any, List


def _as_list_str(x: Any) -> List[str]:
    """
    Converts input into a list of strings.
    - Handles None or "" -> []
    - Handles a list -> [str(v) for v in x]
    - Handles a single item -> [str(x)]
    """
    # 1. Handle empty cases first
    if x in (None, ""):
        return []

    # 2. Handle the list case
    if isinstance(x, list):
        # Filter out any potential Nones inside the list
        return [str(v) for v in x if v not in (None, "")]

    # 3. Handle the single item case (this is the fix)
    # If it's not a list and not empty, wrap it
    return [str(x)]

DEFAULT_SCALARS: Dict[str, Any] = {
    "name": "",
    "lastname": "",
    "mobile_number": "",
    "race": "",
    "marital_status": "",
    "gender": "",
    "dob": None,  # se setea a date si se puede parsear
}

DEFAULT_ATTRS: Dict[str, List[str]] = {
    "address": [],
    "akas": [],
    "landlines": [],
}

def normalize_person_data(person: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(person, dict):
        raise ValueError("person must be a dict")

    out: Dict[str, Any] = {}

    # Inicializa con defaults de escalares
    out.update(DEFAULT_SCALARS)

    # Procesa claves de entrada
    for k, v in person.items():
        lk = k.lower()
        if lk == "attrs":
            continue  # se maneja luego
        if lk == "dob":
            out["dob"] = parse_date(v)
        elif lk in ("name", "lastname", "mobile_number", "race"):
            out[lk] = "" if v in (None, "") else str(v).strip()
        elif lk in ("marital_status", "gender"):
            out[lk] = "" if v in (None, "") else str(v).strip().capitalize()
        else:
            # si aparece otro escalar no-lista, lo guardamos como string
            if not isinstance(v, list):
                out[lk] = "" if v in (None, "") else str(v)

    # Asegura attrs y sus listas conocidas
    attrs_in = person.get("attrs")
    attrs: Dict[str, Any] = dict(attrs_in) if isinstance(attrs_in, dict) else {}

    # Si address vino al top-level, muévelo a attrs (tu test lo manda así)
    if "address" in person and "address" not in attrs:
        attrs["address"] = person.get("address")
    print(f"[normalize_person_data] attrs: {attrs}")
    # Normaliza listas conocidas
    for key, default_list in DEFAULT_ATTRS.items():
        attrs[key] = _as_list_str(attrs.get(key, default_list))

    out["attrs"] = attrs

    return out

def find_closest_match_db(query_person, threshold=0.7, limit=5, collection_name: str = "people"):
    """Vector search in Milvus; returns top matches with a 'similarity' field."""
    col = ensure_people_collection(collection_name)
    normalized_query = normalize_person_data(query_person)
    qhv = encode_person(normalized_query)
    qpayload = _encode_for_milvus(qhv)

    if VECTOR_MODE == "binary":
        print(">>> vector mode is BINARY")
        # HAMMING distance: smaller is better; convert to similarity in [0,1]
        search_params = {"metric_type": "HAMMING", "params": {}}
        metric = "HAMMING"
    else:
        print(">>> vector mode is BIPOLAR")
        # Usa IP (Inner Product). Mayor es mejor.
        # Milvus devuelve la 'distancia' como el producto interno real.
        search_params = {"metric_type": "IP", "params": {"ef": 128}}
        metric = "IP"

    results = col.search(
        data=[qpayload],
        anns_field="hv",
        param=search_params,
        limit=limit*3,            # fetch more, filter with threshold below
        output_fields=["name","lastname","dob","gender"],
    )
    hits = results[0] if results else []

    out = []
    for h in hits:
        # distance -> similarity
        if metric == "HAMMING":
            sim = 1.0 - (h.distance / float(DIMENSION))
        else:  # producto interno
            # h.distance ES la similitud (el producto interno)
            # Un valor más alto es mejor.
            # Lo normalizamos a [0, 1] para que sea consistente
            # (asumiendo que la dimensión es el score máximo posible)
            sim = float(h.distance) / float(DIMENSION)
        if sim >= threshold:
            out.append({
                "id": int(h.id),
                "name": h.entity.get("name"),
                "lastname": h.entity.get("lastname"),
                "dob": h.entity.get("dob"),
                "gender": h.entity.get("gender"),
                "similarity": sim,
            })
    # Sort by similarity desc and cut to limit
    out.sort(key=lambda x: x["similarity"], reverse=True)
    return out[:limit]


def find_similar_by_date(target_date, range_days=30, limit=5, collection_name: str = "people"):
    """Scalar filter by date range using dob stored as ISO 'YYYY-MM-DD' strings."""
    col = ensure_people_collection(collection_name)

    # Normalize target_date to a datetime.date
    if not isinstance(target_date, (datetime, date_cls)):
        target_date = parse_date(target_date)
    if isinstance(target_date, datetime):
        target_date = target_date.date()
    if target_date is None:
        return []

    # Build ISO strings (lexicographic ordering == chronological in YYYY-MM-DD)
    lo = (target_date - timedelta(days=range_days)).strftime("%Y-%m-%d")
    hi = (target_date + timedelta(days=range_days)).strftime("%Y-%m-%d")

    expr = f'dob >= "{lo}" && dob <= "{hi}"'
    res = col.query(
        expr=expr,
        output_fields=["id", "name", "lastname", "dob"],
        consistency_level="Strong",
        limit=limit
    )

    # Sort by dob to mimic SQL ORDER BY (string sort works with ISO format)
    res.sort(key=lambda r: r.get("dob", ""))

    return [{"id": int(r["id"]), "name": r["name"], "lastname": r["lastname"], "dob": r["dob"]}
            for r in res][:limit]

'''
def parse_date(s: str | None) -> date_cls | None:
    """
    Convert ISO 'YYYY-MM-DD' to datetime.date.
    Accepts '' or None -> None.
    Raises ValueError for any non-ISO input.
    """
    if s in (None, ""):
        return None
    if not isinstance(s, str):
        raise TypeError("dob must be an ISO string 'YYYY-MM-DD' or ''")
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError("Expected ISO 'YYYY-MM-DD' in 'dob'") from e
    '''


def parse_date(s: str | None) -> date_cls | None:
    """
    Convert a date string to datetime.date.
    Accepts '' or None -> None.
    Handles various common date formats (ISO, slashes, etc.).
    Raises ValueError for unparseable input.
    """
    if s in (None, ""):
        return None

    # This was the fix for your previous problem (keep it)
    if isinstance(s, date_cls):
        return s

    if not isinstance(s, str):
        raise TypeError("dob must be a string, '', or None")

    try:
        # Use dateutil.parse to flexibly handle formats
        return dateutil_parse(s).date()
    except (ValueError, OverflowError) as e:
        # Re-raise with a clear message
        raise ValueError(f"Could not parse '{s}' as a valid date") from e