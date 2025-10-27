# --- Milvus-backed storage/search (replaces psycopg2 bits) ---
import os
from datetime import datetime, date
from typing import Dict, Any, List
import numpy as np
import hashlib
import re
import ast
import pandas as pd
from datetime import datetime, date as date_cls, timedelta
from configs.settings import HDC_DIM as DIMENSION
from database_utils.milvus_db_connection import ensure_people_collection, VECTOR_MODE
# from encoding_methods.encoding_and_search_typed import normalize_person_data

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

def _encode_for_milvus(hv: np.ndarray):
    if VECTOR_MODE == "binary":
        # Already binary, just pack bits into bytes
        return np.packbits(hv, bitorder="big").tobytes()
    else:
        # float mode: convert to float32
        return hv.astype(np.float32)

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

def deterministic_hash(key: str) -> int:
    import hashlib
    key_bytes = str(key).encode("utf-8")
    h = hashlib.md5(key_bytes).digest()
    return int.from_bytes(h[:8], "little")


def get_hv(key) -> np.ndarray:
    """Get a reproducible binary hypervector for a key based on its hash"""
    key_str = str(key)
    if key_str not in hv_dict:
        rng = np.random.RandomState(deterministic_hash(key_str) % (2**32))
        # Generate binary vectors with 0/1
        hv_dict[key_str] = rng.choice([0, 1], DIMENSION).astype(np.uint8)
    return hv_dict[key_str]

def encode_date(date_obj):
    """Special encoding for date objects using binary HDC that preserves semantic meaning"""
    if not date_obj:
        return np.zeros(DIMENSION, dtype=np.uint8)

    # Get basic encoding for the full date
    base_encoding = get_hv(str(date_obj))

    # Also encode year and month separately
    year_encoding = get_hv(f"year_{date_obj.year}")
    month_encoding = get_hv(f"month_{date_obj.month}") #we used to have weights here?
    
    # Combine using binary operations (logical OR)
    result = np.zeros(DIMENSION, dtype=np.uint8)
    for enc in [base_encoding, year_encoding, month_encoding]:
        result = np.logical_or(result, enc).astype(np.uint8)
    
    return result

def encode_person(person):
    """Encode a person's data into a binary hypervector."""


    components = []

    # Sort keys to ensure consistent order
    for key in sorted(person.keys()):
        value = person[key]

        # Handle different data types
        if isinstance(value, list):
            if not value:
                encoded_value = np.zeros(DIMENSION, dtype=np.uint8)
            else:
                for i, v in enumerate(value):
                    print(f"     item[{i}] = {repr(v)}")
                # For binary, we use logical OR to combine multiple items
                encoded_value = np.zeros(DIMENSION, dtype=np.uint8)
                for v in value:
                    encoded_value = np.logical_or(encoded_value, get_hv(str(v))).astype(np.uint8)
        elif isinstance(value, (date, datetime)):
            encoded_value = encode_date(value)
        elif value is None:
            encoded_value = np.zeros(DIMENSION, dtype=np.uint8)
        else:
            encoded_value = get_hv(str(value))

        # Bind key and value using XOR (equivalent of multiplication in bipolar)
        bound = np.logical_xor(get_hv(key), encoded_value).astype(np.uint8)
        components.append(bound)

    if not components:
        return np.zeros(DIMENSION, dtype=np.uint8)

    # Combine components using majority voting
    result = np.zeros(DIMENSION, dtype=np.uint8)
    for comp in components:
        result = np.logical_or(result, comp).astype(np.uint8)
    
    return result


def store_person(person) -> int:
    """Almacena una persona en Milvus con su HV; devuelve el ID generado automáticamente."""
    col = ensure_people_collection()
    normalized = normalize_person_data(person)
    hv = encode_person(normalized)

    attrs = _merge_attrs(normalized)
    dob_val = normalized.get("dob")
    # Store dob as ISO string for easy range filtering
    if isinstance(dob_val, (date, datetime)):
        dob_str = dob_val.strftime("%Y-%m-%d")
    else:
        dob_str = dob_val if dob_val else ""

    entity = {
        "name": normalized.get("name", ""),
        "lastname": normalized.get("lastname", ""),
        "dob": dob_str,
        "marital_status": normalized.get("marital_status", ""),
        "mobile_number": normalized.get("mobile_number", ""),
        "gender": normalized.get("gender", ""),
        "race": normalized.get("race", ""),
        "attrs": attrs,
        "hv": _encode_for_milvus(hv),
    }
    mr = col.insert(entity)
    col.flush()
    return int(mr.primary_keys[0])

def get_person_details(person_id: int):
    """Get complete details for a person by ID (Milvus)."""
    col = ensure_people_collection()
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

def normalize_person_data(person):
    """Normalize person data to handle format inconsistencies, convert types, and standardize key case"""
    normalized = {}
    for key, value in person.items():
        # Standardize key to lowercase
        normalized_key = key.lower()

        # Handle list strings
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            normalized[normalized_key] = parse_list_string(value)
        # Convert date strings to date objects
        elif normalized_key == 'dob' and value and not isinstance(value, (date, datetime)):
            normalized[normalized_key] = parse_date(value)
        else:
            normalized[normalized_key] = value

    return normalized

def find_closest_match_db(query_person, threshold=0.7, limit=5):
    """Vector search in Milvus; returns top matches with a 'similarity' field."""
    col = ensure_people_collection()
    normalized_query = normalize_person_data(query_person)
    qhv = encode_person(normalized_query)
    qpayload = _encode_for_milvus(qhv)

    if VECTOR_MODE == "binary":
        # HAMMING distance: smaller is better; convert to similarity in [0,1]
        search_params = {"metric_type": "HAMMING", "params": {}}
        metric = "HAMMING"
    else:
        # COSINE distance in Milvus is (1 - cosine_similarity). Convert back to similarity.
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
        metric = "COSINE"

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
        else:  # COSINE
            sim = 1.0 - float(h.distance)
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


def find_similar_by_date(target_date, range_days=30, limit=5):
    """Scalar filter by date range using dob stored as ISO 'YYYY-MM-DD' strings."""
    col = ensure_people_collection()

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