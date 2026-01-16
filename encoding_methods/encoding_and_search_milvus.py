# --- Milvus-backed storage/search (replaces psycopg2 bits) ---
from datetime import date
import datetime
import numpy as np
import torch
import re
import ast
from typing import Any, Dict
from datetime import datetime, date as date_cls, timedelta
from configs.settings import HDC_DIM as DIMENSION, HDC_DIM
from database_utils.milvus_db_connection import ensure_people_collection, VECTOR_MODE
from hdc.binary_hdc import HyperDimensionalComputingBinary
from utils.person_data_normalization import parse_date, normalize_person_data
from hdc.bipolar_hdc import HyperDimensionalComputingBipolar




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

def _encode_for_milvus(hv) -> bytes | list[float]:
    """
    Prepares vector for Milvus based on VECTOR_MODE:
    - For binary mode: packs bits into bytes
    - For float mode: converts to a list of float values
    """
    from database_utils.milvus_db_connection import get_vector_mode
    import torch
    import numpy as np

    vector_mode = get_vector_mode()

    # Handle PyTorch Tensors
    if isinstance(hv, torch.Tensor):
        hv = hv.detach().cpu()
        if vector_mode == "float":
            # Direct conversion for float mode
            return hv.float().tolist()
        else:
            # For binary mode, convert to numpy in case _bipolar_to_binary_bytes expects it
            hv = hv.numpy()

    if vector_mode == "binary":
        # Use your existing function for bipolar to binary conversion
        return _bipolar_to_binary_bytes(hv)
    else:  # vector_mode == "float"
        # For float vectors, convert to list of floats
        return hv.astype(float).tolist()


def _split_attrs(attrs: Dict[str, Any] | None):
    attrs = attrs or {}
    return attrs.get("address", []), attrs.get("akas", []), attrs.get("landlines", [])

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
        return hdc_bipolar.encode_person_generalized(person)
    else:
        raise ValueError(f"Invalid vector mode: {vector_mode}")


def store_person(person: Dict[str, Any], collection_name: str = "people") -> int:
    """
    Store a person's details in Milvus, handling various input scenarios.

    Args:
        person: Dictionary containing person details
        collection_name: Name of the Milvus collection

    Returns:
        The ID of the stored person
    """
    import datetime  # Import explicitly to be sure

    # Create a copy to avoid modifying the original input
    person_data = person.copy()

    # Extract embedding if present
    embedding = person_data.pop('embedding', None)

    # Prepare attributes: ensure 'attrs' exists and move top-level lists into it
    # This ensures normalize_person_data includes them in the encoding
    if 'attrs' not in person_data or not isinstance(person_data['attrs'], dict):
        person_data['attrs'] = {}

    for attr_key in ['address', 'akas', 'landlines']:
        if attr_key in person_data:
            person_data['attrs'][attr_key] = person_data.pop(attr_key)

    # Normalize the person data
    normalized_person = normalize_person_data(person_data)

    # Encode the person
    hv = encode_person(normalized_person)

    # Prepare the document to insert
    doc_to_insert = {
        **normalized_person,
        "hv": hv.tolist(),  # Convert numpy array to list for Milvus
    }

    # Ensure dob is a string (YYYY-MM-DD) for Milvus insertion
    if "dob" in doc_to_insert:
        d = doc_to_insert["dob"]
        # Check against standard datetime types
        if isinstance(d, (datetime.date, datetime.datetime)):
            doc_to_insert["dob"] = d.strftime("%Y-%m-%d")
        elif d is None:
            # Explicitly handle None -> empty string for VARCHAR field
            doc_to_insert["dob"] = ""
        elif not isinstance(d, str):
            # Fallback: force string conversion if it's not None and not a string
            doc_to_insert["dob"] = str(d)

    # Handle embedding
    if embedding is not None:
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        elif isinstance(embedding, torch.Tensor):
            embedding = embedding.numpy().tolist()

        if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
            # Ensure embedding has correct dimension
            doc_to_insert["embedding"] = embedding[:128]  # Trim to max 128 dimensions

    # Insert into Milvus
    col = ensure_people_collection(collection_name)

    # Check for missing vector fields that are required by schema
    # Milvus requires vector fields to be populated
    schema_fields = {f.name: f for f in col.schema.fields}
    
    if "embedding" in schema_fields and "embedding" not in doc_to_insert:
        # We need to fill it with a placeholder (e.g., zero vector)
        # Try to determine dimension from schema
        emb_field = schema_fields["embedding"]
        dim = 128  # Default from our codebase
        
        # specific to pymilvus versions, dim might be in params or properties
        if hasattr(emb_field, 'params') and 'dim' in emb_field.params:
            dim = int(emb_field.params['dim'])
            
        doc_to_insert["embedding"] = [0.0] * dim

    res = col.insert(doc_to_insert)

    # Return the inserted ID (accessing primary_keys if inserted_ids is missing/warned)
    if hasattr(res, 'primary_keys'):
        return res.primary_keys[0]
    return res.inserted_ids[0]


def get_person_details(person_id: int, collection_name: str = "people") -> Dict[str, Any]:
    """Get complete details for a person by ID (Milvus)."""
    col = ensure_people_collection(collection_name)

    # Base output fields that we know exist
    output_fields = [
        "name", "lastname", "dob", "marital_status", "mobile_number",
        "gender", "race", "attrs", "id"
    ]

    # Check if 'embedding' exists in the collection schema before asking for it
    schema_fields = {f.name for f in col.schema.fields}
    if "embedding" in schema_fields:
        output_fields.append("embedding")

    res = col.query(
        expr=f"id == {person_id}",
        output_fields=output_fields,
        consistency_level="Strong",
    )

    if not res:
        return None

    r = res[0]
    address, akas, landlines = _split_attrs(r.get("attrs", {}))

    # Construct the return dictionary, including embedding if present
    return_dict = {
        "id": r.get("id", person_id),
        "name": r.get("name", ""),
        "lastname": r.get("lastname", ""),
        "dob": r.get("dob", ""),
        "address": address,
        "marital_status": r.get("marital_status", ""),
        "akas": akas,
        "landlines": landlines,
        "mobile_number": r.get("mobile_number", ""),
        "gender": r.get("gender", ""),
        "race": r.get("race", ""),
    }

    # Add embedding to the return dictionary if it exists in the result
    if "embedding" in r and r.get("embedding") is not None:
        return_dict["embedding"] = r["embedding"]

    return return_dict


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

'''