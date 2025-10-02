from typing import Dict, List, Optional, Union
import numpy as np

from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection,
    utility, MilvusException
)

# ---------- CONFIG ----------
MILVUS_ALIAS = "default"
MILVUS_HOST  = "localhost"     # mapped in your docker-compose (standalone -> 19530)
MILVUS_PORT  = "19530"
COLLECTION   = "people"

# Choose ONE representation for your HD vectors:
# A) Binary vectors (compact, use HAMMING/JACCARD/TANIMOTO metrics)
USE_BINARY_VECTORS = True
HYPERVEC_DIM       = 10_000     # must be multiple of 8 for binary vectors

# B) If you prefer cosine similarity exactly as before, set USE_BINARY_VECTORS=False
#    (stores float vectors; larger on disk/RAM)

# ---------- VECTOR CONVERSION ----------
def bipolar_to_binary_bytes(hv: np.ndarray) -> bytes:
    """
    Convert a bipolar vector of shape (D,) with values in {-1, +1}
    to a packed bitstring (0 for -1, 1 for +1) for Milvus BinaryVector.
    """
    assert hv.ndim == 1 and hv.shape[0] == HYPERVEC_DIM, "Unexpected HV shape"
    # Map -1 -> 0, +1 -> 1
    bits = (hv > 0).astype(np.uint8)
    # Pack bits into bytes (little-endian within each byte as per numpy)
    return np.packbits(bits, bitorder="big").tobytes()

def bipolar_to_float32(hv: np.ndarray) -> np.ndarray:
    """
    Convert bipolar {-1, +1} to float32; preserves cosine similarity.
    """
    assert hv.ndim == 1 and hv.shape[0] == HYPERVEC_DIM, "Unexpected HV shape"
    return hv.astype(np.float32)

# ---------- CONNECT ----------
def connect():
    # idempotent: safe to call many times
    if not connections.has_connection(MILVUS_ALIAS):
        connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)

# ---------- SCHEMA / COLLECTION ----------
def get_or_create_collection() -> Collection:
    """
    people schema:
      - id:      Int64 (auto_id primary key)
      - name:    VarChar
      - lastname:VarChar
      - dob:     VarChar  (keep as text for now)
      - marital_status, gender, race: VarChar
      - mobile_number:   VarChar
      - attrs:   JSON  (hold arrays like address, akas, landlines as a single JSON bag)
      - hv:      BinaryVector(dim) OR FloatVector(dim)
    """
    connect()

    if utility.has_collection(COLLECTION):
        col = Collection(COLLECTION)
        return col

    fields: List[FieldSchema] = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="lastname", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="dob", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="marital_status", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="mobile_number", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="gender", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="race", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="attrs", dtype=DataType.JSON),
    ]

    if USE_BINARY_VECTORS:
        fields.append(FieldSchema(name="hv", dtype=DataType.BINARY_VECTOR, dim=HYPERVEC_DIM))
    else:
        fields.append(FieldSchema(name="hv", dtype=DataType.FLOAT_VECTOR, dim=HYPERVEC_DIM))

    schema = CollectionSchema(fields=fields, description="People with hypervectors")

    col = Collection(name=COLLECTION, schema=schema, using=MILVUS_ALIAS)

    # Create indexes
    if USE_BINARY_VECTORS:
        # Typical choice for packed bit HVs is HAMMING or JACCARD/TANIMOTO
        col.create_index(
            field_name="hv",
            index_params={
                "index_type": "BIN_FLAT",        # or BIN_IVF_FLAT for larger collections
                "metric_type": "HAMMING",        # Hamming correlates with your ±1 overlap
                "params": {}
            }
        )
    else:
        # IVF_FLAT/IVF_SQ8/IVF_PQ/HNSW are common; COSINE mirrors your old cosine sim
        col.create_index(
            field_name="hv",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200}
            }
        )

    # Optional inverted index for fast filtering on text (lastname)
    try:
        col.create_index(
            field_name="lastname",
            index_params={
                "index_type": "INVERTED",
                "metric_type": "L2",   # ignored for scalar index; required by API in some versions
                "params": {}
            }
        )
    except MilvusException:
        pass  # scalar index availability varies by version; safe to ignore

    col.load()  # ready for search
    return col

# ---------- INSERT ----------
def insert_person(
    hv: np.ndarray,
    name: str,
    lastname: str,
    dob: str = "",
    marital_status: str = "",
    mobile_number: str = "",
    gender: str = "",
    race: str = "",
    attrs: Optional[Dict] = None,
) -> int:
    """
    Insert one person with their HV and metadata.
    attrs can hold arrays: {"address": [...], "akas": [...], "landlines": [...]}
    Returns the auto-generated id.
    """
    col = get_or_create_collection()

    if USE_BINARY_VECTORS:
        hv_payload = bipolar_to_binary_bytes(hv)
    else:
        hv_payload = bipolar_to_float32(hv)

    data = [
        # id omitted -> auto_id
        [name],
        [lastname],
        [dob],
        [marital_status],
        [mobile_number],
        [gender],
        [race],
        [attrs or {}],
        [hv_payload],
    ]

    # The order in 'data' must match schema field order excluding auto_id
    # Rebuild to ensure correct order:
    ordered = {
        "name": data[0], "lastname": data[1], "dob": data[2],
        "marital_status": data[3], "mobile_number": data[4],
        "gender": data[5], "race": data[6], "attrs": data[7], "hv": data[8]
    }
    mr = col.insert(ordered)
    col.flush()
    return int(mr.primary_keys[0])

# ---------- SEARCH ----------
def search_similar(
    query_hv: np.ndarray,
    top_k: int = 5,
    expr: Optional[str] = None,      # e.g., 'lastname == "machado"' or 'gender == "female"'
    output_fields: Optional[List[str]] = None
):
    """
    kNN search by vector similarity with optional scalar filtering (expr).
    """
    col = get_or_create_collection()

    if USE_BINARY_VECTORS:
        query_vec = [bipolar_to_binary_bytes(query_hv)]
        search_params = {"metric_type": "HAMMING", "params": {"radius": -1, "range_filter": -1}}
    else:
        query_vec = [bipolar_to_float32(query_hv)]
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    results = col.search(
        data=query_vec,
        anns_field="hv",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=output_fields or ["name", "lastname", "dob", "mobile_number", "attrs"]
    )
    return results[0]  # first (and only) query
