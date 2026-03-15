import datetime
import torch
import re
import ast
import hashlib
from typing import Any, Dict, List, Union
from datetime import datetime, date as date_cls, timedelta

from configs.settings import HDC_DIM as DIMENSION, HDC_DIM
from database_utils.milvus_db_connection import ensure_people_collection, VECTOR_MODE, get_vector_mode
from hdc.binary_hdc import HyperDimensionalComputingBinary
from utils.person_data_normalization import parse_date, normalize_person_data
from hdc.bipolar_hdc import HyperDimensionalComputingBipolar

# Global dictionary to cache hypervectors
hv_dict = {}


# --- PyTorch Bit Packing Helpers (Replacing NumPy packbits/unpackbits) ---

def _bipolar_to_binary_bytes(hv: torch.Tensor) -> bytes:
    """
    Converts a bipolar vector ({-1, 1}) to packed binary bytes using PyTorch.
    """
    # Ensure we are working with a 1D CPU tensor of 0s and 1s
    bits = (hv.detach().cpu() > 0).to(torch.uint8)

    # Calculate padding to make it a multiple of 8
    pad = (8 - (bits.shape[0] % 8)) % 8
    if pad > 0:
        bits = torch.cat([bits, torch.zeros(pad, dtype=torch.uint8)])

    # Reshape to groups of 8 bits
    bits = bits.view(-1, 8)

    # Create weights for bit-shifting (big-endian: 128, 64, 32, 16, 8, 4, 2, 1)
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)

    # Multiply and sum to get packed bytes
    packed = (bits * weights).sum(dim=1).to(torch.uint8)
    return bytes(packed.tolist())


def _binary_bytes_to_bipolar(b: bytes, dim: int) -> torch.Tensor:
    """
    Unpacks bytes into a bipolar {-1, 1} PyTorch tensor.
    """
    # Convert bytes to a uint8 tensor
    byte_tensor = torch.tensor(list(b), dtype=torch.uint8)

    # Extract bits using bitwise right shift and mask
    # We want to extract bits from 7 down to 0 for big-endian
    bits = []
    for i in range(7, -1, -1):
        bits.append((byte_tensor >> i) & 1)

    # Stack and flatten, then trim to original dimension
    unpacked = torch.stack(bits, dim=1).view(-1)[:dim]

    # Map 1 -> 1 and 0 -> -1
    return torch.where(unpacked == 1,
                       torch.tensor(1, dtype=torch.int8),
                       torch.tensor(-1, dtype=torch.int8))


def _encode_for_milvus(hv: torch.Tensor) -> Union[bytes, List[float]]:
    """
    Prepares vector for Milvus based on VECTOR_MODE using PyTorch.
    """
    vector_mode = get_vector_mode()

    # Ensure it's a tensor
    if not isinstance(hv, torch.Tensor):
        hv = torch.tensor(hv)

    hv = hv.detach().cpu()

    if vector_mode == "binary":
        # Para vectores binarios, asegurarse de que estamos mandando bits
        # Verificar dimensionalidad
        if hv.numel() != DIMENSION:
            print(f"[WARNING] Vector dimensión incorrecta: {hv.numel()}, esperado: {DIMENSION}")
            # Ajustar dimensión si es necesario
            if hv.numel() < DIMENSION:
                # Padding
                padding = DIMENSION - hv.numel()
                hv = torch.cat([hv, torch.zeros(padding, dtype=hv.dtype)])
            else:
                # Truncar
                hv = hv[:DIMENSION]

        # Convertir a bits (0/1)
        binary_hv = (hv > 0).to(torch.uint8)
        return _bipolar_to_binary_bytes(binary_hv)
    else:  # vector_mode == "float"
        # Para vectores float, asegurarse de que sean float32
        float_hv = hv.float()

        # Verificar dimensionalidad
        if float_hv.numel() != DIMENSION:
            print(f"[WARNING] Vector dimensión incorrecta: {float_hv.numel()}, esperado: {DIMENSION}")
            # Ajustar dimensión si es necesario
            if float_hv.numel() < DIMENSION:
                # Padding
                padding = DIMENSION - float_hv.numel()
                float_hv = torch.cat([float_hv, torch.zeros(padding, dtype=torch.float32)])
            else:
                # Truncar
                float_hv = float_hv[:DIMENSION]

        # Asegurarse de que el tipo sea correcto (float32)
        return float_hv.tolist()


# --- Attribute Helpers ---

def _split_attrs(attrs: Dict[str, Any] | None):
    attrs = attrs or {}
    return attrs.get("address", []), attrs.get("akas", []), attrs.get("landlines", [])


def _merge_attrs(person: Dict[str, Any]) -> Dict[str, Any]:
    """Move list-like fields into one JSON bag for Milvus"""
    if "attrs" in person and isinstance(person["attrs"], dict):
        result = {
            "address": person["attrs"].get("address", []) or [],
            "akas": person["attrs"].get("akas", []) or [],
            "landlines": person["attrs"].get("landlines", []) or [],
        }
    else:
        result = {
            "address": person.get("address", []) or [],
            "akas": person.get("akas", []) or [],
            "landlines": person.get("landlines", []) or [],
        }
    return result


# --- Encoding Factory Functions ---

def encode_date(date_obj, mode="binary"):
    hdc_bipolar = HyperDimensionalComputingBipolar()
    hdc_binary = HyperDimensionalComputingBinary()
    if mode == "binary":
        return hdc_binary.encode_date_binary(date_obj)
    else:
        return hdc_bipolar.encode_date_bipolar(date_obj)


def encode_person(person, mode="binary", field_weights=None, excluded_fields=None):
    hdc_bipolar = HyperDimensionalComputingBipolar(dim=HDC_DIM)
    hdc_binary = HyperDimensionalComputingBinary(dim=HDC_DIM)

    vector_mode = get_vector_mode()
    if vector_mode == "binary":
        return hdc_binary.encode_person_binary(
            person,
            field_weights=field_weights,
            excluded_fields=excluded_fields,
        )
    elif vector_mode == "float":
        return hdc_bipolar.encode_person_generalized(
            person,
            field_weights=field_weights,
            excluded_fields=excluded_fields,
        )
    else:
        raise ValueError(f"Invalid vector mode: {vector_mode}")


# --- Database Operations ---
def store_person(
    person: Dict[str, Any],
    collection_name: str = "people",
    field_weights=None,
    excluded_fields=None,
) -> int:
    person_data = person.copy()
    embedding = person_data.pop('embedding', None)

    if 'attrs' not in person_data or not isinstance(person_data['attrs'], dict):
        person_data['attrs'] = {}

    for attr_key in ['address', 'akas', 'landlines']:
        if attr_key in person_data:
            person_data['attrs'][attr_key] = person_data.pop(attr_key)

    normalized_person = normalize_person_data(person_data)
    hv = encode_person(normalized_person, field_weights=field_weights, excluded_fields=excluded_fields)
    
    # Usar _encode_for_milvus para preparar el vector en el formato correcto
    milvus_vector = _encode_for_milvus(hv)
    
    doc_to_insert = {
        **normalized_person,
        "hv": milvus_vector,
    }

    if "dob" in doc_to_insert:
        d = doc_to_insert["dob"]
        if isinstance(d, (date_cls, datetime)):
            doc_to_insert["dob"] = d.strftime("%Y-%m-%d")
        elif d is None:
            doc_to_insert["dob"] = ""
        else:
            doc_to_insert["dob"] = str(d)

    if embedding is not None:
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        doc_to_insert["embedding"] = embedding.detach().cpu().float().tolist()[:128]

    col = ensure_people_collection(collection_name)
    schema_fields = {f.name: f for f in col.schema.fields}

    if "embedding" in schema_fields and "embedding" not in doc_to_insert:
        emb_field = schema_fields["embedding"]
        dim = 128
        if hasattr(emb_field, 'params') and 'dim' in emb_field.params:
            dim = int(emb_field.params['dim'])
        doc_to_insert["embedding"] = [0.0] * dim

    res = col.insert(doc_to_insert)
    return res.primary_keys[0] if hasattr(res, 'primary_keys') else res.inserted_ids[0]


def get_person_details(person_id: int, collection_name: str = "people") -> Dict[str, Any]:
    col = ensure_people_collection(collection_name)
    output_fields = ["name", "lastname", "dob", "marital_status", "mobile_number", "gender", "race", "attrs", "id"]

    schema_fields = {f.name for f in col.schema.fields}
    if "embedding" in schema_fields:
        output_fields.append("embedding")

    res = col.query(expr=f"id == {person_id}", output_fields=output_fields, consistency_level="Strong")
    if not res: return None

    r = res[0]
    address, akas, landlines = _split_attrs(r.get("attrs", {}))

    return {
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
        "embedding": r.get("embedding")
    }


def find_closest_match_db(query_person, threshold=0.7, limit=5, collection_name: str = "people"):
    col = ensure_people_collection(collection_name)
    normalized_query = normalize_person_data(query_person)
    current_mode = get_vector_mode()
    qhv = encode_person(normalized_query)
    qpayload = _encode_for_milvus(qhv)

    if current_mode == "binary":
        search_params = {"metric_type": "HAMMING", "params": {}}
        metric = "HAMMING"
    else:
        search_params = {"metric_type": "IP", "params": {"ef": 128}}
        metric = "IP"

    results = col.search(
        data=[qpayload],
        anns_field="hv",
        param=search_params,
        limit=limit * 3,
        output_fields=["name", "lastname", "dob", "gender"],
    )
    hits = results[0] if results else []

    out = []
    for h in hits:
        if metric == "HAMMING":
            sim = 1.0 - (h.distance / float(DIMENSION))
        else:
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
    out.sort(key=lambda x: x["similarity"], reverse=True)
    return out[:limit]


def find_similar_by_date(target_date, range_days=30, limit=5, collection_name: str = "people"):
    col = ensure_people_collection(collection_name)
    if not isinstance(target_date, (datetime, date_cls)):
        target_date = parse_date(target_date)
    if isinstance(target_date, datetime):
        target_date = target_date.date()
    if target_date is None: return []

    lo = (target_date - timedelta(days=range_days)).strftime("%Y-%m-%d")
    hi = (target_date + timedelta(days=range_days)).strftime("%Y-%m-%d")

    expr = f'dob >= "{lo}" && dob <= "{hi}"'
    res = col.query(expr=expr, output_fields=["id", "name", "lastname", "dob"], consistency_level="Strong", limit=limit)
    res.sort(key=lambda r: r.get("dob", ""))

    return [{"id": int(r["id"]), "name": r["name"], "lastname": r["lastname"], "dob": r["dob"]} for r in res][:limit]