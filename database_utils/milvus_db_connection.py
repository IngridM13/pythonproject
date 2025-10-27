import os
from typing import List
from pymilvus import(connections, utility, Collection, FieldSchema, CollectionSchema, DataType, MilvusException)
from configs.settings import HDC_DIM

# ------ CONFIGURACIÓN ------ Usa os y valores por defecto. esto ayuda con el mantenimiento del código.
ALIAS = "default"
HOST = os.getenv("MILVUS_HOST", "localhost")
PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION = "people"

# Vector mode: "binary" (BINARY_VECTOR + HAMMING) or "float" (FLOAT_VECTOR + COSINE)
VECTOR_MODE = os.getenv("MILVUS_VECTOR_MODE", "binary").lower()

def connect():
    if not connections.has_connection(ALIAS):
        connections.connect(alias=ALIAS, host=HOST, port=PORT)

def ensure_people_collection() -> Collection:
    """
    schema Milvus equivalente a la tabla Postgres previamente definida.
    Arrays (address, akas, landlines) van en el JSON 'attrs'.
    Si existe el collection lo retorne , de lo contrario lo crea con el schema que usamos.
    """
    connect()

    if utility.has_collection(COLLECTION, using=ALIAS):
        col = Collection(COLLECTION, using=ALIAS)
    else:
        fields: List[FieldSchema] = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="lastname", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="dob", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="marital_status", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="mobile_number", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="gender", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="race", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="attrs", dtype=DataType.JSON),  # address/akas/landlines van acá
        ]

        if VECTOR_MODE == "binary":
            if HDC_DIM % 8 != 0:
                raise ValueError("Binary vectors require HDC_DIM to be a multiple of 8.")
            fields.append(FieldSchema(name="hv", dtype=DataType.BINARY_VECTOR, dim=HDC_DIM))
        elif VECTOR_MODE == "float":
            fields.append(FieldSchema(name="hv", dtype=DataType.FLOAT_VECTOR, dim=HDC_DIM))
        else:
            raise ValueError("MILVUS_VECTOR_MODE must be 'binary' or 'float'.")

        schema = CollectionSchema(fields=fields, description="People with hypervectors")
        col = Collection(name=COLLECTION, schema=schema, using=ALIAS)

        # Vector index
        if VECTOR_MODE == "binary":
            col.create_index("hv", {"index_type": "BIN_FLAT", "metric_type": "HAMMING", "params": {}})
        else:
            col.create_index("hv", {"index_type": "HNSW", "metric_type": "COSINE",
                                    "params": {"M": 16, "efConstruction": 200}})

        # index escalar (ocional, si hay soporte para ello en el Milvus build)
        try:
            col.create_index("lastname", {"index_type": "INVERTED"})
        except MilvusException:
            pass

    col.load()
    return col

if __name__ == "__main__":
    col = ensure_people_collection()
    print(f"Collection ready: {col.name}")
    print("Fields:", [f.name for f in col.schema.fields])