import os
from dotenv import load_dotenv
from typing import List, Optional
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    MilvusException
)
from configs.settings import HDC_DIM

# Cargar variables de entorno
load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = "people"
ALIAS = "default"
VECTOR_MODE = os.getenv("MILVUS_VECTOR_MODE", "binary")  # "binary" or "float"

def connect():
    """Establece la conexión con Milvus si no existe."""
    if not connections.has_connection(ALIAS):
        print(f"Connecting to Milvus at {MILVUS_URI}...")
        connections.connect(alias=ALIAS, uri=MILVUS_URI)

def get_vector_mode():
    return VECTOR_MODE

def ensure_people_collection(collection_name: str = COLLECTION) -> Collection:
    """
    Schema Milvus equivalente a la tabla Postgres previamente definida.
    Arrays (address, akas, landlines) van en el JSON 'attrs'.
    Si existe el collection lo retorna, de lo contrario lo crea con el schema que usamos.
    
    Args:
        collection_name: Nombre de la colección a verificar/crear (default: COLLECTION de settings)
    
    Returns:
        Collection: La colección de Milvus
    """
    connect()

    if utility.has_collection(collection_name, using=ALIAS):
        col = Collection(collection_name, using=ALIAS)
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

        # Campo opcional para embeddings densos (e.g., de redes neuronales convencionales)
        # Asumimos dimensión 128 por ahora basado en los tests
        fields.append(FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="Optional dense embedding"))

        if VECTOR_MODE == "binary":
            print(">>> creando collection para vector_mode == binary")
            if HDC_DIM % 8 != 0:
                raise ValueError("Binary vectors require HDC_DIM to be a multiple of 8.")
            fields.append(FieldSchema(name="hv", dtype=DataType.BINARY_VECTOR, dim=HDC_DIM))
        elif VECTOR_MODE == "float":
            print(">>> creando collection para vector_mode == float")
            fields.append(FieldSchema(name="hv", dtype=DataType.FLOAT_VECTOR, dim=HDC_DIM))
        else:
            print(">>> milvus_db_connection.VECTOR_MODE no reconocido")
            raise ValueError("MILVUS_VECTOR_MODE must be 'binary' or 'float'.")

        schema = CollectionSchema(fields=fields, description="People with hypervectors")
        col = Collection(name=collection_name, schema=schema, using=ALIAS)

        # Vector index for HV
        if VECTOR_MODE == "binary":
            col.create_index("hv", {"index_type": "BIN_IVF_FLAT", "metric_type": "HAMMING", "params": {}})
        else:
            col.create_index("hv", {"index_type": "HNSW", "metric_type": "IP",
                                   "params": {"M": 16, "efConstruction": 200}})
        
        # Vector index for embedding (optional field needs index too to be searchable/loaded)
        # Usamos IVF_FLAT simple para el embedding secundario
        col.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

        # index escalar (opcional, si hay soporte para ello en el Milvus build)
        try:
            col.create_index("lastname", {"index_type": "INVERTED"})
        except MilvusException:
            pass

    col.load()
    return col