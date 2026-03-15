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

# Cache of collection_name -> Collection for collections already set up in this process.
# Avoids repeating index creation and schema validation on every store_person() call.
_collection_cache: dict = {}

def connect():
    """Establece la conexión con Milvus si no existe."""
    if not connections.has_connection(ALIAS):
        print(f"Connecting to Milvus at {MILVUS_URI}...")
        connections.connect(alias=ALIAS, uri=MILVUS_URI)

def get_vector_mode():
    return VECTOR_MODE


def ensure_people_collection(collection_name: str = COLLECTION) -> Collection:
    cache_key = f"{collection_name}_{VECTOR_MODE}"
    if cache_key in _collection_cache:
        return _collection_cache[cache_key]

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

    # Primero verificamos si la colección existe
    if utility.has_collection(collection_name, using=ALIAS):
        try:
            # Intentar cargar la colección existente
            col = Collection(collection_name, using=ALIAS)

            # Obtener la información del esquema para determinar si necesitamos recrear la colección
            fields = col.schema.fields
            field_names = {field.name for field in fields}

            # Verificar tipos de campos para asegurar consistencia con VECTOR_MODE actual
            hv_field = next((f for f in fields if f.name == "hv"), None)
            vector_type_mismatch = False

            if hv_field:
                # Verificar si el tipo de campo coincide con el VECTOR_MODE actual
                if VECTOR_MODE == "binary" and hv_field.dtype != DataType.BINARY_VECTOR:
                    print(f">>> Tipo de vector actual (float) no coincide con el modo deseado (binary)")
                    vector_type_mismatch = True
                elif VECTOR_MODE == "float" and hv_field.dtype != DataType.FLOAT_VECTOR:
                    print(f">>> Tipo de vector actual (binary) no coincide con el modo deseado (float)")
                    vector_type_mismatch = True

            # Si hay inconsistencia de tipos o faltan campos, recreamos la colección
            if vector_type_mismatch:
                print(f">>> Eliminando colección {collection_name} para recrearla con tipo correcto")
                col.release()
                utility.drop_collection(collection_name)
                # La colección se creará más adelante
            else:
                # Intentamos crear los índices necesarios si no hay inconsistencia de tipos
                try:
                    # Crear índice para embedding
                    col.create_index("embedding",
                                     {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
                    print(f">>> Índice para 'embedding' creado en {collection_name}")
                except MilvusException as e:
                    if "index already exists" not in str(e).lower():
                        print(f">>> Error al crear índice para 'embedding': {e}")

                try:
                    # Crear índice para hv según el modo
                    if VECTOR_MODE == "binary":
                        col.create_index("hv", {"index_type": "BIN_IVF_FLAT", "metric_type": "HAMMING", "params": {}})
                    else:  # float
                        col.create_index("hv",
                                         {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}})
                    print(f">>> Índice para 'hv' creado en {collection_name}")
                except MilvusException as e:
                    if "index already exists" not in str(e).lower():
                        print(f">>> Error al crear índice para 'hv': {e}")

                # Cargar la colección
                try:
                    col.load()
                    _collection_cache[f"{collection_name}_{VECTOR_MODE}"] = col
                    return col
                except MilvusException as e:
                    print(f">>> Error al cargar colección: {e}")
                    # Si no podemos cargar, recreamos la colección
                    col.release()
                    utility.drop_collection(collection_name)
                    # La colección se creará más adelante
        except MilvusException as e:
            print(f">>> Error al procesar colección existente: {e}")
            try:
                utility.drop_collection(collection_name)
            except MilvusException:
                pass
            # La colección se creará más adelante

    # Si llegamos aquí, necesitamos crear una nueva colección
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
    fields.append(FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128,
                              description="Optional dense embedding"))

    # Campo para hypervector según el modo
    if VECTOR_MODE == "binary":
        print(f">>> Creando colección {collection_name} con campo 'hv' como BINARY_VECTOR")
        if HDC_DIM % 8 != 0:
            raise ValueError(f"Binary vectors require HDC_DIM to be a multiple of 8. Current: {HDC_DIM}")
        fields.append(FieldSchema(name="hv", dtype=DataType.BINARY_VECTOR, dim=HDC_DIM))
    else:  # float
        print(f">>> Creando colección {collection_name} con campo 'hv' como FLOAT_VECTOR")
        fields.append(FieldSchema(name="hv", dtype=DataType.FLOAT_VECTOR, dim=HDC_DIM))

    # Crear la colección con el esquema definido
    schema = CollectionSchema(fields=fields, description="People with hypervectors")
    col = Collection(name=collection_name, schema=schema, using=ALIAS)

    # Crear índices para los campos vectoriales
    print(">>> Creando índices...")
    if VECTOR_MODE == "binary":
        col.create_index("hv", {"index_type": "BIN_IVF_FLAT", "metric_type": "HAMMING", "params": {}})
    else:  # float
        col.create_index("hv", {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}})

    col.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

    # Índice opcional para texto
    try:
        col.create_index("lastname", {"index_type": "INVERTED"})
    except MilvusException as e:
        print(f">>> No se pudo crear índice para 'lastname': {str(e)}")

    # Cargar colección
    print(f">>> Cargando colección {collection_name}...")
    col.load()

    _collection_cache[f"{collection_name}_{VECTOR_MODE}"] = col
    return col
