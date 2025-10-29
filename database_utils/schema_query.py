from pymilvus import connections, Collection, DataType

# Conectarse al servidor de Milvus
connections.connect("default", uri="http://127.0.0.1:19530")

# Nombre de la colección que qierp inspeccionar
collection_name = "people"

# Crear el objeto de colección
collection = Collection(collection_name)

# Obtener y mostrar el esquema de la colección
schema = collection.schema

print(f"\nCampos en la colección '{collection_name}':\n")
for field in schema.fields:
    print(f"Nombre: {field.name}")
    print(f"Tipo: {field.dtype}: {DataType(field.dtype).name}")
    print(f"Descripción: {field.description}")
    print(f"Es primario: {field.is_primary}")
    print(f"AutoID: {field.auto_id}")
    print(f"Dimensión (si aplica): {getattr(field.params, 'dim', 'N/A')}")
    print("-" * 40)
