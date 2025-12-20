from pymilvus import connections, Collection, DataType
from database_utils.milvus_db_connection import connect, ALIAS, COLLECTION

def get_field_data_type(field_name, collection_name=COLLECTION, connect_first=True):
    """
    Obtiene el tipo de dato numérico de un campo específico en una colección de Milvus.

    Args:
        field_name (str): Nombre del campo del cual se desea obtener el tipo de dato
        collection_name (str): Nombre de la colección que contiene el campo
        connect_first (bool): Indica si se debe establecer la conexión antes de realizar la consulta

    Returns:
        int: El ID numérico del tipo de dato del campo

    Raises:
        ValueError: Si el campo no existe en la colección
    """
    if connect_first:
        connect()

    collection = Collection(collection_name, using=ALIAS)
    schema = collection.schema

    # Buscar el campo en el esquema
    for field in schema.fields:
        if field.name == field_name:
            return field.dtype

    # Si llegamos acá, el campo no fue encontrado
    raise ValueError(f"Campo '{field_name}' no encontrado en la colección '{collection_name}'")

def get_field_data_type_name(field_name, collection_name=COLLECTION, connect_first=True):
    """
    Obtiene el nombre del tipo de dato de un campo específico en una colección de Milvus.
    
    Args:
        field_name (str): Nombre del campo del cual se desea obtener el tipo de dato
        collection_name (str): Nombre de la colección que contiene el campo
        connect_first (bool): Indica si se debe establecer la conexión antes de realizar la consulta
    
    Returns:
        str: El nombre del tipo de dato del campo (por ejemplo, "VARCHAR", "INT64", etc.)
    
    Raises:
        ValueError: Si el campo no existe en la colección
    """
    if connect_first:
        connect()
    
    collection = Collection(collection_name, using=ALIAS)
    schema = collection.schema
    
    # Find the field in the schema
    for field in schema.fields:
        if field.name == field_name:
            return DataType(field.dtype).name
    
    # If we get here, the field wasn't found
    raise ValueError(f"Campo '{field_name}' no encontrado en la colección '{collection_name}'")


if __name__ == "__main__":
    # Example usage demonstrating the function
    try:
        # Test with different field types
        print(f"Field 'id' has type ID: {get_field_data_type('id')}")
        print(f"Field 'name' has type ID: {get_field_data_type('name')}")
        print(f"Field 'attrs' has type ID: {get_field_data_type('attrs')}")
        print(f"Field 'hv' has type ID: {get_field_data_type('hv')}")

        # Test with non-existent field
        print(f"Field 'non_existent_field' has type ID: {get_field_data_type('non_existent_field')}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Test with different field types
        print(f"Field 'id' has type name: {get_field_data_type_name('id')}")
        print(f"Field 'name' has type name: {get_field_data_type_name('name')}")
        print(f"Field 'attrs' has type name: {get_field_data_type_name('attrs')}")
        print(f"Field 'hv' has type name: {get_field_data_type_name('hv')}")

        # Test with non-existent field
        print(f"Field 'non_existent_field' has type name: {get_field_data_type_name('non_existent_field')}")
    except ValueError as e:
        print(f"Error: {e}")