from pymilvus import connections, Collection, DataType
from database_utils.milvus_db_connection import connect, ALIAS, COLLECTION


def get_collection_schema(collection_name=COLLECTION, connect_first=True):
    """
    Returns the schema of a Milvus collection.

    Args:
        collection_name (str): Name of the collection to get schema for
        connect_first (bool): Whether to establish connection before querying

    Returns:
        The schema object of the collection
    """
    if connect_first:
        connect()

    collection = Collection(collection_name, using=ALIAS)
    return collection.schema


def print_collection_schema(collection_name=COLLECTION, format="table", connect_first=True):
    """
    Prints the schema of a Milvus collection in a readable format.

    Args:
        collection_name (str): Name of the collection to show schema for
        format (str): Output format, either "table" or "list"
        connect_first (bool): Whether to establish connection before querying

    Returns:
        None - prints the schema to stdout
    """
    if connect_first:
        connect()

    collection = Collection(collection_name, using=ALIAS)
    schema = collection.schema

    # Create a mapping of numeric values to their names
    dtype_names = {getattr(DataType, name): name
                   for name in dir(DataType)
                   if not name.startswith('_') and isinstance(getattr(DataType, name), int)}

    # Format: table (more compact)
    if format.lower() == "table":
        print(f"\nEstructura de la colección '{collection_name}':")
        print("-" * 80)
        print(f"{'Campo':<20} {'Tipo':<15} {'Propiedades':<30}")
        print("-" * 80)

        for field in schema.fields:
            dtype_name = dtype_names.get(field.dtype, "Unknown")

            properties = []
            if field.is_primary:
                properties.append("PRIMARY KEY")
            if field.auto_id:
                properties.append("AUTO_ID")

            if field.dtype == DataType.VARCHAR:
                # Check if params is a dictionary or an object
                if isinstance(field.params, dict):
                    if 'max_length' in field.params:
                        properties.append(f"max_length={field.params['max_length']}")
                else:
                    # Try accessing as an attribute
                    try:
                        properties.append(f"max_length={field.params.max_length}")
                    except AttributeError:
                        pass  # Skip if attribute doesn't exist

            if field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
                # Same check for dim attribute
                if isinstance(field.params, dict):
                    if 'dim' in field.params:
                        properties.append(f"dim={field.params['dim']}")
                else:
                    try:
                        properties.append(f"dim={field.params.dim}")
                    except AttributeError:
                        pass

            properties_str = ", ".join(properties)
            print(f"{field.name:<20} {dtype_name:<15} {properties_str:<30}")

        print("-" * 80)
        print(f"Total de campos: {len(schema.fields)}")

    # Format: list (more detailed)
    else:
        print(f"\nCampos en la colección '{collection_name}':\n")
        for field in schema.fields:
            print(f"Nombre: {field.name}")
            print(f"Tipo: {field.dtype} ({dtype_names.get(field.dtype, 'Unknown')})")
            print(f"Descripción: {field.description}")
            print(f"Es primario: {field.is_primary}")
            print(f"AutoID: {field.auto_id}")

            # Print additional parameters based on field type
            if field.dtype == DataType.VARCHAR:
                # Check if params is a dictionary or an object
                if isinstance(field.params, dict):
                    if 'max_length' in field.params:
                        print(f"Longitud máxima: {field.params['max_length']}")
                else:
                    try:
                        print(f"Longitud máxima: {field.params.max_length}")
                    except AttributeError:
                        pass

            if field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
                # Same check for dim attribute
                if isinstance(field.params, dict):
                    if 'dim' in field.params:
                        print(f"Dimensión: {field.params['dim']}")
                else:
                    try:
                        print(f"Dimensión: {field.params.dim}")
                    except AttributeError:
                        pass

            print("-" * 40)


def get_schema_as_dict(collection_name=COLLECTION, connect_first=True):
    """
    Returns the schema of a Milvus collection as a dictionary.
    Useful for programmatic access to schema information.

    Args:
        collection_name (str): Name of the collection
        connect_first (bool): Whether to establish connection before querying

    Returns:
        dict: Dictionary containing schema information
    """
    if connect_first:
        connect()

    collection = Collection(collection_name, using=ALIAS)
    schema = collection.schema

    # Create a mapping of numeric values to their names
    dtype_names = {getattr(DataType, name): name
                   for name in dir(DataType)
                   if not name.startswith('_') and isinstance(getattr(DataType, name), int)}

    result = {
        "collection_name": collection_name,
        "description": schema.description,
        "fields": []
    }

    for field in schema.fields:
        field_info = {
            "name": field.name,
            "type": dtype_names.get(field.dtype, "Unknown"),
            "type_id": field.dtype,
            "is_primary": field.is_primary,
            "auto_id": field.auto_id,
            "description": field.description,
        }

        # Add type-specific parameters
        if field.dtype == DataType.VARCHAR:
            if isinstance(field.params, dict):
                if 'max_length' in field.params:
                    field_info["max_length"] = field.params['max_length']
            else:
                try:
                    field_info["max_length"] = field.params.max_length
                except AttributeError:
                    pass

        if field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
            if isinstance(field.params, dict):
                if 'dim' in field.params:
                    field_info["dim"] = field.params['dim']
            else:
                try:
                    field_info["dim"] = field.params.dim
                except AttributeError:
                    pass

        result["fields"].append(field_info)

    return result


if __name__ == "__main__":
    # Example usage when run as a script
    # print_collection_schema(format="table")

    # Uncomment to see detailed list format
    print_collection_schema(format="list")