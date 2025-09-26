import numpy as np
import pandas as pd
import psycopg2
import pickle
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
import random
import ast

# Establece dimensiones de los hypervectores
DIMENSION = 10000


# Dictionario global para cachear los
hv_dict = {}

def deterministic_hash(key):
    """Generar un hash entero reproducible a partir de cualquier clave"""
    # Conversion de la clave a string y codificación a bytes
    key_bytes = str(key).encode('utf-8')
    # Creacion de un hash
    hash_obj = hashlib.md5(key_bytes)
    # Conversion a entero usando los primeros 8 bytes
    hash_int = int.from_bytes(hash_obj.digest()[:8], byteorder='little')
    return hash_int

def get_hv(key):
    """
    Obtener un hipervector reproducible para una clave a partir de su hash
    """
    key_str = str(key)
    if key_str not in hv_dict:
        # Uso de hash determinado por la clave como semilla
        seed = deterministic_hash(key_str) % (2**32) # chequea que esté dentro del rango aceptado x numpy
        # Creacion de un generador de números aleatorios inicializado con la semilla
        rng = np.random.RandomState(seed)
        # Genera el hypervector
        hv_dict[key_str] = rng.choice([-1, 1], DIMENSION)
    return hv_dict[key_str]

def encode_person(person):
    """Codifica los datos de una persona en un hypervector"""
    components = []

    # ordena las claves para asegurar orden consistente
    for key in sorted(person.keys()):
        value = person[key]
        # si el valor es una lista (x ejemplo multiples direcciones)
        if isinstance(value, list): # necesito probar la decodificación de esto!
            if not value:  # Manejo de listas vacías
                encoded_value = np.zeros(DIMENSION)
            else:
                encoded_value = sum(get_hv(str(v)) for v in value) # sumo todos los vectores correspondientes a cada item de la lista
        else:
            encoded_value = get_hv(str(value))

        components.append(get_hv(key) * encoded_value)  # Binding de clave y valor

    if not components:  # Manejo de componentes vacíos (1 componente = clave * valor)
        return np.zeros(DIMENSION)

    return sum(components)  # Ahora sumo todos los vectores correspondientes a los atributos individuales para generar el vector que representa  a la persona

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="postgres",
    user="uapp",
    password="papp",
    host="localhost",
    port="5433"
)
cursor = conn.cursor()

# Drop and recreate the table
cursor.execute("DROP TABLE IF EXISTS people")
cursor.execute("""
CREATE TABLE people (
    id SERIAL PRIMARY KEY,
    name TEXT,
    lastname TEXT,
    dob TEXT,
    address TEXT[],
    marital_status TEXT,
    akas TEXT[],
    landlines TEXT[],
    mobile_number TEXT,
    gender TEXT,
    race TEXT,
    hdv BYTEA  -- Store HDV as binary
)
""")
conn.commit()

def parse_list_string(list_str):
    """Parse list strings from CSV"""
    if not list_str or list_str == '[]':
        return []
    try:
        return ast.literal_eval(list_str)
    except:
        return [list_str]

def normalize_person_data(person):
    """Normalize person data to handle format inconsistencies and standardize key case"""
    normalized = {}
    for key, value in person.items():
        # Standardize key to lowercase
        normalized_key = key.lower()

        # Handle list strings
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            normalized[normalized_key] = parse_list_string(value)
        else:
            normalized[normalized_key] = value

    return normalized

def store_person(person):
    """Store a person in the database with their HDV"""
    # Normalize data
    normalized_data = normalize_person_data(person)

    # Generate HDV
    hdv = encode_person(normalized_data)
    hdv_binary = pickle.dumps(hdv)

    # Insert into database
    cursor.execute("""
        INSERT INTO people (name, lastname, dob, address, marital_status, akas, landlines, mobile_number, gender, race, hdv)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        normalized_data.get("name", ""),
        normalized_data.get("lastname", ""),
        normalized_data.get("dob", ""),
        normalized_data.get("address", []),
        normalized_data.get("marital_status", ""),
        normalized_data.get("akas", []),
        normalized_data.get("landlines", []),
        normalized_data.get("mobile_number", ""),
        normalized_data.get("gender", ""),
        normalized_data.get("race", ""),
        psycopg2.Binary(hdv_binary)
    ))
    conn.commit()

def load_database(csv_file):
    """Load database from CSV file"""
    df = pd.read_csv(csv_file)

    # Process each row
    for _, row in df.iterrows():
        store_person(row.to_dict())

    print("Database successfully stored in PostgreSQL!")

def find_closest_match_db(query_person, threshold=0.7):
    """Find the closest match to a query person"""
    # Normalize query data
    normalized_query = normalize_person_data(query_person)

    # Encode query
    query_vector = encode_person(normalized_query)

    # Retrieve all HDVs
    cursor.execute("SELECT id, name, lastname, hdv FROM people")
    results = cursor.fetchall()

    best_match = None
    best_score = 0

    print("\n--- Debugging: Comparing Stored vs Computed HDVs ---")

    for person_id, name, lastname, hdv_binary in results:
        # Unpickle the HDV
        stored_hdv = pickle.loads(hdv_binary)

        # Debug only for known target
        if name.lower() == "danielle" and lastname.lower() == "rhodes":
            print(f"\nChecking {name} {lastname}:")

            # Compare vectors directly
            are_equal = np.array_equal(query_vector, stored_hdv)
            print(f"Vectors are equal: {are_equal}")

            if not are_equal:
                # Check specific differences
                diff_count = np.sum(query_vector != stored_hdv)
                print(f"Number of different elements: {diff_count} out of {DIMENSION}")

                # Show sample values
                print("First 5 query vector values:", query_vector[:5])
                print("First 5 stored vector values:", stored_hdv[:5])

                # Check cosine similarity
                sim = cosine_similarity([query_vector], [stored_hdv])[0][0]
                print(f"Cosine similarity between vectors: {sim}")

        # Compute similarity using cosine similarity
        score = cosine_similarity([query_vector], [stored_hdv])[0][0]

        if score > best_score:
            best_score = score
            best_match = {"id": person_id, "name": name, "lastname": lastname}

    if best_score < threshold:
        return None, best_score

    return best_match, best_score

# --- MAIN ---

# Verify encoding consistency
# verify_encoding_consistency()

# Load DB from CSV
csv_file = "test_synthetic_dataset.csv"
load_database(csv_file)

# Example query
query_entries = [
    {"name": "Danielle", "lastname": "Rhodes", "DOB": "1950-03-03",
     "address": ["9402 Peterson Drives, Port Matthew, CO 50298", "407 Teresa Lane Apt. 849, Barbaraland, AZ 87174"],
     "marital_status": "Divorced", "Akas": ["Sonya Johnston", "John Russell"], "landlines": ["538.372.6247"],
     "mobile_number": "(613)354-2784x980", "gender": "Other", "race": "Other"}
]

for query in query_entries:
    best_match, similarity_score = find_closest_match_db(query)
    if best_match:
        print("Closest match found:", best_match)
        print("Similarity Score:", similarity_score)
    else:
        print("No good match found. Closest score:", similarity_score)