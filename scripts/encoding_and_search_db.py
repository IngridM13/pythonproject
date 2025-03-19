import numpy as np
import pandas as pd
import psycopg2
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import random

# Set a fixed dimension for hypervectors
DIMENSION = 10000

# Generate a consistent dictionary of random hypervectors
random.seed(42)
hv_dict = {}


def get_hv(key):
    if key not in hv_dict:
        hv_dict[key] = np.random.choice([-1, 1], DIMENSION)  # Binary bipolar representation
    return hv_dict[key]

def normalize_hv(hv):
    return hv / np.linalg.norm(hv)


# Function to encode a person's data into an HDV
def encode_person(person):
    components = []

    for key, value in person.items():
        if isinstance(value, list):
            encoded_value = sum(get_hv(str(v)) for v in value)  # Bundle list elements
        else:
            encoded_value = get_hv(str(value))

        components.append(get_hv(key) * encoded_value)  # Bind key and value

    return normalize_hv(sum(components))  # Bundle all components


# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname="postgres",  # Matches POSTGRES_DB in docker-compose
    user="uapp",  # Matches POSTGRES_USER
    password="papp",  # Matches POSTGRES_PASSWORD
    host="localhost",  # Running locally via Docker
    port="5433"  # Matches mapped port in Docker
)
cursor = conn.cursor()

# Create table for storing HDVs
cursor.execute("""
CREATE TABLE IF NOT EXISTS people (
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
    hdv BYTEA
)
""")
conn.commit()


# Function to store a person in the database
def store_person(person, hdv):
    """Store a person's data and encoded hyperdimensional vector in PostgreSQL."""
    hdv_binary = pickle.dumps(hdv)  # Serialize HDV to binary format

    # Ensure lists are stored as proper PostgreSQL arrays
    def format_list(value):
        return value if isinstance(value, list) else [value]  # Ensure all values are lists

    cursor.execute("""
        INSERT INTO people (name, lastname, dob, address, marital_status, akas, landlines, mobile_number, gender, race, hdv)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        person["name"],
        person["lastname"],
        person["DOB"],
        format_list(person["address"]),  # Convert to PostgreSQL array
        person["marital_status"],
        format_list(person["Akas"]),     # Convert to PostgreSQL array
        format_list(person["landlines"]),# Convert to PostgreSQL array
        person["mobile_number"],
        person["gender"],
        person["race"],
        hdv_binary
    ))
    conn.commit()



# Load database from CSV file
def load_database(csv_file):
    df = pd.read_csv(csv_file)
    people_db = df.to_dict(orient='records')

    for person in people_db:
        hdv = encode_person(person)  # Encode as HDV
        store_person(person, hdv)  # Store in database

    print("Database successfully stored in PostgreSQL!")


# Function to find closest match in the database
def find_closest_match_db(query_person, threshold=0.7):
    query_vector = encode_person(query_person)

    cursor.execute("SELECT id, name, lastname, hdv FROM people")
    results = cursor.fetchall()

    best_match = None
    best_score = 0

    print("\n--- Debugging: Comparing Stored vs Computed HDVs ---")

    for person_id, name, lastname, hdv_binary in results:
        stored_hdv = pickle.loads(hdv_binary)

        # Debugging: Check stored vs computed HDV for a specific name
        if name == "Danielle" and lastname == "Rhodes":
            print(f"\nComparing HDVs for {name} {lastname}:")
            print(f"Stored HDV (first 10 dims): {stored_hdv[:10]}")
            print(f"Query HDV (first 10 dims): {query_vector[:10]}")

        # Normalize before computing cosine similarity
        score = cosine_similarity([normalize_hv(query_vector)], [normalize_hv(stored_hdv)])[0][0]

        if score > best_score:
            best_score = score
            best_match = {"id": person_id, "name": name, "lastname": lastname}

    if best_score < threshold:
        return None, best_score

    return best_match, best_score


# Load the database
# csv_file = "synthetic_dataset.csv"
csv_file = "test_synthetic_dataset.csv"
load_database(csv_file)

# Example queries (iterate over multiple entries) what about order sensitivity?
query_entries = [
    {"name": "Danielle", "lastname": "Rhodes", "DOB": "1950 - 03 - 03",
     "address": ["9402 Peterson Drives, Port Matthew, CO 50298", "407 Teresa Lane Apt. 849, Barbaraland, AZ 87174"],
     "marital_status": "Divorced", "Akas": ["Sonya Johnston", "John Russel"], "landlines": ["538.372.6247"],
     "mobile_number": "(613)354-2784x980", "gender": "Other", "race": "Other"}
]



for query in query_entries:
    best_match, similarity_score = find_closest_match_db(query)
    if best_match:
        print("Closest match found:", best_match)
        print("Similarity Score:", similarity_score)
    else:
        print("No good match found, the closest has a similarity score below the threshold:", similarity_score)
