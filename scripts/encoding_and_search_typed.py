import numpy as np
import pandas as pd
import psycopg2
import pickle
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
import random
import ast
from datetime import datetime, date
import re
from psycopg2.extensions import register_adapter, AsIs
from psycopg2.extras import Json, DictCursor

# Set dimension for hypervectors
DIMENSION = 10000

# Global dictionary to cache hypervectors
hv_dict = {}


def deterministic_hash(key):
    """Generate a reproducible integer hash from any key"""
    # Convert key to string and encode to bytes
    key_bytes = str(key).encode('utf-8')
    # Create hash
    hash_obj = hashlib.md5(key_bytes)
    # Convert to integer using first 8 bytes
    hash_int = int.from_bytes(hash_obj.digest()[:8], byteorder='little')
    return hash_int


def get_hv(key):
    """Get a reproducible hypervector for a key based on its hash"""
    key_str = str(key)
    if key_str not in hv_dict:
        # Use hash determined by the key as seed
        seed = deterministic_hash(key_str) % (2 ** 32)  # Ensure within range accepted by numpy
        # Create random number generator initialized with the seed
        rng = np.random.RandomState(seed)
        # Generate hypervector
        hv_dict[key_str] = rng.choice([-1, 1], DIMENSION)
    return hv_dict[key_str]


def encode_date(date_obj):
    """Special encoding for date objects that preserves semantic meaning"""
    if not date_obj:
        return np.zeros(DIMENSION)

    # Get basic encoding for the full date
    base_encoding = get_hv(str(date_obj))

    # Also encode year and month separately to allow for approximate matching
    year_encoding = get_hv(f"year_{date_obj.year}") * 0.5  # Lower weight
    month_encoding = get_hv(f"month_{date_obj.month}") * 0.3  # Lower weight

    # Combine all encodings
    return base_encoding + year_encoding + month_encoding


def encode_person(person):
    """Encode a person's data into a hypervector, with special handling for dates and numeric values"""
    components = []

    # Sort keys to ensure consistent order
    for key in sorted(person.keys()):
        value = person[key]

        # Handle different data types
        if isinstance(value, list):
            # For lists (e.g., multiple addresses)
            if not value:  # Handle empty lists
                encoded_value = np.zeros(DIMENSION)
            else:
                # Sum vectors for each item in the list
                encoded_value = sum(get_hv(str(v)) for v in value)
        elif isinstance(value, (date, datetime)):
            # Special handling for date objects
            encoded_value = encode_date(value)
        elif value is None:
            # Handle None values
            encoded_value = np.zeros(DIMENSION)
        elif key == "annual_income" or isinstance(value, (int, float)) or (
                isinstance(value, str) and value.replace(".", "", 1).isdigit()):
            # Normalize numeric values by converting to a standard format
            # This ensures "100000", 100000, and 100000.00 all encode the same way
            try:
                # Try to convert to float first to handle both integers and decimals
                numeric_value = float(value)
                # For integers or whole numbers, use the integer representation
                if numeric_value.is_integer():
                    normalized_value = int(numeric_value)
                else:
                    normalized_value = numeric_value
                encoded_value = get_hv(str(normalized_value))
            except (ValueError, TypeError):
                # If conversion fails, use the original value
                encoded_value = get_hv(str(value))
        else:
            # Default encoding for other types
            encoded_value = get_hv(str(value))

        # Bind key and value
        components.append(get_hv(key) * encoded_value)

    if not components:  # Handle empty components
        return np.zeros(DIMENSION)

    # Sum all component vectors to create the person representation
    return sum(components)

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="postgres",
    user="uapp",
    password="papp",
    host="localhost",
    port="5433"
)
cursor = conn.cursor()

# Drop and recreate the table with proper data types
cursor.execute("DROP TABLE IF EXISTS people_typed")
cursor.execute("""
               CREATE TABLE people_typed
               (
                   id             SERIAL PRIMARY KEY,
                   name           TEXT,
                   lastname       TEXT,
                   dob            DATE,  -- Proper date type
                   address        TEXT[],
                   marital_status TEXT,
                   akas           TEXT[],
                   landlines      TEXT[],
                   mobile_number  TEXT,
                   gender         TEXT,
                   race           TEXT,
                   annual_income  NUMERIC(12,2),
                   hdv            BYTEA, -- Store HDV as binary
                   created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )
               """)
conn.commit()


def parse_date(date_str):
    """Parse date strings to date objects"""
    if not date_str or date_str == '':
        return None

    # Try standard ISO format first
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        pass

    # Try other common formats
    formats = [
        "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%m-%d-%Y", "%d-%m-%Y",
        "%b %d, %Y", "%B %d, %Y",
        "%d %b %Y", "%d %B %Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

    # If no standard format works, try to extract with regex
    # This is a simple regex that might need adjustment based on your data
    date_match = re.search(r'(\d{4})[-/]?(\d{1,2})[-/]?(\d{1,2})', date_str)
    if date_match:
        try:
            year, month, day = map(int, date_match.groups())
            return date(year, month, day)
        except (ValueError, OverflowError):
            pass

    # Return None if date can't be parsed
    print(f"Warning: Could not parse date '{date_str}'")
    return None


def parse_list_string(list_str):
    """Parse list strings from CSV"""
    if not list_str or list_str == '[]':
        return []
    try:
        return ast.literal_eval(list_str)
    except:
        return [list_str]


def normalize_person_data(person):
    """Normalize person data to handle format inconsistencies, convert types, and standardize key case"""
    normalized = {}
    for key, value in person.items():
        # Standardize key to lowercase
        normalized_key = key.lower()

        # Handle list strings
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            normalized[normalized_key] = parse_list_string(value)
        # Convert date strings to date objects
        elif normalized_key == 'dob' and value and not isinstance(value, (date, datetime)):
            normalized[normalized_key] = parse_date(value)
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
                   INSERT INTO people_typed (name, lastname, dob, address, marital_status, akas, landlines,
                                             mobile_number, gender, race, annual_income, hdv)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                   """, (
                       normalized_data.get("name", ""),
                       normalized_data.get("lastname", ""),
                       normalized_data.get("dob"),  # Use lowercase consistently
                       normalized_data.get("address", []),
                       normalized_data.get("marital_status", ""),
                       normalized_data.get("akas", []),  # Use lowercase consistently
                       normalized_data.get("landlines", []),
                       normalized_data.get("mobile_number", ""),
                       normalized_data.get("gender", ""),
                       normalized_data.get("race", ""),
                       normalized_data.get("annual_income", None),
                       psycopg2.Binary(hdv_binary)
                   ))

    person_id = cursor.fetchone()[0]
    conn.commit()
    return person_id


def load_database(csv_file):
    """Load database from CSV file"""
    df = pd.read_csv(csv_file)

    # Convert DOB column to datetime if it exists
    if 'dob' in df.columns:
        # Keep original string for reference
        df['dob_original'] = df['dob']
        # Parse dates
        df['dob'] = df['dob'].apply(parse_date)

    # Process each row
    ids = []
    for _, row in df.iterrows():
        person_id = store_person(row.to_dict())
        ids.append(person_id)

    print(f"Database successfully stored in PostgreSQL! {len(ids)} records inserted.")
    return ids


def find_closest_match_db(query_person, threshold=0.7, limit=5):
    """Find the closest matches to a query person"""
    # Normalize query data
    normalized_query = normalize_person_data(query_person)

    # Encode query
    query_vector = encode_person(normalized_query)

    # Retrieve all HDVs with more complete information
    cursor.execute("""
                   SELECT id, name, lastname, dob, gender, hdv
                   FROM people_typed
                   """)
    results = cursor.fetchall()

    matches = []

    for person_id, name, lastname, dob, gender, hdv_binary in results:
        # Unpickle the HDV
        stored_hdv = pickle.loads(hdv_binary)

        # Compute similarity using cosine similarity
        score = cosine_similarity([query_vector], [stored_hdv])[0][0]

        # Add to matches if above threshold
        if score >= threshold:
            matches.append({
                "id": person_id,
                "name": name,
                "lastname": lastname,
                "dob": dob,
                "gender": gender,
                "similarity": score
            })

    # Sort by similarity score (highest first)
    matches.sort(key=lambda x: x["similarity"], reverse=True)

    # Return top matches up to limit
    return matches[:limit]


def get_person_details(person_id):
    """Get complete details for a person by ID"""
    cursor.execute("""
                   SELECT id,
                          name,
                          lastname,
                          dob,
                          address,
                          marital_status,
                          akas,
                          landlines,
                          mobile_number,
                          gender,
                          race,
                          annual_income
                   FROM people_typed
                   WHERE id = %s
                   """, (person_id,))

    result = cursor.fetchone()

    if not result:
        return None

    # Convert to dictionary
    columns = ["id", "name", "lastname", "dob", "address", "marital_status",
               "akas", "landlines", "mobile_number", "gender", "race", "annual_income"]

    return dict(zip(columns, result))


def find_similar_by_date(target_date, range_days=30, limit=5):
    """Find people with dates within a certain range of days"""
    if not isinstance(target_date, (date, datetime)):
        target_date = parse_date(target_date)

    if not target_date:
        return []

    # First try direct SQL query for date range (efficient)
    cursor.execute("""
                   SELECT id, name, lastname, dob
                   FROM people_typed
                   WHERE dob BETWEEN %s AND %s
                   ORDER BY dob
                       LIMIT %s
                   """, (
                       target_date - pd.Timedelta(days=range_days),
                       target_date + pd.Timedelta(days=range_days),
                       limit
                   ))

    results = cursor.fetchall()

    return [{"id": id, "name": name, "lastname": lastname, "dob": dob}
            for id, name, lastname, dob in results]


# For testing and example usage
if __name__ == "__main__":
    # Example of loading a CSV file
    print("To load data from a CSV file, use:")
    print("ids = load_database('your_file.csv')")
    
    # Example of using the find functions
    print("\nTo search for similar people:")
    print("matches = find_closest_match_db({'name': 'John', 'DOB': '1990-01-01'})")
    
    print("\nTo search by date:")
    print("date_matches = find_similar_by_date('1990-05-15', range_days=30)")