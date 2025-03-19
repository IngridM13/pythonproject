import psycopg2
import pickle

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname="postgres",      # Matches POSTGRES_DB in your compose file
    user="uapp",           # Matches POSTGRES_USER
    password="papp",       # Matches POSTGRES_PASSWORD
    host="localhost",      # Running locally via Docker
    port="5433"            # Matches mapped port in Docker
)
cursor = conn.cursor()

# Create the table for storing hyperdimensional vectors (HDVs)
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
