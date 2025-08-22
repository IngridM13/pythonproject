import pandas as pd
import numpy as np
import faker
import random
from tqdm import tqdm
from configs.settings import NUM_ROWS, CHUNK_SIZE, OUTPUT_FILE, DEFAULT_SEED

# Initialize Faker
fake = faker.Faker()
faker.Faker.seed(DEFAULT_SEED)  # Ensure reproducibility

# Predefined lists for faster selection
genders = ["Male", "Female", "Non-binary", "Other"]
marital_statuses = ["Single", "Married", "Divorced", "Widowed"]
races = ["White", "Black", "Asian", "Hispanic", "Mixed", "Other"]

# Function to generate random data
def generate_data_chunk(num_rows):
    data = {
        "name": [fake.first_name() for _ in range(num_rows)],
        "lastname": [fake.last_name() for _ in range(num_rows)],
        "DOB": [fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat() for _ in range(num_rows)],
        "address": [str([fake.address().replace("\n", ", ") for _ in range(random.randint(0, 5))]) for _ in range(num_rows)],
        "marital_status": np.random.choice(marital_statuses, num_rows),
        "Akas": [str([fake.name() for _ in range(random.randint(0, 3))]) for _ in range(num_rows)],
        "landlines": [str([fake.phone_number() for _ in range(random.randint(0, 2))]) for _ in range(num_rows)],
        "mobile_number": [fake.phone_number() for _ in range(num_rows)],
        "gender": np.random.choice(genders, num_rows),
        "race": np.random.choice(races, num_rows),
    }
    return pd.DataFrame(data)

def generate_data_and_save(OUTPUT_FILE=OUTPUT_FILE):
    # Write CSV in chunks
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        first_chunk = True  # To handle header writing

        for _ in tqdm(range(NUM_ROWS // CHUNK_SIZE), desc="Generating dataset", unit="chunk"):
            df_chunk = generate_data_chunk(CHUNK_SIZE)
            df_chunk.to_csv(f, mode="a", header=first_chunk, index=False)
            first_chunk = False  # Only write header for the first chunk

    print(f"Dataset generated: {OUTPUT_FILE} (5M rows)") #TODO BERNIE: En vez de print, hacelo con log.
