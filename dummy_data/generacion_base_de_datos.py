import pandas as pd
import torch
import faker
import random
from tqdm import tqdm
from configs.settings import NUM_ROWS, CHUNK_SIZE, OUTPUT_FILE, DEFAULT_SEED

# Initialize Faker
fake = faker.Faker()
faker.Faker.seed(DEFAULT_SEED)  # Ensure reproducibility
random.seed(DEFAULT_SEED)  # Ensure reproducibility for other random operations
torch.manual_seed(DEFAULT_SEED)  # Set PyTorch seed for reproducibility

# Predefined lists for faster selection
genders = ["Male", "Female", "Non-binary", "Other"]
marital_statuses = ["Single", "Married", "Divorced", "Widowed"]
races = ["White", "Black", "Asian", "Hispanic", "Mixed", "Other"]


# Function to generate random data
def generate_data_chunk(num_rows):
    # Generate lists for all fields
    names = [fake.first_name() for _ in range(num_rows)]
    lastnames = [fake.last_name() for _ in range(num_rows)]
    dobs = [fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat() for _ in range(num_rows)]
    addresses = [str([fake.address().replace("\n", ", ") for _ in range(random.randint(0, 5))]) for _ in
                 range(num_rows)]
    akas = [str([fake.name() for _ in range(random.randint(0, 3))]) for _ in range(num_rows)]
    landlines = [str([fake.phone_number() for _ in range(random.randint(0, 2))]) for _ in range(num_rows)]
    mobile_numbers = [fake.phone_number() for _ in range(num_rows)]

    # Use PyTorch for random choices
    gender_indices = torch.randint(0, len(genders), (num_rows,))
    marital_indices = torch.randint(0, len(marital_statuses), (num_rows,))
    race_indices = torch.randint(0, len(races), (num_rows,))

    # Convert indices to categorical values
    genders_selected = [genders[idx.item()] for idx in gender_indices]
    marital_selected = [marital_statuses[idx.item()] for idx in marital_indices]
    races_selected = [races[idx.item()] for idx in race_indices]

    # Create the data dictionary
    data = {
        "name": names,
        "lastname": lastnames,
        "DOB": dobs,
        "address": addresses,
        "marital_status": marital_selected,
        "Akas": akas,
        "landlines": landlines,
        "mobile_number": mobile_numbers,
        "gender": genders_selected,
        "race": races_selected,
    }

    return pd.DataFrame(data)


def generate_data_and_save(output_file=OUTPUT_FILE):
    # Write CSV in chunks
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        first_chunk = True  # To handle header writing

        for _ in tqdm(range(NUM_ROWS // CHUNK_SIZE), desc="Generating dataset", unit="chunk"):
            df_chunk = generate_data_chunk(CHUNK_SIZE)
            df_chunk.to_csv(f, mode="a", header=first_chunk, index=False)
            first_chunk = False  # Only write header for the first chunk

    # Use logging instead of print
    import logging
    logging.info(f"Dataset generated: {output_file} ({NUM_ROWS} rows)")


if __name__ == "__main__":
    generate_data_and_save()