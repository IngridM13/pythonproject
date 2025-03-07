import pandas as pd
import numpy as np
import hdlib  # Ensure you have a library for Hyperdimensional Computing
from tqdm import tqdm

# Load CSV in chunks
INPUT_CSV = "synthetic_dataset.csv"
OUTPUT_CSV = "hdv_encoded_dataset_rbb.csv"
CHUNK_SIZE = 500_000  # Process in chunks to optimize memory

# Hyperdimensional settings
DIMENSIONS = 10_000  # HDV size
BINARY_MODE = True  # Set to False for real-valued HDVs


# Function to encode a row into HDV
def encode_row(row):
    # Create a random HDV for each column's value
    hdv = np.zeros(DIMENSIONS) if not BINARY_MODE else np.random.choice([0, 1], DIMENSIONS)

    # Bind values (superposition)
    for col in row.index:
        value_hdv = np.random.choice([-1, 1], DIMENSIONS)  # Random bipolar vector
        hdv += value_hdv if not BINARY_MODE else np.bitwise_xor(hdv, value_hdv)

    return hdv


# Process CSV in chunks
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
    first_chunk = True  # Handle writing headers

    for chunk in tqdm(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE), desc="Encoding dataset", unit="chunk"):
        chunk["HDV"] = chunk.apply(encode_row, axis=1)  # Encode each row into HDV

        chunk.to_csv(f_out, mode="a", header=first_chunk, index=False)
        first_chunk = False  # Write header only once

print(f"Encoding complete. New file: {OUTPUT_CSV}")

