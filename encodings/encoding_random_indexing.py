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


def generate_sparse_vector():
    vec = np.zeros(DIMENSIONS)
    indices = np.random.choice(DIMENSIONS, size=10, replace=False)  # Small number of active dimensions
    vec[indices] = np.random.choice([-1, 1], size=10)
    return vec


# Store sparse vectors
ri_vectors = defaultdict(generate_sparse_vector)


def encode_row_ri(row):
    hdv = np.zeros(DIMENSIONS)

    for col, value in row.items():
        hdv += ri_vectors[value]  # Aggregate sparse vectors

    return hdv


# Process CSV in chunks
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
    first_chunk = True  # Handle writing headers

    for chunk in tqdm(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE), desc="Encoding dataset", unit="chunk"):
        chunk["HDV"] = chunk.apply(encode_row, axis=1)  # Encode each row into HDV

        chunk.to_csv(f_out, mode="a", header=first_chunk, index=False)
        first_chunk = False  # Write header only once

print(f"Encoding complete. New file: {OUTPUT_CSV}")

