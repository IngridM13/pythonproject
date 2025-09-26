#sería el mejor approach

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine

# Hyperdimensional settings
DIMENSIONS = 10_000  # HDV size
BINARY_MODE = True  # Set to False for real-valued HRR

# Generate random feature vectors
feature_vectors = {}

def get_feature_hdv(value):
    """Assigns a unique HDV for each unique feature value."""
    if value not in feature_vectors:
        feature_vectors[value] = np.random.choice([-1, 1] if not BINARY_MODE else [0, 1], DIMENSIONS)
    return feature_vectors[value]

def encode_row(row):
    """Encodes a single row into a high-dimensional vector (HDV)."""
    hdv = np.zeros(DIMENSIONS) if not BINARY_MODE else np.random.choice([0, 1], DIMENSIONS)  # Initialize HDV

    for col, value in row.items():
        value_hdv = get_feature_hdv(str(value))  # Convert feature value to HDV
        hdv = np.bitwise_xor(hdv, value_hdv) if BINARY_MODE else hdv + value_hdv  # XOR for binary, SUM for real

    return hdv

# Load dataset and encode HDVs
df = pd.read_csv("synthetic_dataset.csv")
df["HDV"] = df.apply(encode_row, axis=1)

# Save encoded dataset
df.to_csv("hdv_encoded_dataset_hrr_bsc.csv", index=False)
print("Encoding complete!")
