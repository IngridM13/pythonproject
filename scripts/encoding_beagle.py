#retomando como opcion porque quiero hacer un fuzzy search...
# misam data -> mismo vector
# lo uso en conjunción con cosine similarity
import numpy as np
import pandas as pd
from tqdm import tqdm

# Hyperdimensional settings
DIMENSIONS = 10_000  # HDV size
BINARY_MODE = False  # Use real-valued HDVs for BEAGLE

# Store learned context vectors
beagle_vectors = {}


def get_feature_hdv(value):
    """Returns a learned HDV for a feature value, updating it with new context."""
    if value not in beagle_vectors:
        beagle_vectors[value] = np.random.uniform(-1, 1, DIMENSIONS)  # Random init

    return beagle_vectors[value]


def encode_row(row):
    """Encodes a single row into an HDV using BEAGLE principles."""
    hdv = np.zeros(DIMENSIONS)  # Initialize HDV

    for col, value in row.items():
        if pd.notna(value):  # Ignore NaN values
            value_hdv = get_feature_hdv(str(value))  # Retrieve context vector
            hdv += value_hdv  # Add to HDV (superposition)

    return hdv / np.linalg.norm(hdv)  # Normalize vector


# Load dataset and encode HDVs
df = pd.read_csv("synthetic_dataset.csv")

# First pass: Learn context vectors
for _, row in tqdm(df.iterrows(), total=len(df
