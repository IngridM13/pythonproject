from typing import Dict

# Constants
#NUM_ROWS = 5_000_000
NUM_ROWS = 5
#CHUNK_SIZE = 500_000  # Process in chunks to optimize memory
CHUNK_SIZE = 5
OUTPUT_FILE = "test_synthetic_dataset.csv"
DEFAULT_SEED=42

# Defininicion de la hiperdimensionalidad hypervector
HDC_DIM = 10000

# Default field weights — name_and_date configuration
# Improves dedup recall@5: binary 92.3% → 98.7%, float 95.7% → 99.3%
NAME_AND_DATE_WEIGHTS: Dict[str, int] = {
    "name": 2,
    "lastname": 2,
    "dob": 2,
}

# Experiment 4 — Scalability
SCALABILITY_N_VALUES = [100, 500, 1000, 5000, 10000]
SCALABILITY_V = 3
SCALABILITY_NOISE = 0.30
SCALABILITY_K = 5
SCALABILITY_SEED = 42
