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

# Experiment 5 — Ranking Metrics
RANKING_N = 200
RANKING_V = 3
RANKING_NOISE = 0.30
RANKING_K = 5
RANKING_SEED = 42

# Experiment 6 — Per-Field Noise
PER_FIELD_N = 200
PER_FIELD_V = 3
PER_FIELD_NOISE = 0.30
PER_FIELD_K = 5
PER_FIELD_SEED = 42

# Experiment 7 — Per-Field Noise Sweep
PER_FIELD_SWEEP_FIELDS = ["name", "lastname", "dob"]
PER_FIELD_SWEEP_NOISE_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
PER_FIELD_SWEEP_N = 200
PER_FIELD_SWEEP_V = 3
PER_FIELD_SWEEP_K = 5
PER_FIELD_SWEEP_SEED = 42

# Experiment 8 — Dimensionality Sweep
DIM_SWEEP_VALUES = [1000, 2000, 5000, 10000]
DIM_SWEEP_N = 200
DIM_SWEEP_V = 3
DIM_SWEEP_NOISE = 0.30
DIM_SWEEP_K = 5
DIM_SWEEP_SEED = 42
