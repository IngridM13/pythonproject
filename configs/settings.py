from typing import Dict

# Constants
NUM_ROWS = 5_000_000
CHUNK_SIZE = 500_000  # Process in chunks to optimize memory
OUTPUT_FILE = "test_synthetic_dataset.csv"
DEFAULT_SEED=42

# Dimensión del hipervector
HDC_DIM = 10000

# nprobe controls ANN search approximation (IVF index).
# NPROBE_ANN=8  → mode A: true approximate search (production-realistic, fast)
# NPROBE_EXHAUSTIVE=128 → mode B: exhaustive (nprobe=nlist, reference quality)
# Runtime value is read from env var HDC_NPROBE; these are the canonical defaults.
NPROBE_ANN = 8
NPROBE_EXHAUSTIVE = 128

# Default field weights — name_and_date configuration
# Improves dedup recall@5: binary 92.3% → 98.7%, float 95.7% → 99.3%
NAME_AND_DATE_WEIGHTS: Dict[str, int] = {
    "name": 2,
    "lastname": 2,
    "dob": 2,
}

# Experiment 2 — Dedup Recall
DEDUP_N_IDENTITIES = 1000
DEDUP_VARIANTS_PER_IDENTITY = 3
DEDUP_NOISE_FRACTION = 0.30
DEDUP_TOP_K = 3
DEDUP_N_SAMPLES = 5
DEDUP_SEED = 42

# Experiment 3 — Field Weighting Ablation
FIELD_WEIGHTING_N = 1000
FIELD_WEIGHTING_V = 3
FIELD_WEIGHTING_NOISE = 0.30
FIELD_WEIGHTING_TOP_K = 3
FIELD_WEIGHTING_SEED = 42

# Experiment 4 — Scalability
SCALABILITY_N_VALUES = [100, 500, 1000, 5000, 10000]
SCALABILITY_V = 3
SCALABILITY_NOISE = 0.30
SCALABILITY_TOP_K = 3
SCALABILITY_SEED = 42

# Experiment 5 — Ranking Metrics
RANKING_N = 5000
RANKING_V = 3
RANKING_NOISE = 0.30
RANKING_TOP_K = 3
RANKING_SEED = 42

# Experiment 6 — Per-Field Noise
PER_FIELD_N = 1000
PER_FIELD_V = 3
PER_FIELD_NOISE = 0.30
PER_FIELD_TOP_K = 3
PER_FIELD_SEED = 42

# Experiment 7 — Per-Field Noise Sweep
PER_FIELD_SWEEP_FIELDS = ["name", "lastname", "dob"]
PER_FIELD_SWEEP_NOISE_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# Raised from 200 to 1000 (27/07/2026): at N=200 the sweep showed a flat
# ~0.995-1.000 recall curve across the whole 0-90% range, matching the exact
# same low-competition saturation issue already documented for Experiment 6
# ("con N=200 el recall baseline se aproxima al 100%..."). N=1000 matches the
# scale Experiment 6 already uses for this same reason.
PER_FIELD_SWEEP_N = 1000
PER_FIELD_SWEEP_V = 3
PER_FIELD_SWEEP_TOP_K = 3
PER_FIELD_SWEEP_SEED = 42

# Experiment 8 — Dimensionality Sweep
DIM_SWEEP_VALUES = [1000, 2000, 5000, 10000]
DIM_SWEEP_N = 1000
DIM_SWEEP_V = 3
DIM_SWEEP_NOISE = 0.30
DIM_SWEEP_TOP_K = 3
DIM_SWEEP_SEED = 42

# Experiment 9 — Date Encoding Comparison
DATE_ENC_N = 1000
DATE_ENC_V = 3
DATE_ENC_NOISE = 0.30
DATE_ENC_TOP_K = 3
DATE_ENC_SEED = 42

# Scale experiments (10, 12, 13) share a common N grid for direct comparability.
# 14a extends beyond it (overlaps at 50_000 / 100_000) for extreme-scale capacity testing.
SCALE_N_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000]

# Experiment 10 — Scalability with Noisy Duplicates
EXP10_COLLECTION_SIZES = SCALE_N_VALUES
EXP10_NOISE_RATIO = 0.20
EXP10_NOISE_LEVEL = 0.30
EXP10_TOP_K = 3
EXP10_SEED = 42
EXP10_DUPLICATES_PER_ORIGINAL = 3

# Experiment 11 — 2D sweep: collection size (N) x top_k -> Recall@k
EXP11_N_VALUES = [200, 1_000, 5_000]
EXP11_K_VALUES = [2, 3, 5]
EXP11_NOISE_FRACTION = 0.30
EXP11_VARIANTS = 3
EXP11_SEED = 42

# Experiment 12 — Recall@1 under noise across collection sizes
# Noise levels 20% and 30% straddle the critical degradation threshold identified
# by Experiment 13's separability analysis; both are kept to see whether the ANN
# gap itself widens across that transition, not just recall in isolation.
EXP12_N_VALUES = SCALE_N_VALUES
EXP12_M_QUERIES = 200
EXP12_NOISE_LEVELS = [0.20, 0.30]
EXP12_SEED = 42

# Experiment 13 — Separability Analysis
EXP13_N_VALUES = SCALE_N_VALUES
EXP13_M_QUERIES = 200
EXP13_NOISE_LEVELS = [0.20, 0.30]
EXP13_SEED = 42
