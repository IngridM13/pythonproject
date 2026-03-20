# Experiments

This directory contains research experiments for the HDC-based data reconciliation system. All experiments run against a live Milvus instance and are parametrized to run in both `binary` and `float` vector modes.

## Prerequisites

```bash
make up                     # Start Milvus (etcd + minio + standalone)
pip install -r requirements.txt
```

Results are saved as JSON (and CSV where noted) to `test_results/`. Use `make results-<name>` to view the latest output.

---

## Shared infrastructure

### `noise_injection.py`

Central module used by all experiments. Provides `inject_noise(person, noise_fraction, rng)`, which corrupts `floor(noise_fraction × 10)` fields of a normalized person dict using realistic strategies:

| Field | Corruption strategy |
|---|---|
| `name`, `lastname` | Transposition, deletion, insertion, substitution, accent stripping |
| `dob` | Day/month swap, year offset (±1–5 yr), day shift (±1–30 d) |
| `mobile_number` | 1–3 digit errors (65%) or fully new number (35%) |
| `attrs.address` | Remove one entry or modify a street number |
| `attrs.akas`, `attrs.landlines` | Remove one entry or modify one (never fully cleared) |
| `marital_status`, `gender`, `race` | Replaced with a different valid category |

### `experiment_utils.py`

Shared helpers used across multiple experiments:

- `generate_canonical_persons(n)` — generates N normalized synthetic person dicts
- `insert_noisy_variants(persons, V, noise, seed, col)` — inserts V noisy variants per identity
- `run_dedup_recall(...)` — evaluates Recall@K, MRR, Hit@1 over all stored variants
- `save_report(prefix, mode, report)` — serializes results to `test_results/`

### `conftest.py`

Shared pytest fixtures:

- `with_vector_mode` — parametrizes tests to run in both `binary` and `float` modes
- `test_collection` — creates a UUID-named Milvus collection before the test and drops it on teardown. Set `KEEP_COLLECTION=1` to skip teardown for manual inspection.

---

## Experiment 1 — Recall Under Noise

**File**: `test_exp01_recall_under_noise.py` | **Run**: `make experiment`

**Question**: Can the system retrieve a specific stored record when the query is a corrupted version of it?

**Method**: Insert N clean persons. For each noise level, corrupt each person with `inject_noise()` and query top-1. Check if the result is the original record.

**Metric**: `Recall@1 = hits / N` per noise level.

**Assertions**: Recall@1 = 1.0 at noise=0.0; Recall@1 ≥ 0.5 at noise=0.5.

| Variable | Default | Description |
|---|---|---|
| `RECALL_N_PEOPLE` | `1000` | Number of persons to insert |
| `RECALL_NOISE_LEVELS` | `0.0,0.1,...,1.0` | Comma-separated noise levels |
| `RECALL_THRESHOLD` | `0.0` | Similarity threshold for search |
| `RECALL_SEED` | `42` | RNG seed |
| `RECALL_NEAR_DUPE_FRACTION` | `0.0` | Fraction of extra confuser records (e.g. `0.2` adds 200 confusers to a 1000-person run) |

---

## Experiment 2 — Dedup Recall

**File**: `test_exp02_dedup_recall.py` | **Run**: `make experiment-dedup`

**Question**: Given a stored noisy record, does at least one other variant of the same identity appear in its top-K neighbours?

**Method**: Generate N canonical identities × V noisy variants each. Insert all N×V records. For each record, re-generate its query and search top-(K+1), exclude self, check for a same-identity neighbour.

**Metric**: `Recall@K = hits / (N×V)`. Also computes MRR and Hit@1.

**Assertion**: Recall@K ≥ 0.5.

| Variable | Default | Description |
|---|---|---|
| `DEDUP_N_IDENTITIES` | `1000` | Number of canonical identities |
| `DEDUP_VARIANTS_PER_IDENTITY` | `3` | Noisy variants per identity |
| `DEDUP_NOISE_FRACTION` | `0.3` | Fraction of fields corrupted |
| `DEDUP_TOP_K` | `3` | K for Recall@K |
| `DEDUP_SEED` | `42` | RNG seed |

---

## Experiment 3 — Field Weighting Ablation

**File**: `test_exp03_field_weighting.py` | **Run**: `make experiment-weights`

**Question**: How much does field weighting (upweighting name + dob) improve dedup recall compared to uniform weights?

**Method**: Run dedup recall evaluation under multiple weight configurations: `uniform`, `name_heavy`, `name_and_date`, `date_heavy`, `name_only`, `date_only`.

**Metric**: Recall@K, MRR, Hit@1 per configuration; delta vs uniform baseline.

| Variable | Default | Description |
|---|---|---|
| `FIELD_WEIGHT_N` | `200` | Number of canonical identities |
| `FIELD_WEIGHT_V` | `3` | Noisy variants per identity |
| `FIELD_WEIGHT_NOISE` | `0.3` | Noise fraction |
| `FIELD_WEIGHT_K` | `5` | K for Recall@K |
| `FIELD_WEIGHT_SEED` | `42` | RNG seed |

---

## Experiment 4 — Scalability

**File**: `test_exp04_scalability.py` | **Run**: `make experiment-scalability`

**Question**: How do insertion time, query time, and dedup recall scale with collection size?

**Method**: For each N in N_VALUES: generate N identities × V variants, insert all, evaluate dedup recall, record insertion and query times.

**Metrics**: Recall@K, insertion time (s), query time (s) per N.

| Variable | Default | Description |
|---|---|---|
| `SCALABILITY_N_VALUES` | `100,500,1000,5000,10000` | Comma-separated collection sizes |
| `SCALABILITY_V` | `3` | Noisy variants per identity |
| `SCALABILITY_NOISE` | `0.30` | Noise fraction |
| `SCALABILITY_K` | `5` | K for Recall@K |
| `SCALABILITY_SEED` | `42` | RNG seed |

---

## Experiment 5 — Ranking Metrics

**File**: `test_exp05_ranking_metrics.py` | **Run**: `make experiment-ranking`

**Question**: Beyond Recall@K, how good is the ranking quality? Does the correct match appear near the top?

**Method**: Same setup as Dedup Recall, but computes MRR and Hit@1 in addition to Recall@K.

**Metrics**: Recall@K, MRR, Hit@1.

| Variable | Default | Description |
|---|---|---|
| `RANKING_N` | `200` | Number of canonical identities |
| `RANKING_V` | `3` | Noisy variants per identity |
| `RANKING_NOISE` | `0.30` | Noise fraction |
| `RANKING_K` | `5` | K for Recall@K |
| `RANKING_SEED` | `42` | RNG seed |

---

## Experiment 6 — Per-Field Noise Sensitivity

**File**: `test_exp06_per_field_noise.py` | **Run**: `make experiment-per-field-noise`

**Question**: Which individual fields contribute most to retrieval quality?

**Method**: Insert N×V noisy variants once (baseline). Then for each field: corrupt only that field in a clean query, measure Recall@K, MRR, Hit@1. Compute delta vs clean baseline to rank fields by their informational weight.

**Metric**: Δ Recall@K per field vs clean baseline (positive = degradation).

| Variable | Default | Description |
|---|---|---|
| `PER_FIELD_N` | `200` | Number of canonical identities |
| `PER_FIELD_V` | `3` | Noisy variants per identity |
| `PER_FIELD_NOISE` | `0.30` | Noise fraction for variant generation |
| `PER_FIELD_K` | `5` | K for Recall@K |
| `PER_FIELD_SEED` | `42` | RNG seed |

---

## Experiment 7 — Per-Field Noise Sweep

**File**: `test_exp07_per_field_noise_sweep.py` | **Run**: `make experiment-per-field-sweep`

**Question**: For each field individually, how does recall degrade as noise on that field increases from 0% to 90%?

**Method**: For each (field, noise_level) pair, corrupt only that field at the given level and measure recall. All other fields remain clean.

**Metric**: Recall@K per (field, noise_level).

| Variable | Default | Description |
|---|---|---|
| `PER_FIELD_SWEEP_N` | `200` | Number of canonical identities |
| `PER_FIELD_SWEEP_V` | `3` | Noisy variants per identity |
| `PER_FIELD_SWEEP_K` | `5` | K for Recall@K |
| `PER_FIELD_SWEEP_SEED` | `42` | RNG seed |

---

## Experiment 8 — Dimensionality Ablation

**File**: `test_exp08_dimensionality.py` | **Run**: `make experiment-dimensionality`

**Question**: At what number of HDC dimensions does recall saturate? Is 10,000 necessary?

**Method**: For each dim in `[1000, 2000, 5000, 10000]`: rebuild the HDC encoder at that dimension, insert N×V records, evaluate recall.

**Metric**: Recall@K, MRR, Hit@1 per dimension.

| Variable | Default | Description |
|---|---|---|
| `DIM_SWEEP_VALUES` | `1000,2000,5000,10000` | Comma-separated dimensions to test |
| `DIM_SWEEP_N` | `200` | Number of canonical identities |
| `DIM_SWEEP_V` | `3` | Noisy variants per identity |
| `DIM_SWEEP_NOISE` | `0.30` | Noise fraction |
| `DIM_SWEEP_K` | `5` | K for Recall@K |
| `DIM_SWEEP_SEED` | `42` | RNG seed |

---

## Experiment 9 — Date Encoding Comparison

**File**: `test_exp09_date_encoding.py` | **Run**: `make experiment-date-encoding`

**Question**: Does a circular/FPE date encoder outperform the current thermometer encoder?

**Method**: Run dedup recall under both date encoding strategies and compare Recall@K, MRR, Hit@1.

| Variable | Default | Description |
|---|---|---|
| `DATE_ENC_N` | `200` | Number of canonical identities |
| `DATE_ENC_V` | `3` | Noisy variants per identity |
| `DATE_ENC_NOISE` | `0.30` | Noise fraction |
| `DATE_ENC_K` | `5` | K for Recall@K |
| `DATE_ENC_SEED` | `42` | RNG seed |

---

## Experiment 10 — Scalability with Noisy Duplicates

**File**: `test_exp10_scalability_noisy_dupes.py` | **Run**: `make experiment-exp10`

**Question**: How does recall scale when the database contains both clean originals and noisy duplicates mixed together — a realistic production scenario?

**Method**:
1. Insert N clean canonical records.
2. Select `n_sources = int(N × noise_ratio) // duplicates_per_original` originals without replacement.
3. For each selected original, generate `duplicates_per_original` independent noisy variants and insert them.
4. Total noisy records ≈ `int(N × noise_ratio)`. Total in collection ≈ `N × (1 + noise_ratio)`.
5. For each noisy record: query top-(K+1), exclude self, check if the original it came from appears in the top-K results.

**Metrics**: Recall@1, Recall@K, Recall@D (D = `duplicates_per_original`), MRR, Hit@1 per N.

**Output**: CSV + JSON saved to `test_results/exp10_scalability_noisy_dupes/`.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `EXP10_COLLECTION_SIZES` | `10000,50000,100000` | Comma-separated N values to test |
| `EXP10_NOISE_RATIO` | `0.20` | Fraction of noisy duplicates relative to N |
| `EXP10_NOISE_LEVEL` | `0.30` | Corruption level passed to `inject_noise` |
| `EXP10_TOP_K` | `5` | K for Recall@K |
| `EXP10_DUPLICATES_PER_ORIGINAL` | `3` | Noisy variants generated per source record |
| `EXP10_MODES` | `binary,float` | Comma-separated modes to run |
| `EXP10_SEED` | `42` | RNG seed |

### Common usage examples

```bash
# Full run (both modes, default sizes)
make experiment-exp10

# Run only float mode (e.g. to resume after binary already completed)
EXP10_MODES=float make experiment-exp10

# Run only binary mode
EXP10_MODES=binary make experiment-exp10

# Quick smoke test with small sizes
EXP10_COLLECTION_SIZES=100,500 make experiment-exp10

# Higher noise to stress-test recall
EXP10_NOISE_LEVEL=0.5 make experiment-exp10

# More duplicates per original (denser cluster in the DB)
EXP10_DUPLICATES_PER_ORIGINAL=5 make experiment-exp10

# Resume float only at full scale after a crash
EXP10_MODES=float EXP10_COLLECTION_SIZES=10000,50000,100000 make experiment-exp10
```

### Memory requirements

Float vectors at 10,000 dims use ~40 KB per record. At N=100,000 the collection holds ~120,000 records ≈ **~4.8 GB of vectors** plus Milvus index overhead. Docker Desktop must be configured with sufficient memory:

- `Docker Desktop → Settings → Resources → Memory` → **at least 8 GB** (14 GB recommended for N=100,000 in float mode)
- Binary mode uses 32× less memory (1,250 bytes per vector vs 40,000) and runs comfortably at lower memory limits.

---

## NK Sweep (unnumbered)

**File**: `test_recall_nk_sweep.py` | **Run**: `make experiment-nk-sweep`

**Question**: How does recall change as both collection size N and search depth K vary simultaneously?

**Method**: 2D sweep over N × K. For each (N, K) pair: insert N×V variants, retrieve neighbour lists up to max(K), slice to each K and compute recall. Prints a pivot table per mode.

**Note**: Parameters are hardcoded in the file (`N_VALUES=[200,1000,5000]`, `K_VALUES=[2,3,5]`, `NOISE=0.3`, `V=3`). No environment variable configuration.

---

## Results summary

| Experiment | Make target | Output location |
|---|---|---|
| 1 — Recall Under Noise | `make results` | `test_results/recall_under_noise_*.json` |
| 2 — Dedup Recall | `make results-dedup` | `test_results/dedup_recall_*.json` |
| 3 — Field Weighting | `make results-weights` | `test_results/field_weighting_*.json` |
| 4 — Scalability | `make results-scalability` | `test_results/scalability_*.json` |
| 5 — Ranking Metrics | `make results-ranking` | `test_results/ranking_metrics_*.json` |
| 6 — Per-Field Noise | `make results-per-field-noise` | `test_results/per_field_noise_*.json` |
| 7 — Per-Field Sweep | `make results-per-field-sweep` | `test_results/per_field_sweep_*.json` |
| 8 — Dimensionality | `make results-dimensionality` | `test_results/dimensionality_*.json` |
| 9 — Date Encoding | `make results-date-encoding` | `test_results/date_encoding_*.json` |
| 10 — Noisy Dupes | `make results-exp10` | `test_results/exp10_scalability_noisy_dupes/` |
| NK Sweep | `make results-nk-sweep` | `test_results/recall_nk_sweep_*.json` |
