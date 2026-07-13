# Experiments

Research experiments for the HDC-based data reconciliation system. All experiments run against a live Milvus instance and are parametrized for both `binary` and `float` vector modes.

## Prerequisites

```bash
make up                     # Start Milvus (etcd + minio + standalone)
pip install -r requirements.txt
```

Results are saved as JSON to `test_results/`. Use `make results-<name>` to view the latest output.

---

## Choosing the search mode (nprobe)

All experiments respect the `HDC_NPROBE` environment variable, which controls how many IVF cells are probed during ANN search:

| Value | Mode | Description |
|---|---|---|
| `8` (default) | Mode A — approximate | Production-realistic; true ANN search |
| `128` | Mode B — exhaustive | `nprobe=nlist`; functionally equivalent to brute-force for small N |

```bash
# Run a single experiment in mode A (default, nprobe=8)
make experiment01-recall-under-noise-ann

# Run a single experiment in mode B (exhaustive, nprobe=128)
make experiment01-recall-under-noise-exhaustive

# Run ALL experiments in mode A
make experiments-ann

# Run ALL experiments in mode B
make experiments-exhaustive
```

---

## Shared infrastructure

### `noise_injection.py`

Central module used by all experiments. Provides `inject_noise(person, noise_fraction, rng)`, which corrupts `floor(noise_fraction × 10)` fields using realistic strategies:

| Field | Corruption strategy |
|---|---|
| `name`, `lastname` | Transposition, deletion, insertion, substitution, accent stripping |
| `dob` | Day/month swap, year offset (±1–5 yr), day shift (±1–30 d) |
| `mobile_number` | 1–3 digit errors (65%) or fully new number (35%) |
| `attrs.address` | Remove one entry or modify a street number |
| `attrs.akas`, `attrs.landlines` | Remove one entry or modify one (never fully cleared) |
| `marital_status`, `gender`, `race` | Replaced with a different valid category |

### `experiment_utils.py`

Shared helpers:

- `generate_canonical_persons(n)` — generates N normalized synthetic person dicts
- `insert_noisy_variants(persons, V, noise, seed, col)` — inserts V noisy variants per identity
- `run_dedup_recall(...)` — evaluates Recall@K, MRR, Hit@1 over all stored variants
- `save_report(prefix, mode, report)` — serializes results to `test_results/`

### `conftest.py`

Shared pytest fixtures:

- `with_vector_mode` — parametrizes tests to run in both `binary` and `float` modes
- `test_collection` — creates a UUID-named Milvus collection before the test and drops it on teardown. Set `KEEP_COLLECTION=1` to skip teardown for manual inspection.

---

## Evaluation scenarios

Experiments use one of two collection setups:

- **Scenario A** (Experiments 1, 3, 6, 7, 8, 9, 12, 13): collection contains only canonical records; the noisy query is generated on the fly at query time. Models production: clean reference DB, noisy incoming record.
- **Scenario B** (Experiments 2, 4, 5, 10): collection contains N×V noisy variants per identity. Models integration of two degraded data sources.

---

## Experiment 1 — Recall Under Noise

**File**: `test_exp01_recall_under_noise.py` | **Run**: `make experiment01-recall-under-noise-ann` / `make experiment01-recall-under-noise-exhaustive`

**Question**: Can the system retrieve a specific stored record when the query is a corrupted version of it?

**Method**: Insert N clean persons (Scenario A). For each noise level, corrupt each person with `inject_noise()` and query top-1. Check if the result is the original record.

**Metric**: `Recall@1 = hits / N` per noise level.

| Variable | Default | Description |
|---|---|---|
| `RECALL_N_PEOPLE` | `1000` | Number of persons to insert |
| `RECALL_NOISE_LEVELS` | `0.0,0.1,...,1.0` | Comma-separated noise levels |
| `RECALL_THRESHOLD` | `0.0` | Similarity threshold for search |
| `RECALL_SEED` | `42` | RNG seed |
| `RECALL_NEAR_DUPE_FRACTION` | `0.0` | Fraction of extra confuser records |

---

## Experiment 2 — Dedup Recall

**File**: `test_exp02_dedup_recall.py` | **Run**: `make experiment02-dedup-recall-ann` / `make experiment02-dedup-recall-exhaustive`

**Question**: Given a stored noisy record, does at least one other variant of the same identity appear in its top-K neighbours?

**Method**: Generate N canonical identities × V noisy variants each (Scenario B). Insert all N×V records. For each record, query top-(K+1), exclude self, check for a same-identity neighbour.

**Metric**: `Recall@K = hits / (N×V)`. Also computes MRR and Hit@1.

| Variable | Default | Description |
|---|---|---|
| `DEDUP_N_IDENTITIES` | `1000` | Number of canonical identities |
| `DEDUP_VARIANTS_PER_IDENTITY` | `3` | Noisy variants per identity |
| `DEDUP_NOISE_FRACTION` | `0.3` | Fraction of fields corrupted |
| `DEDUP_TOP_K` | `3` | K for Recall@K |
| `DEDUP_SEED` | `42` | RNG seed |

---

## Experiment 3 — Field Weighting Ablation

**File**: `test_exp03_field_weighting.py` | **Run**: `make experiment03-weights`

**Question**: How much does field weighting (upweighting name + dob) improve recall compared to uniform weights?

**Method**: Run dedup recall under multiple weight configurations: `uniform`, `name_heavy`, `name_and_date`, `date_heavy`, `name_only`, `date_only`.

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

**File**: `test_exp04_scalability.py` | **Run**: `make experiment04-scalability-ann` / `make experiment04-scalability-exhaustive`

**Question**: How do insertion time, query time, and dedup recall scale with collection size?

**Method**: For each N in N_VALUES: generate N identities × V variants (Scenario B), insert all, evaluate dedup recall, record times.

**Metrics**: Recall@K, insertion time (s), query time (s) per N.

| Variable | Default | Description |
|---|---|---|
| `SCALABILITY_N_VALUES` | `100,500,1000,5000,10000` | Comma-separated collection sizes |
| `SCALABILITY_V` | `3` | Noisy variants per identity |
| `SCALABILITY_NOISE` | `0.30` | Noise fraction |
| `SCALABILITY_TOP_K` | `3` | K for Recall@K |
| `SCALABILITY_SEED` | `42` | RNG seed |

---

## Experiment 5 — Ranking Metrics

**File**: `test_exp05_ranking_metrics.py` | **Run**: `make experiment05-ranking-ann` / `make experiment05-ranking-exhaustive`

**Question**: Beyond Recall@K, how good is the ranking quality? Does the correct match appear near the top, and how pure is the result set?

**Method**: Generate N=5,000 canonical identities × V=3 noisy variants (Scenario B). For each stored record, query top-(K+1), exclude self, compute ranking metrics.

**Metrics**: Recall@K, MRR, Hit@1, Precision@K.

| Variable | Default | Description |
|---|---|---|
| `RANKING_N_IDENTITIES` | `5000` | Number of canonical identities |
| `RANKING_VARIANTS_PER_IDENTITY` | `3` | Noisy variants per identity |
| `RANKING_NOISE_FRACTION` | `0.30` | Noise fraction |
| `RANKING_TOP_K` | `5` | K for ranking metrics |
| `RANKING_SEED` | `42` | RNG seed |

---

## Experiment 6 — Per-Field Noise Sensitivity

**File**: `test_exp06_per_field_noise.py` | **Run**: `make experiment06-per-field-noise`

**Question**: Which individual fields contribute most to retrieval quality?

**Method**: Insert N=1,000 canonical identities × V=3 noisy variants (Scenario B). Establish a clean-query baseline. Then for each field: corrupt only that field in the canonical query, measure Recall@K, MRR, Hit@1. Compute delta vs baseline to rank fields by informational weight.

**Metric**: Δ Recall@K per field vs clean baseline (positive = degradation).

| Variable | Default | Description |
|---|---|---|
| `PER_FIELD_N` | `1000` | Number of canonical identities |
| `PER_FIELD_V` | `3` | Noisy variants per identity |
| `PER_FIELD_NOISE` | `0.30` | Noise fraction for variant generation |
| `PER_FIELD_TOP_K` | `3` | K for Recall@K |
| `PER_FIELD_SEED` | `42` | RNG seed |

---

## Experiment 7 — Per-Field Noise Sweep

**File**: `test_exp07_per_field_noise_sweep.py` | **Run**: `make experiment07-per-field-sweep-ann` / `make experiment07-per-field-sweep-exhaustive`

**Question**: For each key field individually, how does recall degrade as noise on that field increases from 0% to 90%?

**Method**: For each (field, noise_level) pair: corrupt only that field at the given level, measure Recall@K, MRR, Hit@1. All other fields remain clean. Fields swept are determined by Experiment 6 results.

**Metric**: Recall@K per (field, noise_level).

| Variable | Default | Description |
|---|---|---|
| `PER_FIELD_SWEEP_FIELDS` | `name,lastname,dob` | Comma-separated fields to sweep |
| `PER_FIELD_SWEEP_NOISE_LEVELS` | `0,10,20,...,90` | Noise levels (integers 0–100) |
| `PER_FIELD_SWEEP_N` | `200` | Number of canonical identities |
| `PER_FIELD_SWEEP_V` | `3` | Noisy variants per identity |
| `PER_FIELD_SWEEP_TOP_K` | `3` | K for Recall@K |
| `PER_FIELD_SWEEP_SEED` | `42` | RNG seed |

---

## Experiment 8 — Dimensionality Ablation

**File**: `test_exp08_dimensionality.py` | **Run**: `make experiment08-dimensionality-ann` / `make experiment08-dimensionality-exhaustive`

**Question**: At what number of HDC dimensions does recall saturate? Is D=10,000 necessary?

**Method**: For each dim in `[1000, 2000, 5000, 10000]`: rebuild the HDC encoder at that dimension, insert N×V records, evaluate recall.

**Metric**: Recall@K, MRR, Hit@1 per dimension.

| Variable | Default | Description |
|---|---|---|
| `DIM_SWEEP_VALUES` | `1000,2000,5000,10000` | Comma-separated dimensions |
| `DIM_SWEEP_N` | `1000` | Number of canonical identities |
| `DIM_SWEEP_V` | `3` | Noisy variants per identity |
| `DIM_SWEEP_NOISE` | `0.30` | Noise fraction |
| `DIM_SWEEP_TOP_K` | `3` | K for Recall@K |
| `DIM_SWEEP_SEED` | `42` | RNG seed |

---

## Experiment 9 — Date Encoding Comparison

**File**: `test_exp09_date_encoding.py` | **Run**: `make experiment09-date-encoding`

**Question**: Does a circular/FPE date encoder outperform the current thermometer encoder?

**Method**: Run dedup recall under both date encoding strategies and compare metrics.

| Variable | Default | Description |
|---|---|---|
| `DATE_ENC_N` | `1000` | Number of canonical identities |
| `DATE_ENC_V` | `3` | Noisy variants per identity |
| `DATE_ENC_NOISE` | `0.30` | Noise fraction |
| `DATE_ENC_TOP_K` | `3` | K for Recall@K |
| `DATE_ENC_SEED` | `42` | RNG seed |

---

## Experiment 10 — Scalability with Noisy Duplicates

**File**: `test_exp10_scalability_noisy_dupes.py` | **Run**: `make experiment10-scalability-noisy-dupes-ann` / `make experiment10-scalability-noisy-dupes-exhaustive`

**Question**: How does recall scale when the database contains both clean originals and noisy duplicates mixed together (Scenario B at large scale)?

**Method**:
1. Insert N clean canonical records.
2. Select `n_sources = int(N × noise_ratio) // duplicates_per_original` originals.
3. For each selected original, generate `duplicates_per_original` noisy variants and insert them.
4. For each noisy record: query top-(K+1), exclude self, check if the original appears in top-K.

**Metrics**: Recall@1, Recall@K, MRR, Hit@1 per N.

| Variable | Default | Description |
|---|---|---|
| `EXP10_COLLECTION_SIZES` | `10000,50000,100000` | Comma-separated N values |
| `EXP10_NOISE_RATIO` | `0.20` | Fraction of noisy duplicates relative to N |
| `EXP10_NOISE_LEVEL` | `0.30` | Corruption level |
| `EXP10_TOP_K` | `5` | K for Recall@K |
| `EXP10_DUPLICATES_PER_ORIGINAL` | `3` | Noisy variants per source record |
| `EXP10_SEED` | `42` | RNG seed |

**Memory note**: float mode at N=100,000 requires ~8 GB Docker memory (14 GB recommended). Binary mode uses 32× less memory.

---

## Experiment 11 — NK Sweep

**File**: `test_exp11_recall_nk_sweep.py` | **Run**: `make experiment11-nk-sweep-ann` / `make experiment11-nk-sweep-exhaustive`

**Question**: How does recall change as both collection size N and search depth K vary simultaneously?

**Method**: 2D sweep over N × K. For each (N, K) pair: insert N×V variants, retrieve neighbours up to max(K), compute recall. Prints a pivot table per mode.

**Note**: Parameters are hardcoded in the file (`N_VALUES=[200,1000,5000]`, `K_VALUES=[2,3,5]`, `NOISE=0.3`, `V=3`).

---

## Experiment 12 — Recall@1 vs Collection Size (Scenario A)

**File**: `test_exp12_recall_n_sweep.py` | **Run**: `make experiment12-recall-n-sweep-ann` / `make experiment12-recall-n-sweep-exhaustive`

**Question**: How does recall degrade as collection size grows, under the production scenario (clean DB, noisy query)?

**Method**: For each N in N_VALUES (Scenario A): insert N canonical records, generate M noisy queries on the fly, measure Recall@1, Recall@K, MRR, Hit@1. This is the primary comparative experiment between binary and float modes.

**Metric**: Recall@1, Recall@K, MRR, Hit@1 per N and mode.

| Variable | Default | Description |
|---|---|---|
| `EXP12_N_VALUES` | `1000,5000,10000,50000` | Comma-separated collection sizes |
| `EXP12_M_QUERIES` | `200` | Number of queries per N |
| `EXP12_NOISE_LEVEL` | `0.20` | Noise fraction |
| `EXP12_SEED` | `42` | RNG seed |

---

## Experiment 13 — Separability Analysis

**File**: `test_exp13_separability_analysis.py` | **Run**: `make experiment13-separability`

**Question**: How well separated are correct matches from incorrect ones in the vector space? What fraction of queries result in a collision (a wrong record ranking above the correct one)?

**Method**: For each (mode, N) pair (Scenario A): insert N canonical records, generate M noisy queries, capture `sim_pos` (similarity to ground truth) and `sim_neg` (best false positive similarity). Run with `nprobe=nlist` (exhaustive) to isolate encoder quality from ANN approximation effects.

**Metrics**: `gap = sim_pos - sim_neg` (mean, std, p25, p75), `pct_collision`, `recall@1`.

**Note**: Run directly with Python (not pytest): `python tests/experiments/test_exp13_separability_analysis.py`

| Variable | Default | Description |
|---|---|---|
| `EXP13_N_VALUES` | `1000,5000,10000,50000` | Comma-separated collection sizes |
| `EXP13_M_QUERIES` | `200` | Queries per (mode, N) |
| `EXP13_NOISE` | `0.20` | Noise fraction |
| `EXP13_SEED` | `42` | RNG seed |
| `EXP13_MODES` | `binary,float` | Comma-separated modes |

---

## Results summary

| Experiment | Run target (mode A / mode B) | Results target | Output location |
|---|---|---|---|
| 1 — Recall Under Noise | `make experiment01-recall-under-noise-ann` / `-exhaustive` | `make results01-recall-under-noise` | `test_results/recall_under_noise_*.json` |
| 2 — Dedup Recall | `make experiment02-dedup-recall-ann` / `-exhaustive` | `make results02-dedup-recall` | `test_results/dedup_recall_*.json` |
| 3 — Field Weighting | `make experiment03-weights` | `make results03-weights` | `test_results/field_weighting_*.json` |
| 4 — Scalability | `make experiment04-scalability-ann` / `-exhaustive` | `make results04-scalability` | `test_results/scalability_*.json` |
| 5 — Ranking Metrics | `make experiment05-ranking-ann` / `-exhaustive` | `make results05-ranking` | `test_results/ranking_metrics_*.json` |
| 6 — Per-Field Noise | `make experiment06-per-field-noise` | `make results06-per-field-noise` | `test_results/per_field_noise_*.json` |
| 7 — Per-Field Sweep | `make experiment07-per-field-sweep-ann` / `-exhaustive` | `make results07-per-field-sweep` | `test_results/per_field_sweep_*.json` |
| 8 — Dimensionality | `make experiment08-dimensionality-ann` / `-exhaustive` | `make results08-dimensionality` | `test_results/dimensionality_*.json` |
| 9 — Date Encoding | `make experiment09-date-encoding` | `make results09-date-encoding` | `test_results/date_encoding_*.json` |
| 10 — Noisy Dupes | `make experiment10-scalability-noisy-dupes-ann` / `-exhaustive` | `make results10-scalability-noisy-dupes` | `test_results/exp10_scalability_noisy_dupes/` |
| 11 — NK Sweep | `make experiment11-nk-sweep-ann` / `-exhaustive` | `make results11-nk-sweep` | `test_results/recall_nk_sweep_*.json` |
| 12 — Recall vs N | `make experiment12-recall-n-sweep-ann` / `-exhaustive` | `make results12-recall-n-sweep` | `test_results/exp12_recall_n_sweep_*.json` |
| 13 — Separability | `make experiment13-separability` | `make results13-separability` | `test_results_128/exp13_separability_exhaustive_*.json` |
