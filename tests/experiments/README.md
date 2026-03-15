# Experiments

This directory contains research experiments for the HDC-based data reconciliation system. All experiments run against a live Milvus instance and are parametrized to run in both `binary` and `float` vector modes.

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

### `conftest.py`

Shared pytest fixtures:

- `with_vector_mode` — parametrizes all tests to run in both `binary` and `float` modes.
- `test_collection` — creates a UUID-named Milvus collection before the test class and drops it on teardown. Set `KEEP_COLLECTION=1` to skip teardown and inspect the collection manually.

---

## Experiment 1: Recall Under Noise

**File**: `test_recall_under_noise.py` | **Command**: `make experiment`

**Question**: Can the system retrieve a specific record when the query is a corrupted version of that same record?

**Method**:
1. Insert N unique persons into Milvus.
2. Optionally insert near-duplicate confusers (`RECALL_NEAR_DUPE_FRACTION`).
3. For each noise level (0.0, 0.1, ..., 1.0): corrupt each stored record with `inject_noise()` and query top-1.
4. Measure `recall@1 = hits / N` per noise level.

**Assertions**: perfect recall at noise=0.0; recall ≥ 0.6 at noise=0.5.

**Environment variables**:

| Variable | Default | Description |
|---|---|---|
| `RECALL_N_PEOPLE` | `1000` | Number of persons to insert |
| `RECALL_NOISE_LEVELS` | `0.0,0.1,...,1.0` | Comma-separated noise levels to evaluate |
| `RECALL_THRESHOLD` | `0.0` | Similarity threshold for `find_closest_match_db` |
| `RECALL_SEED` | `DEFAULT_SEED` | RNG seed for reproducibility |
| `RECALL_NEAR_DUPE_FRACTION` | `0.0` | Fraction of extra confuser records (e.g. `0.2` adds 200 confusers to a 1000-person run) |

---

## Experiment 2: Dedup Recall

**File**: `test_dedup_recall.py` | **Command**: `make experiment-dedup`

**Question**: Given a stored record, does at least one other variant of the same identity appear in its top-K neighbours?

**Method**:
1. Generate N canonical identities.
2. For each identity, produce V noisy variants with `inject_noise()`. Insert all N×V records.
3. For each stored record, re-generate its query vector using the same seed and search top-(K+1).
4. Exclude self, check whether any neighbour belongs to the same identity.
5. Measure `recall@K = hits / (N×V)`.

**Assertion**: recall@K ≥ 0.5.

**Environment variables**:

| Variable | Default | Description |
|---|---|---|
| `DEDUP_N_IDENTITIES` | `200` | Number of canonical identities |
| `DEDUP_VARIANTS_PER_IDENTITY` | `3` | Noisy variants per identity (total records = N×V) |
| `DEDUP_NOISE_FRACTION` | `0.3` | Fraction of fields corrupted per variant |
| `DEDUP_TOP_K` | `5` | K for recall@K |
| `DEDUP_SEED` | `DEFAULT_SEED` | RNG seed for reproducibility |

---

## Experiment 3: Dedup Showcase

**File**: `test_dedup_showcase.py` | **Command**: `make experiment-showcase`

**Question**: What do the results look like qualitatively?

**Method**: Same setup as Dedup Recall (N identities × V variants), but instead of computing an aggregate metric:
1. Sample N_SAMPLES random stored records.
2. For each, retrieve top-K neighbours and print the full record content.
3. Label each result `[MATCH]` (same identity) or `[DIFF]` (different identity).

This is a visual inspection experiment — no assertions. Results are saved to JSON and also printed to stdout during the run.

**Environment variables**:

| Variable | Default | Description |
|---|---|---|
| `SHOWCASE_N_IDENTITIES` | `100` | Number of canonical identities |
| `SHOWCASE_VARIANTS_PER_IDENTITY` | `3` | Noisy variants per identity |
| `SHOWCASE_NOISE_FRACTION` | `0.3` | Fraction of fields corrupted per variant |
| `SHOWCASE_N_SAMPLES` | `5` | Number of records to query |
| `SHOWCASE_TOP_K` | `2` | Top results to retrieve per query |
| `SHOWCASE_SEED` | `DEFAULT_SEED` | RNG seed for reproducibility |

---

## Relationship between experiments

```
Recall Under Noise  →  Can I retrieve the exact record under increasing noise?
Dedup Recall        →  Can I find other variants of the same subject? (metric)
Dedup Showcase      →  How good does that look in practice? (visual inspection)
```

All three experiments save results as JSON to `test_results/`. Use `make results` (or `make results-dedup`) to view the latest output in a human-readable format.
