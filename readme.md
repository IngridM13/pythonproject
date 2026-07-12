
# HDC-Based Data Reconciliation

This project implements Hyperdimensional Computing (HDC) techniques for data reconciliation tasks — encoding person records as hypervectors, storing them in Milvus, and performing similarity search for fault-tolerant record matching.

## Project Overview

The pipeline works as follows:

1. **Data Generation**: Creates realistic synthetic person records (name, lastname, DOB, phone, gender, addresses, etc.)
2. **Hypervector Encoding**: Converts records into high-dimensional binary or bipolar vectors using data type-specific strategies
3. **Storage**: Stores hypervectors in Milvus alongside scalar fields
4. **Similarity Search**: Queries Milvus to find the closest matching record using Hamming (binary) or inner-product (float) distance
5. **Reconciliation**: Matches records across datasets, tolerating noise and partial corruption

## Key Features

- Binary (`{0,1}`) and bipolar (`{-1,1}`) hypervector encoding strategies
- Data type-aware encoding: dates, names, categorical strings, phone numbers, lists, attribute dicts
- Deterministic hypervectors — same input always produces the same vector (SHA-256 seeded RNG)
- Milvus integration with support for both `BINARY_VECTOR` (HAMMING) and `FLOAT_VECTOR` (IP) index modes
- Recall-under-noise experiment to measure fault tolerance quantitatively

## Field Weighting

During encoding, each field's contribution to the final hypervector can be scaled by an integer weight. A weight of `N` means the field's bound hypervector is counted `N` times in the bundling step — equivalent to multiplying its influence by `N` in the final majority vote.

The default weight configuration is `NAME_AND_DATE_WEIGHTS` (defined in `configs/settings.py`):

| Field | Default weight |
|---|---|
| `name` | 2 |
| `lastname` | 2 |
| `dob` | 2 |
| all other fields | 1 (implicit) |

This configuration is applied by default to all encoding and search operations — `encode_person()`, `store_person()`, `find_closest_match_db()`, and `search_for_eval()`. It was selected because it improves deduplication recall@5: binary 92.3% → 98.7%, float 95.7% → 99.3%.

To use equal weights or a custom configuration, pass `field_weights=None` or a custom `Dict[str, int]` to any of those functions. Fields absent from the dict receive weight 1.

Experiment 3 (`test_exp03_field_weighting.py`) ablates eleven weighting variants to characterize the sensitivity of recall to different configurations.

## Setup

Requires `pyenv` for Python version management.

```bash
pyenv install                               # installs the version pinned in .python-version
python3.11 -m venv .venv311                 # create the project virtual environment
.venv311/bin/pip install -r requirements.txt
```

Start Milvus (required for most tests and experiments):

```bash
docker-compose -f infra/docker-compose.yml up -d
```

Key environment variables (configure via `.env`):

| Variable | Default | Description |
|---|---|---|
| `MILVUS_URI` | `http://localhost:19530` | Milvus connection URL |
| `MILVUS_VECTOR_MODE` | `binary` | `binary` or `float` — controls vector type and index |
| `HDC_NPROBE` | `8` | IVF search cells to probe: `8` = approximate (mode A), `128` = exhaustive (mode B) |
| `SKIP_MILVUS_TESTS` | — | Set to `True` to skip tests requiring a live Milvus instance |

## Project Structure

```
hdc/                  Core HDC encoding (binary and bipolar)
encoding_methods/     Data type-specific encoding strategies + Milvus search
database_utils/       Milvus connection and collection management
utils/                Person data normalization
dummy_data/           Synthetic data generation
configs/              Settings (HDC_DIM, DEFAULT_SEED, etc.)
tests/
  unit/               Encoding and normalization (no Milvus required)
  integration/        Milvus insert/query operations
  bench/              Performance benchmarks
  functional/         End-to-end reconciliation tests
  experiments/        Research experiments (see below)
infra/                Docker configurations for dev and test environments
test_results/         JSON output from bench and experiment runs
scripts/              Utility scripts (e.g. show_results.py)
```

## Common Commands

A `Makefile` is provided for convenience:

| Command | Description |
|---|---|
| `make up` | Start Milvus |
| `make down` | Stop Milvus |
| `make test` | Run unit + integration + functional tests |
| `make test-unit` | Unit tests only |
| `make test-integration` | Integration tests only |
| `make test-bench` | Benchmarks only |
| `make test-functional` | Functional tests only |
| `make experiments-ann` | Run all 13 experiments with nprobe=8 (approximate, mode A) |
| `make experiments-exhaustive` | Run all 13 experiments with nprobe=128 (exhaustive, mode B) |
| `make experiment01-recall-under-noise-ann` | Run Experiment 1 (approximate, nprobe=8) |
| `make experiment01-recall-under-noise-exhaustive` | Run Experiment 1 (exhaustive, nprobe=128) |
| `make experiment06-per-field-noise` | Run Experiment 6 (no mode suffix — inherits `HDC_NPROBE`) |
| `make experiment13-separability` | Run Experiment 13 (separability analysis, always exhaustive) |
| `make results01-recall-under-noise` | Show latest results for Experiment 1 |

See `tests/experiments/README.md` for the full list of experiment commands and configuration options.

## Running Tests

```bash
# All tests via Docker (includes Milvus)
docker-compose -f infra/docker-compose.test.yml up --build

# By category
pytest tests/unit/
pytest tests/integration/
pytest tests/bench/
pytest tests/functional/

# Specific file or test
pytest tests/unit/test_encoding_methods.py
pytest tests/unit/test_encoding_methods.py::TestClassName::test_method_name
```

## Experiments

The system includes 13 numbered research experiments covering recall under noise, deduplication, field sensitivity, scalability, and separability analysis. All experiments run against a live Milvus instance and support both `binary` and `float` vector modes.

### Running experiments

```bash
# Run all experiments — approximate search (production-realistic)
make experiments-ann

# Run all experiments — exhaustive search (reference quality)
make experiments-exhaustive

# Run a single experiment
make experiment01-recall-under-noise-ann
HDC_NPROBE=128 make experiment13-separability

# View results for a specific experiment
make results06-per-field-noise
```

`HDC_NPROBE` can be set per-command without modifying any file. Default is `8`. See `tests/experiments/README.md` for the full list of experiments, configuration variables, and output locations.

---

### Recall Under Noise

Measures how well the system finds the correct record when the query is a corrupted version of a stored record.

**Setup**: Generates N synthetic person records, encodes and stores them in Milvus, then for each noise level corrupts each record (swapping letters, shifting dates, changing categories, etc.) and checks whether the top-1 search result is the original.

**Metric**: `recall@1 = hits / N` per noise level.

```bash
make experiment01-recall-under-noise-ann
# or: pytest tests/experiments/test_exp01_recall_under_noise.py -v -s
```

Configuration via environment variables:

| Variable | Default | Description |
|---|---|---|
| `RECALL_N_PEOPLE` | `1000` | Number of persons to insert |
| `RECALL_NOISE_LEVELS` | `0.0,0.1,...,1.0` | Comma-separated noise levels to evaluate |
| `RECALL_THRESHOLD` | `0.0` | Similarity threshold for `find_closest_match_db` |
| `RECALL_SEED` | `DEFAULT_SEED` | RNG seed for reproducibility |
| `RECALL_NEAR_DUPE_FRACTION` | `0.0` | Fraction of extra confuser records to insert as near-duplicates (e.g. `0.2` adds 200 confusers to a 1000-person run) |
| `KEEP_COLLECTION` | — | Set to `1` to skip teardown and keep the Milvus collection alive for inspection |

#### Inspecting the collection after an experiment run

By default the test collection is dropped after the experiment. To keep it alive:

```bash
KEEP_COLLECTION=1 pytest tests/experiments/test_exp01_recall_under_noise.py -v -s
```

The fixture will print the collection name at the end of the run, e.g.:

```
[FIXTURE] KEEP_COLLECTION set — skipping teardown for 'people_test_a3f8c1b2'.
[FIXTURE] Collection has 1000 entities. Inspect it, then drop manually.
```

Query it with a Python shell:

```python
from pymilvus import Collection, connections
connections.connect(uri="http://localhost:19530")

col = Collection("people_test_a3f8c1b2")  # use the name printed above
col.load()
rows = col.query(expr="id >= 0", output_fields=["id", "name", "lastname", "dob"], limit=10)
for r in rows:
    print(r)
```

When done, drop the collection manually:

```python
Collection("people_test_a3f8c1b2").drop()
```

Results are saved as JSON to `test_results/recall_under_noise_<mode>_<timestamp>.json`.

### Deduplication Recall

Measures how well the system surfaces same-person candidates when multiple
noisy variants of each identity are stored alongside each other — simulating
records for the same person arriving from different data sources.

**Setup**: Generates N canonical synthetic identities. For each identity,
produces V noisy variants using `inject_noise()`. All N×V records are inserted
into Milvus with distinct IDs. For each stored record, the experiment queries
its top-(K+1) neighbours, excludes self, and checks whether any of the top-K
results belongs to the same identity.

**Metric**: `recall@K = hits / (N×V)` — a result is a hit if at least one
neighbour in the top-K comes from the same canonical identity.

```bash
make experiment02-dedup-recall-ann
# or: pytest tests/experiments/test_exp02_dedup_recall.py -v -s
```

Configuration via environment variables:

| Variable | Default | Description |
|---|---|---|
| `DEDUP_N_IDENTITIES` | `1000` | Number of canonical identities to generate |
| `DEDUP_VARIANTS_PER_IDENTITY` | `3` | Noisy variants per identity (total records = N×V) |
| `DEDUP_NOISE_FRACTION` | `0.3` | Fraction of fields corrupted per variant |
| `DEDUP_TOP_K` | `3` | K for recall@K |
| `DEDUP_SEED` | `DEFAULT_SEED` | RNG seed for reproducibility |
| `KEEP_COLLECTION` | — | Set to `1` to keep the Milvus collection alive after the run |

Results are saved as JSON to `test_results/dedup_recall_<mode>_<timestamp>.json`.

```bash
make results02-dedup-recall

# or for a specific file:
python scripts/show_results.py test_results/dedup_recall_binary_<timestamp>.json
```
