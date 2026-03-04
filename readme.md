
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

## Setup

Requires `pyenv` for Python version management.

```bash
pyenv install
pip install -r requirements.txt
```

Start Milvus (required for most tests and experiments):

```bash
docker-compose -f infra/docker-compose.yml up -d
```

Key environment variables (configure via `.env`):

| Variable | Default | Description |
|---|---|---|
| `MILVUS_URI` | `http://localhost:19530` | Milvus connection URL |
| `MILVUS_VECTOR_MODE` | `float` | `binary` or `float` |
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
| `make test` | Run all test categories |
| `make test-unit` | Unit tests only |
| `make test-integration` | Integration tests only |
| `make test-bench` | Benchmarks only |
| `make test-functional` | Functional tests only |
| `make experiment` | Run the recall-under-noise experiment |
| `make results` | Show latest experiment results |
| `make results-float` | Show latest float mode results |
| `make results-binary` | Show latest binary mode results |

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

### Recall Under Noise

Measures how well the system finds the correct record when the query is a corrupted version of a stored record.

**Setup**: Generates N synthetic person records, encodes and stores them in Milvus, then for each noise level corrupts each record (swapping letters, shifting dates, changing categories, etc.) and checks whether the top-1 search result is the original.

**Metric**: `recall@1 = hits / N` per noise level.

```bash
make experiment
# or: pytest tests/experiments/test_recall_under_noise.py -v -s
```

Configuration via environment variables:

| Variable | Default | Description |
|---|---|---|
| `RECALL_N_PEOPLE` | `1000` | Number of persons to insert |
| `RECALL_NOISE_LEVELS` | `0.0,0.1,...,1.0` | Comma-separated noise levels to evaluate |
| `RECALL_THRESHOLD` | `0.0` | Similarity threshold for `find_closest_match_db` |
| `RECALL_SEED` | `DEFAULT_SEED` | RNG seed for reproducibility |

Results are saved as JSON to `test_results/recall_under_noise_<mode>_<timestamp>.json`.

To view results in a human-readable format:

```bash
make results          # latest result file (any mode)
make results-float    # latest float mode result
make results-binary   # latest binary mode result

# or for a specific file:
python scripts/show_results.py test_results/recall_under_noise_float_<timestamp>.json
```
