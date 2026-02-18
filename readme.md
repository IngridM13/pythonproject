
# HDC-Based Data Reconciliation

This project implements Hyperdimensional Computing (HDC) techniques for data reconciliation tasks, focusing on efficient matching and comparison of data across different sources.

## Project Overview

The HDC-based Data Reconciliation system implements a complete workflow:

1. **Data Generation**: Creates realistic test datasets
2. **Vector Database Storage**: Stores data directly in Milvus
3. **Hypervector Encoding**: Converts row data into hypervectors considering column data types
4. **Vector Indexing**: Uses Milvus for efficient vector indexing and similarity search
5. **Reconciliation**: Performs data matching with configurable distance metrics
6. **Performance Analysis**: Measures speed, accuracy, and coverage of reconciliation

## Key Features

- Binary and Bipolar hypervector encoding strategies
- Data type-specific encoding methods backed by academic research
- Milvus integration for high-performance vector similarity search
- Comprehensive testing framework for reconciliation evaluation

## Technical Approach

### Hypervector Encoding

The project employs both binary and bipolar hypervector representations:

- **Binary vectors**: Values represented as 0s and 1s
- **Bipolar vectors**: Values represented as -1s and 1s

The encoding strategy is chosen based on data type characteristics and supported by research in HDC applications.

### Vector Similarity Search

Milvus is used as the vector database, providing:
- Efficient indexing of high-dimensional vectors
- Configurable distance metrics for similarity measurement
- Scalable performance for large datasets

## Setup

The project utilizes `pyenv` for Python version management. To set up the development environment:

1. **Install the correct Python version:**
    ```bash
    pyenv install
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Launch infrastructure services:**
    The project uses Docker to manage the Milvus service:
    ```bash
    docker-compose -f infra/docker-compose.yml up -d
    ```

## Project Structure

- `hdc/`: Core HDC implementation with binary and bipolar encoding strategies
- `database_utils/`: Database connections for Milvus
- `distance/`: Distance measurement implementations
- `encoding_methods/`: Data type specific encoding techniques
- `dummy_data/`: Test data generation
- `tests/`: Test suite
- `configs/`: Configuration files
- `scripts/`: Utility scripts
- `infra/`: Infrastructure setup (Docker configurations)

## Testing

To execute tests, use:

bash docker-compose -f infra/docker-compose.test.yml up --build


Alternatively, run individual test files directly through your IDE.

## Future Work

- Optimization of encoding strategies for specific data types
- Performance benchmarks across different vector dimensions
- Additional distance metrics evaluation