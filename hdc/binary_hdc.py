from configs.settings import HDC_DIM, DEFAULT_SEED
import numpy as np
from hdc.hdc_common_operations import (
    cosine_similarity, binary_random,
    dot_product, elementwise_product, shifting,
    normalize, binarize
)

class HyperDimensionalComputingBinary:
    """Implements binary hyperdimensional computing operations with vectors in {0,1}.

    This class serves as the binary counterpart to HyperDimensionalComputingBipolar,
    providing operations suitable for binary vectors while delegating to the
    centralized vector operations in hdc_common_operations.py.
    """

    def __init__(self, dim=HDC_DIM, seed=DEFAULT_SEED):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def generate_random_hdv(self, n=1):
        """Generate a random binary hyperdimensional vector."""
        return binary_random(self.dim, self.rng)

    def add_binary_vectors(self, x, y):
        """
        Suma dos vectores binarios {0, 1} y binariza el resultado.

        Suma elemento a elemento y lugo realiza operacion de umbral para también devolver un vector binario.
        """
        # Add the vectors
        result = x + y

        # Binarize the result (convert to binary using a threshold of 0)
        # Any value > 0 becomes 1, and 0 remains 0
        return np.where(result > 0, 1, 0)

    def xor_binary_hv(self, x, y):
        """Apply XOR operation element-wise (primary binding for binary vectors)."""
        return np.logical_xor(x, y).astype(int)


    def dot_product_hv(self, x, y):
        """Calculate dot product between vectors."""
        return dot_product(x, y)

    def elementwise_product_hv(self, x, y):
        """Element-wise multiplication of vectors (logical AND for binary)."""
        return elementwise_product(x, y)

    def shifting_hv(self, x, k=1):
        """Circular shift of vector elements."""
        return shifting(x, k)

    def cosine_similarity(self, x, y):
        """Calculate cosine similarity between vectors."""
        return cosine_similarity(x, y)

    def normalize(self, x):
        """Normalize vector to unit length."""
        return normalize(x)

    def binarize(self, vector, threshold=0.5):
        """Convert vector to binary representation using a threshold."""
        return binarize(vector, threshold)
