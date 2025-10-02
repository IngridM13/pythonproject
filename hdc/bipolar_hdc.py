from configs.settings import HDC_DIM, DEFAULT_SEED
import numpy as np
from hdc.hdc_common_operations import (
    bipolar_random, flip_inplace, dot_product,
    elementwise_product, shifting, normalize, bipolarize
)
from sklearn.metrics.pairwise import cosine_similarity

class HyperDimensionalComputingBipolar:
    """Implements bipolar hyperdimensional computing operations with vectors in {-1,+1}."""

    def __init__(self, dim=HDC_DIM, seed=DEFAULT_SEED):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def generate_random_hdv(self, n=1):
        """Generate a random bipolar hyperdimensional vector."""
        if n == 1:
            return bipolar_random(self.dim, self.rng)
        else:
            return np.array([bipolar_random(self.dim, self.rng) for _ in range(n)])

    def add_bipolar_hv_with_clipping(self, x, y):
        """Suma dos vectores bipolares ocn clipping."""
        return np.clip(x + y, -1, 1)

    def xor_hv(self, x, y):
        """Apply XOR operation element-wise."""
        return np.logical_xor(x, y)

    def dot_product_hv(self, x, y):
        """Calculate dot product between vectors."""
        return dot_product(x, y)

    def elementwise_product_hv(self, x, y):
        """Element-wise multiplication of vectors."""
        return elementwise_product(x, y)

    def shifting_hv(self, x, k=1):
        """Circular shift of vector elements."""
        return shifting(x, k)

    def cosine_similarity(self, x, y):
        """Calculate cosine similarity between vectors with safety checks."""
        return cosine_similarity(x, y)

    def normalize(self, x):
        """Normalize vector to unit length."""
        return normalize(x)

    def bipolarize(self, vector):
        """Convert vector to bipolar representation."""
        return bipolarize(vector)

    def flip_vector_at(self, v, idx):
        """Invierte el signo en posiciones específicas del vector."""
        return flip_inplace(v.copy(), idx)
