from configs.settings import HDC_DIM, DEFAULT_SEED
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Vector utility functions that can be used independently
def bipolar_random(d, rng):
    """Devuelve un vector bipolar {-1,+1}^d."""
    return rng.choice([-1, 1], size=d, replace=True)

def binary_random(d, rng):
    """Devuelve un vector binario {0,1}^d."""
    # Con np.random.default_rng, rng es un Generator y tiene .integers
    return rng.integers(0, 2, size=d, dtype=np.uint8)

def flip_inplace(v, idx):
    """Invierte el signo en v[idx]."""
    v[idx] = -v[idx]
    return v


# Utility functions that can be imported by other modules
def dot_product(x, y):
    """Calculate dot product between vectors."""
    return np.dot(x, y)

def elementwise_product(x, y):
    """Element-wise multiplication of vectors."""
    return x * y

def shifting(x, k=1):
    """Circular shift of vector elements."""
    return np.roll(x, k)

def normalize(x):
    """Normalize vector to unit length."""
    norm = np.linalg.norm(x)
    if norm == 0:  # Previene la división por cero
        return x
    return x / norm

def bipolarize(vector):
    """Convert vector to bipolar representation."""
    return np.where(vector >= 0, 1, -1)

def binarize(vector, threshold=0.5):
    """Convert vector to binary representation using threshold."""
    return np.where(vector >= threshold, 1, 0)
