import numpy as np
# Replace np with torch
import torch

# Vector utility functions that can be used independently
def bipolar_random(d, rng=None):
    """Returns a bipolar vector {-1,+1}^d."""
    if isinstance(rng, np.random.RandomState):
        # Use the seed from numpy's RandomState
        # Note: This affects global torch seed if we use manual_seed. 
        # Better to create a generator if possible, but for now bridging:
        torch.manual_seed(rng.randint(0, 2 ** 32))
        return torch.where(torch.rand(d) > 0.5, 1, -1).to(torch.int8)
    elif isinstance(rng, torch.Generator):
        return torch.where(torch.rand(d, generator=rng) > 0.5, 1, -1).to(torch.int8)

    return torch.where(torch.rand(d) > 0.5, 1, -1).to(torch.int8)

def binary_random(d, rng=None):
    """Returns a binary vector {0,1}^d."""
    if isinstance(rng, np.random.RandomState):
        # Bridge: Create a torch generator seeded from numpy state
        seed = rng.randint(0, 2**32)
        rng = torch.Generator().manual_seed(seed)
    elif rng is None:
        rng = torch.Generator()
    return torch.randint(0, 2, (d,), generator=rng)


def flip_inplace(v, idx):
    """Inverts sign at v[idx]."""
    v[idx] = -v[idx]
    return v


# Utility functions that can be imported by other modules
def dot_product(x, y):
    """Calculate dot product between vectors."""
    return torch.dot(x, y)

def elementwise_product(x, y):
    """Element-wise multiplication of vectors."""
    return x * y

def shifting(x, k=1):
    """Circular shift of vector elements."""
    return torch.roll(x, k)

def normalize(x):
    """Normalize vector to unit length."""
    norm = torch.linalg.norm(x)
    if norm == 0:  # Prevent division by zero
        return x
    return x / norm

def bipolarize(vector):
    """Convert vector to bipolar representation."""
    return torch.where(vector >= 0, 1, -1)

def binarize(vector, threshold=0.5):
    """Convert vector to binary representation using the threshold."""
    return torch.where(vector >= threshold, 1, 0)