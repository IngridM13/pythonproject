
import torch

# Vector utility functions that can be used independently
def bipolar_random(d, rng=None):
    """Returns a bipolar vector {-1,+1}^d."""
    if isinstance(rng, torch.Generator):
        return torch.where(torch.rand(d, generator=rng) > 0.5, 1, -1).to(torch.int8)
    else:
        return torch.where(torch.rand(d) > 0.5, 1, -1).to(torch.int8)

def binary_random(d, rng=None):
    """Returns a binary vector {0,1}^d."""
    if isinstance(rng, torch.Generator):
        return torch.randint(0, 2, (d,), generator=rng, dtype=torch.int8)
    else:
        return torch.randint(0, 2, (d,), dtype=torch.int8)

def flip_inplace(v, idx):
    """Inverts sign at v[idx]."""
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.int8)
    v[idx] = -v[idx]
    return v

# Utility functions that can be imported by other modules
def dot_product(x, y):
    """Calculate dot product between vectors."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float)
    return torch.dot(x.float(), y.float())

def elementwise_product(x, y):
    """Element-wise multiplication of vectors."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    return x * y

def shifting(x, k=1):
    """Circular shift of vector elements."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return torch.roll(x, k)

def normalize(x):
    """Normalize vector to unit length."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)
    norm = torch.linalg.norm(x.float())
    if norm == 0:  # Prevent division by zero
        return x
    return x / norm

def bipolarize(vector):
    """Convert vector to bipolar representation."""
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector)
    return torch.where(vector >= 0, 1, -1).to(torch.int8)

def binarize(vector, threshold=0.5):
    """Convert vector to binary representation using the threshold."""
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector)
    return torch.where(vector >= threshold, 1, 0).to(torch.int8)

def hamming_distance(x, y):
    """Calculate Hamming distance between binary vectors."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    return torch.sum(x != y).item()