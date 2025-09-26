import numpy as np

def bipolar_random(d, rng):
    """Devuelve un vector bipolar {-1,+1}^d."""
    return rng.choice([-1, 1], size=d, replace=True)

def binary_random(d, rng):
    """Devuelve un vector binario {0,1}^d."""
    return rng.integers(0, 2, size=d, dtype=np.uint8)

def flip_inplace(v, idx):
    """Invierte el signo en v[idx]."""
    v[idx] = -v[idx]
    return v

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den != 0 else 0.0
