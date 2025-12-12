from typing import Optional
import numpy as np
from configs.settings import HDC_DIM, DEFAULT_SEED

class HyperDimensionalComputingBipolar:
    def __init__(self, dim: int = HDC_DIM, seed: Optional[int] = DEFAULT_SEED):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def generate_random_hdv(self, n=1):
        hdv = np.random.randint(0, 2, size=(1, self.dim), dtype=np.int8)
        # Reemplazamos 0s por -1s para que sea bipolar
        hdv[hdv == 0] = -1
        return hdv[0]

    def add_hv(self, x, y):
        return np.clip(x + y, -1, 1)

    '''Mas usado para binarios que bipolares. 
        Para que retorne el mismo tipo que las otras 
        operaciones tengo que castear explícitamente y remapear el resultado de lo contrario 
        me va a devolver un array de booleanos.
        Parece que lo mas recomendable en este caso es usar el producto?'''
    def xor_hv(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.logical_xor(x > 0, y > 0).astype(np.int8) * 2 - 1  # map back to ±1

    def dot_product_hv(self, x: np.ndarray, y: np.ndarray) -> int:
        return int(np.dot(x, y))

    # Binding en espacio bipolar: multiplicaion elemento a elemento
    def elementwise_product_hv(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x * y).astype(np.int8)

    # Circle Shifting
    def shifting_hv(self, x: np.ndarray, k: int = 1) -> np.ndarray:
        return np.roll(x, k)

    def cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        if nx == 0 or ny == 0:
            return 0.0
        return float(np.dot(x, y) / (nx * ny))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x)
        return x if norm == 0 else x / norm

    def bipolarize(self, vector: np.ndarray) -> np.ndarray:
        return np.where(vector >= 0, 1, -1).astype(np.int8) # guarda en int8 para ahorrar memoria

    def flip_inplace(v, idx):
        """Invierte el signo en v[idx]."""
        v[idx] = -v[idx]
        return v

# TODO: Bernie: agregate otro no para bipolares, pero para binarios.