# context.py
from typing import Dict, Optional, Tuple
import numpy as np
from configs.settings import HDC_DIM
from ops_bipolar import HyperDimensionalComputingBipolar

class HyperDimensionalContext:
    def __init__(self, vectors_dimension: int = HDC_DIM,
                 ops: Optional[HyperDimensionalComputingBipolar] = None):
        self._vectors: Dict[str, np.ndarray] = {}
        self.vectors_dimension = vectors_dimension
        # Inject ops (better for testing); fall back to a default if none provided
        self.ops = ops or HyperDimensionalComputingBipolar(dim=vectors_dimension)

    def add_vector(self, key: str, vector: np.ndarray) -> None:
        if vector.shape[-1] != self.vectors_dimension:
            raise ValueError(
                f"Vector dim {vector.shape[-1]} != context dim {self.vectors_dimension}"
            )
        self._vectors[key] = np.array(vector, copy=True)

    def get_all_vectors(self) -> Dict[str, np.ndarray]:
        return dict(self._vectors)

    def get_all_vectors_as_list(self):
        return [np.array(v, copy=True) for v in self._vectors.values()]

    # Obtener un vector específico por su clave
    def get_vector(self, key: str) -> Optional[np.ndarray]:
        v = self._vectors.get(key)
        return None if v is None else np.array(v, copy=True)

    # hace una copia, no estoy segura de que querramos una copia (siempre)
    def get_nearest_vector(self, vector: np.ndarray) -> Optional[Tuple[str, np.ndarray, float]]:
        if not self._vectors:
            return None
        best_key, best_sim = None, -1.0
        for k, v in self._vectors.items():
            sim = self.ops.cosine_similarity(vector, v)
            if sim > best_sim:
                best_key, best_sim = k, sim
        return best_key, np.array(self._vectors[best_key], copy=True), best_sim
