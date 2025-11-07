import numpy as np
from datetime import date, datetime
from configs.settings import HDC_DIM, DEFAULT_SEED
from hdc.hdc_common_operations import (
    cosine_similarity, binary_random,
    dot_product, elementwise_product, shifting,
    normalize, binarize
)
from typing import Optional, Dict, Any, Iterable
import hashlib


class HyperDimensionalComputingBinary:
    """Implementa operaciones binarias HDC con vectores en {0,1} (dtype=uint8)."""

    def __init__(self, dim: int = HDC_DIM, seed: Optional[int] = DEFAULT_SEED):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        # FIX: Inicializar el caché interno
        self._hv_cache: Dict[str, np.ndarray] = {}

    def generate_random_hdv(self, n: int = 1) -> np.ndarray:
        """Genera 1 o n vectores binarios aleatorios {0,1}, dtype=uint8."""
        if n == 1:
            return binary_random(self.dim, self.rng)
        return np.stack([binary_random(self.dim, self.rng) for _ in range(n)], axis=0)

    # ---- Core ops ----

    def bind_hv(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Binding binario (XOR)."""
        return np.logical_xor(x, y).astype(np.uint8, copy=False)

    def bundle_init(self) -> np.ndarray:
        """Crea un acumulador int32 para el bundling."""
        return np.zeros(self.dim, dtype=np.int32)

    def bundle_add(self, acc: np.ndarray, *vectors: Iterable[np.ndarray]) -> np.ndarray:
        """Suma uno o más vectores binarios en un acumulador int32."""
        for v in vectors:
            acc += v.astype(np.int32, copy=False)
        return acc

    def bundle_finalize(self, acc: np.ndarray, num_components: int) -> np.ndarray:
        """
        Binariza el acumulador usando el voto mayoritario.
        El umbral es la mitad del número de vectores sumados.
        """
        if num_components == 0:
            return np.zeros(self.dim, dtype=np.uint8)  # Devuelve el vector nulo

        threshold = num_components / 2
        return (acc > threshold).astype(np.uint8, copy=False)

    def hamming_distance(self, x: np.ndarray, y: np.ndarray) -> int:
        """Calcula la Distancia de Hamming (número de bits diferentes)."""
        return int(np.sum(np.logical_xor(x, y)))

    def hamming_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calcula la Similitud de Hamming (1.0 = idénticos, 0.0 = opuestos)."""
        dist = np.sum(np.logical_xor(x, y))
        return (self.dim - dist) / self.dim

    def shifting_hv(self, x: np.ndarray, k: int = 1) -> np.ndarray:
        return shifting(x, k)

    # ... (otros métodos como cosine_similarity, dot_product, etc.) ...

    # ---- Deterministic HVs ----

    def get_binary_hv(self, key: Any) -> np.ndarray:
        """
        Obtiene un hipervector binario reproducible para una clave.
        Usa el caché interno de la clase.
        """
        key_str = str(key)

        # FIX: Usar siempre el caché interno
        if key_str in self._hv_cache:
            return self._hv_cache[key_str]

        seed = self._deterministic_hash(key_str)
        temp_rng = np.random.RandomState(seed)
        hv = binary_random(self.dim, temp_rng)

        # FIX: Guardar en el caché interno
        self._hv_cache[key_str] = hv
        return hv

    def _deterministic_hash(self, key_str: str) -> int:
        """Genera un hash determinístico para una clave string."""
        key_bytes = str(key_str).encode("utf-8")
        h = hashlib.md5(key_bytes).digest()
        return int.from_bytes(h[:8], "little") % (2 ** 32)

    # ---- ENCODING METHODS (MOVED INSIDE CLASS) ----

    def encode_person_binary(self, person: Dict[str, Any]) -> np.ndarray:
        """Codifica los datos de una persona en un hipervector binario (0/1)."""

        # FIX: Usar los métodos de bundling robustos de la clase
        bundle_acc = self.bundle_init()
        num_components = 0

        for key in sorted(person.keys()):
            value = person[key]

            # El elemento neutro (cero) se maneja correctamente al no añadir nada
            if value is None or (isinstance(value, list) and not value):
                continue

            encoded_value: Optional[np.ndarray] = None

            if isinstance(value, list):
                list_acc = self.bundle_init()
                vectors_to_add = [self.get_binary_hv(str(v)) for v in value]
                self.bundle_add(list_acc, *vectors_to_add)
                # FIX: Usar voto mayoritario para la lista
                encoded_value = self.bundle_finalize(list_acc, num_components=len(vectors_to_add))

            elif isinstance(value, (date, datetime)):
                encoded_value = self.encode_date_binary(value)
            else:
                # FIX: Usar el 'get_binary_hv' de la clase
                encoded_value = self.get_binary_hv(str(value))

            # Binding (Binario = XOR)
            key_hv = self.get_binary_hv(key)
            bound_hv = self.bind_hv(key_hv, encoded_value)

            # Añadir al bundle final
            self.bundle_add(bundle_acc, bound_hv)
            num_components += 1

        # Finalizar el vector de la persona
        return self.bundle_finalize(bundle_acc, num_components=num_components)

    def encode_date_binary(self, date_obj: Optional[date]) -> np.ndarray:
        """Codificación binaria especial para fechas."""

        if not date_obj:
            # FIX: Usar self.dim
            return np.zeros(self.dim, dtype=np.uint8)

        # FIX: Usar los métodos de bundling de la clase
        bundle_acc = self.bundle_init()

        # FIX: Usar 'self.get_binary_hv'
        base_encoding = self.get_binary_hv(str(date_obj))
        year_encoding = self.get_binary_hv(f"year_{date_obj.year}")
        month_encoding = self.get_binary_hv(f"month_{date_obj.month}")

        # FIX: Combinar usando bundling de voto mayoritario
        self.bundle_add(bundle_acc, base_encoding, year_encoding, month_encoding)

        # Hay 3 componentes en este bundle de fecha
        return self.bundle_finalize(bundle_acc, num_components=3)