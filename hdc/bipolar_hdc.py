import numpy as np
from datetime import date, datetime
from configs.settings import HDC_DIM, DEFAULT_SEED
from hdc.hdc_common_operations import (
    bipolar_random, flip_inplace, dot_product,
    elementwise_product, shifting, normalize, bipolarize
)
from typing import Optional, Dict, Any, Iterable
import hashlib


class HyperDimensionalComputingBipolar:
    """Bipolar HDC with vectors in {-1,+1} (dtype=int8)."""

    def __init__(self, dim: int = HDC_DIM, seed: Optional[int] = DEFAULT_SEED):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self._hv_cache: Dict[str, np.ndarray] = {}

    # ---- Generation ----
    def generate_random_hdv(self, n: int = 1) -> np.ndarray:
        """Generate 1 or n random bipolar vectors {-1,+1}, dtype=int8."""
        if n == 1:
            return bipolar_random(self.dim, self.rng).astype(np.int8, copy=False)
        return np.stack([bipolar_random(self.dim, self.rng).astype(np.int8, copy=False) for _ in range(n)], axis=0)

    # ---- Core ops ----
    def bind_hv(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bipolar binding (XOR-equivalent): element-wise product."""
        return elementwise_product(x.astype(np.int8, copy=False), y.astype(np.int8, copy=False)).astype(np.int8,
                                                                                                        copy=False)

    def bundle_init(self) -> np.ndarray:
        """Create an int32 accumulator for bundling."""
        return np.zeros(self.dim, dtype=np.int32)

    def bundle_add(self, acc: np.ndarray, *vectors: Iterable[np.ndarray],
                   weights: Optional[Iterable[int]] = None) -> np.ndarray:
        """Add one or more bipolar vectors into an int32 accumulator, with optional integer weights."""
        if weights is None:
            weights = [1] * len(vectors)
        for v, w in zip(vectors, weights):
            if w != 0:
                acc += int(w) * v.astype(np.int8, copy=False).astype(np.int32, copy=False)
        return acc

    def bundle_finalize(self, acc: np.ndarray, tie_key: Optional[str] = None) -> np.ndarray:
        """Sign of accumulator; break ties deterministically if present."""
        res = np.empty(self.dim, dtype=np.int8)
        res[acc > 0] = 1
        res[acc < 0] = -1
        zeros = (acc == 0)
        if zeros.any():
            res[zeros] = self._tie_breaker_bipolar(tie_key or "tb", self.dim)[zeros]
        return res

    def dot_product_hv(self, x: np.ndarray, y: np.ndarray) -> int:
        return int(dot_product(x, y))

    def cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Optimized Cosine similarity for bipolar {-1,+1} vectors.
        For bipolar vectors, ||v|| = sqrt(dim), so ||x||*||y|| = dim.
        """
        x_nd = np.asarray(x)
        y_nd = np.asarray(y)

        if x_nd.ndim == 1 and y_nd.ndim == 1:
            # Denominator is just self.dim
            return float(dot_product(x_nd, y_nd) / self.dim)

        # Batch usage
        sim = (x_nd @ y_nd.T) / self.dim
        return sim  # ndarray

    def shifting_hv(self, x: np.ndarray, k: int = 1) -> np.ndarray:
        return shifting(x, k)

    # ... (otros métodos como normalize, bipolarize, flip_vector_at) ...

    # ---- Deterministic HVs ----
    def get_bipolar_hv(self, key: Any) -> np.ndarray:
        """Reproducible bipolar HV for a key; uses internal cache."""
        k = str(key)
        if k in self._hv_cache:
            return self._hv_cache[k]
        seed = self._deterministic_hash(k)
        temp_rng = np.random.RandomState(seed)
        hv = bipolar_random(self.dim, temp_rng).astype(np.int8, copy=False)
        self._hv_cache[k] = hv
        return hv

    # ---- Internals ----
    def _deterministic_hash(self, key_str: str) -> int:
        h = hashlib.md5(str(key_str).encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little") % (2 ** 32)

    def _tie_breaker_bipolar(self, key: str, dim: int) -> np.ndarray:
        rng = np.random.RandomState(self._deterministic_hash(f"tb:{key}"))
        return rng.choice([-1, 1], size=dim).astype(np.int8)

    # ---- ENCODING METHODS (MOVED INSIDE CLASS) ----

    def encode_person_bipolar(self, person: Dict[str, Any]) -> np.ndarray:
        """Encode a person's data into a bipolar hypervector (-1/+1 representation)."""
        bundle_acc = self.bundle_init()

        for key in sorted(person.keys()):
            value = person[key]

            # --- START FIX 1: Catch ALL empty values ---
            if value is None:
                continue
            if isinstance(value, str) and not value:  # Catches ""
                continue
            if isinstance(value, list) and not value:  # Catches []
                continue
            # --- END FIX 1 ---

            # Initialize encoded_value here
            encoded_value = np.empty(self.dim, dtype=np.int8)

            # --- INICIO DE LA CORRECCIÓN DE 'attrs' ---
            if key == "attrs" and isinstance(value, dict):
                attrs_acc = self.bundle_init()
                # NOTA: Eliminamos el flag 'has_attrs_data'

                for attr_key in sorted(value.keys()):
                    attr_value_list = value[attr_key]
                    if not attr_value_list:
                        continue  # Skip empty lists like {'akas': []}

                    # Si llegamos aquí, hay datos para esta clave de attr
                    list_acc = self.bundle_init()
                    vectors_to_add = [self.get_bipolar_hv(str(v)) for v in attr_value_list]
                    self.bundle_add(list_acc, *vectors_to_add)
                    encoded_list_hv = self.bundle_finalize(list_acc, tie_key=f"list:{attr_key}")

                    attr_key_hv = self.get_bipolar_hv(attr_key)
                    bound_attr_hv = self.bind_hv(attr_key_hv, encoded_list_hv)
                    self.bundle_add(attrs_acc, bound_attr_hv)

                # Siempre finaliza el bundle 'attrs_acc'.
                # Si no se encontraron datos, 'attrs_acc' estará en su estado inicial (vacío)
                # y bundle_finalize() creará el vector "vacío" correcto.
                # Si se encontraron datos, finalizará el bundle con esos datos.
                # Esto garantiza que la ESTRUCTURA sea siempre la misma.
                print(f"[DEBUG-ENCODE] Finalizando 'attrs_bundle' (Datos presentes: {len(value) > 0}).")
                encoded_value = self.bundle_finalize(attrs_acc, tie_key="attrs_bundle")
            # --- FIN DE LA CORRECCIÓN DE 'attrs' ---

            elif isinstance(value, list):
                # This block is now only for *non-attrs* lists
                list_acc = self.bundle_init()
                vectors_to_add = [self.get_bipolar_hv(str(v)) for v in value]
                self.bundle_add(list_acc, *vectors_to_add)
                encoded_value = self.bundle_finalize(list_acc, tie_key=f"list:{key}")

            elif isinstance(value, (date, datetime)):
                encoded_value = self.encode_date_bipolar(value)

            else:
                # This now only runs for non-empty strings/other values
                encoded_value = self.get_bipolar_hv(str(value))

            # --- Final Bundling ---
            key_hv = self.get_bipolar_hv(key)
            bound_hv = self.bind_hv(key_hv, encoded_value)
            self.bundle_add(bundle_acc, bound_hv)

        return self.bundle_finalize(bundle_acc, tie_key="person_bundle")

    def encode_date_bipolar(self, date_obj: Optional[date]) -> np.ndarray:
        """Special BIPOLAR encoding for date objects."""

        bundle_acc = self.bundle_init()

        if date_obj is None:
            # Devuelve un vector neutro para el binding (multiplicación)
            # para que no afecte a la clave (Key * 1 = Key)
            # NOTA: Esto es si decides NO ignorar Nones en el method principal.
            # Dado que los ignoramos arriba, esta línea es menos crítica,
            # pero es bueno tenerla por si se llama directamente.
            return np.ones(self.dim, dtype=np.int8)

            # FIX: Usa get_bipolar_hv

        year_encoding = self.get_bipolar_hv(f"year_{date_obj.year}")
        month_encoding = self.get_bipolar_hv(f"month_{date_obj.month}")
        day_encoding = self.get_bipolar_hv(f"day_{date_obj.day}")  # <-- FIX

        # FIX: Combina usando bundling (suma), no logical_or
        self.bundle_add(bundle_acc, day_encoding, year_encoding, month_encoding)

        # FIX: Finaliza el bundle y devuelve un vector bipolar int8
        return self.bundle_finalize(bundle_acc, tie_key=f"date:{date_obj}")