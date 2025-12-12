import numpy as np
from datetime import date
from configs.settings import HDC_DIM, DEFAULT_SEED
from hdc.hdc_common_operations import (
    bipolar_random, dot_product,
    elementwise_product, shifting
)
from typing import Optional, Dict, Any, Iterable
import hashlib
from hdc.datatype_profiler import DataTypeProfiler
from utils.person_data_normalization import normalize_person_data
from hdc.hdc_encoding_strategy import (
    DefaultEncodingStrategy,
    DateEncodingStrategy,
    ListEncodingStrategy,
    AttrsEncodingStrategy,
    HDCEncodingStrategyFactory
)


class HyperDimensionalComputingBipolar:
    """Bipolar HDC with vectors in {-1,+1} (dtype=int8)."""

    def __init__(self, dim: int = HDC_DIM, seed: Optional[int] = DEFAULT_SEED):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self._hv_cache: Dict[str, np.ndarray] = {}
        self._date_thresholds: Optional[np.ndarray] = None
        self._max_range_days: int = 365 * 200

        # Inicializar el factory de estrategias (nuevo)
        self.strategy_factory = HDCEncodingStrategyFactory(self)
        self.register_default_strategies()

    # Nuevo método para registrar estrategias predeterminadas
    def register_default_strategies(self):
        """Registra las estrategias de codificación predeterminadas."""
        factory = self.strategy_factory
        factory.register_strategy("DATE", DateEncodingStrategy)
        factory.register_strategy("ATTRS_DICT", AttrsEncodingStrategy)
        factory.register_strategy("LIST_OF_STR", ListEncodingStrategy)

        # Otras estrategias
        factory.register_strategy("CATEGORICAL_STR", DefaultEncodingStrategy)
        factory.register_strategy("TEXT_NAME", DefaultEncodingStrategy)
        factory.register_strategy("PHONE_STR", DefaultEncodingStrategy)

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

    # ------------------------------------------------------------------
    # 1) Inicialización de thresholds para fechas
    # ------------------------------------------------------------------
    def _init_date_thresholds(self):
        """Inicializa thresholds aleatorios para el encoding escalar de fechas."""
        if self._date_thresholds is not None:
            return

        # Semilla fija para que siempre dé los mismos HV (muy importante para tu tesis)
        rng = np.random.default_rng(12345)

        # thresholds uniformes en [0, max_range_days]
        self._date_thresholds = rng.integers(
            low=0,
            high=self._max_range_days + 1,
            size=self.dim,
            dtype=np.int32,
        )

    # ------------------------------------------------------------------
    # 2) Nuevo encode_date_bipolar ESCALAR
    # ------------------------------------------------------------------
    def encode_date_bipolar(self, date_obj: Optional[date]) -> np.ndarray:
        """Encoding bipolar escalar de fechas (sin periodicidad artificial)."""
        if date_obj is None:
            # Vec. neutro para binding multiplicativo
            return np.ones(self.dim, dtype=np.int8)

        self._init_date_thresholds()

        reference_date = date(1900, 1, 1)
        days_since_reference = (date_obj - reference_date).days

        # Acotar al rango soportado (por seguridad)
        t = np.int32(
            max(0, min(days_since_reference, self._max_range_days))
        )

        thresholds = self._date_thresholds  # shape (dim,)

        # Regla: hv_j = +1 si t >= θ_j, -1 en otro caso
        hv = np.where(t >= thresholds, 1, -1).astype(np.int8)
        return hv

    # Método legacy: redirecciona al método generalizado
    def encode_person_bipolar_datatype(self, raw_person: Dict[str, Any]) -> np.ndarray:
        """
        Método legacy - Utilizar encode_person_generalized en su lugar para nuevos desarrollos.
        Este método se mantiene por compatibilidad con código existente.

        Args:
            raw_person: Diccionario con los datos de la persona a codificar.

        Returns:
            Hipervector bipolar que representa a la persona.
        """
        print("ADVERTENCIA: Usando método deprecated. Considere migrar a encode_person_generalized")
        return self.encode_person_generalized(raw_person)

    # Nuevo método generalizado basado en estrategias
    def encode_person_generalized(self, raw_person: Dict[str, Any]) -> np.ndarray:
        """
        Codifica los datos de una persona utilizando estrategias basadas en tipos de datos.

        Args:
            raw_person: Diccionario con los datos de la persona a codificar.

        Returns:
            Hipervector bipolar que representa a la persona.
        """
        bundle_acc = self.bundle_init()
        person = normalize_person_data(raw_person)
        profiler = DataTypeProfiler()
        profiler.profile_record(person)

        for key in sorted(person.keys()):
            value = person[key]

            # Información de depuración
            print(f"Key: {key}, Type: {type(value).__name__}, Value: {repr(value)}")

            # Saltar valores vacíos
            if value is None:
                continue
            if isinstance(value, str) and not value:
                continue
            if isinstance(value, list) and not value:
                continue

            # Obtener el tipo de dato según el perfilador
            data_type = profiler.get_type(key)

            # Obtener la estrategia adecuada y codificar el valor
            strategy = self.strategy_factory.get_strategy(key, value, data_type)
            encoded_value = strategy.encode(key, value, profiler)

            # Vincular clave y valor codificado
            key_hv = self.get_bipolar_hv(key)
            bound_hv = self.bind_hv(key_hv, encoded_value)
            self.bundle_add(bundle_acc, bound_hv)

        # Mostrar el resumen del perfil
        profiler.print_summary()

        # Devolver el vector final
        return self.bundle_finalize(bundle_acc, tie_key="person_bundle")
