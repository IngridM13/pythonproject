import numpy as np
from datetime import date
from configs.settings import HDC_DIM, DEFAULT_SEED
from hdc.binary_encoding_strategies import BinaryHDCEncodingStrategyFactory, DateBinaryEncodingStrategy, \
    AttrsBinaryEncodingStrategy, ListBinaryEncodingStrategy, DefaultBinaryEncodingStrategy
from hdc.datatype_profiler import DataTypeProfiler
from hdc.hdc_common_operations import (
    binary_random, shifting
)
from typing import Optional, Dict, Any, Iterable
import hashlib

from utils.person_data_normalization import normalize_person_data


class HyperDimensionalComputingBinary:
    """Implementa operaciones binarias HDC con vectores en {0,1} (dtype=uint8)."""

    def __init__(self, dim: int = HDC_DIM, seed: Optional[int] = DEFAULT_SEED):
        self.dim = dim
        self.seed = seed
        # ✅ API moderna de NumPy: Generator
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        # Inicializar el caché interno
        self._hv_cache: Dict[str, np.ndarray] = {}

        # Atributos para codificación escalar de fechas (sin periodicidad)
        self._date_thresholds: Optional[np.ndarray] = None
        self._max_range_days: int = 365 * 200  # Mismo rango que bipolar

        # Inicializar factory de estrategias
        self.strategy_factory = BinaryHDCEncodingStrategyFactory(self)
        self.register_default_strategies()

    def register_default_strategies(self):
        """Registra las estrategias de codificación binarias predeterminadas."""
        factory = self.strategy_factory
        factory.register_strategy("DATE", DateBinaryEncodingStrategy)
        factory.register_strategy("ATTRS_DICT", AttrsBinaryEncodingStrategy)
        factory.register_strategy("LIST_OF_STR", ListBinaryEncodingStrategy)

        # Otras estrategias
        factory.register_strategy("CATEGORICAL_STR", DefaultBinaryEncodingStrategy)
        factory.register_strategy("TEXT_NAME", DefaultBinaryEncodingStrategy)
        factory.register_strategy("PHONE_STR", DefaultBinaryEncodingStrategy)

    # ------------------------------------------------------------------
    # Inicialización de thresholds para fechas (sin periodicidad)
    # ------------------------------------------------------------------
    def _init_date_thresholds(self):
        """Inicializa thresholds aleatorios para el encoding escalar de fechas binarias."""
        if self._date_thresholds is not None:
            return

        # Semilla fija para reproducibilidad (igual que en bipolar)
        rng = np.random.default_rng(54321)

        # thresholds uniformes en [0, max_range_days] (igual que en bipolar)
        self._date_thresholds = rng.integers(
            low=0,
            high=self._max_range_days + 1,
            size=self.dim,
            dtype=np.int32
        )

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

        threshold = num_components // 2
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

    # ---- Deterministic HVs ----

    def get_binary_hv(self, key: Any) -> np.ndarray:
        """
        Obtiene un hipervector binario reproducible para una clave.
        Usa el caché interno de la clase.
        """
        key_str = str(key)

        # Usar siempre el caché interno
        if key_str in self._hv_cache:
            return self._hv_cache[key_str]

        seed = self._deterministic_hash(key_str)
        # ✅ API moderna: Generator determinista por clave
        temp_rng = np.random.default_rng(seed)
        hv = binary_random(self.dim, temp_rng)

        # Guardar en el caché interno
        self._hv_cache[key_str] = hv
        return hv

    def _deterministic_hash(self, key_str: str) -> int:
        """Genera un hash determinístico para una clave string."""
        key_bytes = str(key_str).encode("utf-8")
        h = hashlib.md5(key_bytes).digest()
        return int.from_bytes(h[:8], "little") % (2 ** 32)

    # ---- ENCODING METHODS  ----

    # ------------------------------------------------------------------
    # Método generalizado para codificar personas (con estrategias)
    # ------------------------------------------------------------------
    def encode_person_binary(self, raw_person: Dict[str, Any]) -> np.ndarray:
        """
        Codifica los datos de una persona utilizando estrategias basadas en tipos de datos.

        Args:
            raw_person: Diccionario con los datos de la persona a codificar.

        Returns:
            Hipervector binario que representa a la persona.
        """
        bundle_acc = self.bundle_init()
        person = normalize_person_data(raw_person)
        profiler = DataTypeProfiler()
        profiler.profile_record(person)

        num_components = 0

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

            # Vincular clave y valor codificado (XOR para binario)
            key_hv = self.get_binary_hv(key)
            bound_hv = self.bind_hv(key_hv, encoded_value)
            self.bundle_add(bundle_acc, bound_hv)
            num_components += 1

        # Mostrar el resumen del perfil
        profiler.print_summary()

        # Devolver el vector final usando voto mayoritario
        return self.bundle_finalize(bundle_acc, num_components=num_components)

    # ------------------------------------------------------------------
    # Método de encoding escalar para fechas (sin periodicidad)
    # ------------------------------------------------------------------
    def encode_date_binary(self, date_obj: Optional[date]) -> np.ndarray:
        """
        Encoding binario escalar de fechas (sin periodicidad artificial).
        Preserva distancias naturales entre fechas.

        Args:
            date_obj: Objeto de fecha o None

        Returns:
            Vector binario {0,1} representando la fecha
        """
        if date_obj is None:
            # Vector neutro para binding con XOR
            return np.zeros(self.dim, dtype=np.uint8)

        self._init_date_thresholds()

        reference_date = date(1970, 1, 1)  # Misma referencia que bipolar
        days_since_reference = (date_obj - reference_date).days

        # Acotar al rango soportado
        t = np.int32(
            max(0, min(days_since_reference, self._max_range_days))
        )

        thresholds = self._date_thresholds  # shape (dim,)

        # Regla: hv_j = 1 si t >= θ_j, 0 en otro caso (versión binaria)
        hv = np.where(t >= thresholds, 1, 0).astype(np.uint8)
        return hv

