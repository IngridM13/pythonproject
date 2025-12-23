import hashlib
from datetime import date
from typing import Any, List, Dict
import numpy as np

# Imports para la codificación generalizada
from hdc.binary_encoding_strategies import (
    BinaryHDCEncodingStrategyFactory,
    DateBinaryEncodingStrategy,
    AttrsBinaryEncodingStrategy,
    ListBinaryEncodingStrategy,
    DefaultBinaryEncodingStrategy,
)
from hdc.datatype_profiler import DataTypeProfiler
from utils.person_data_normalization import normalize_person_data

def binary_random(d, rng):
    """Devuelve un vector binario {0,1}^d usando la API moderna de NumPy."""
    return rng.choice(np.array([0, 1], dtype=np.uint8), size=d, replace=True)

# Inicializar el diccionario global para almacenar los hipervectores
hv_dict = {}

class HyperDimensionalComputingBinary:
    def __init__(self, dim=10000, seed=None):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._hv_cache = {}

        # Inicializar y registrar las estrategias de codificación
        self.strategy_factory = BinaryHDCEncodingStrategyFactory(self)
        self.register_default_strategies()

    def register_default_strategies(self):
        """Registra las estrategias de codificación binaria predeterminadas."""
        factory = self.strategy_factory
        factory.register_strategy("DATE", DateBinaryEncodingStrategy)
        factory.register_strategy("ATTRS_DICT", AttrsBinaryEncodingStrategy)
        factory.register_strategy("LIST_OF_STR", ListBinaryEncodingStrategy)
        factory.register_strategy("CATEGORICAL_STR", DefaultBinaryEncodingStrategy)
        factory.register_strategy("TEXT_NAME", DefaultBinaryEncodingStrategy)
        factory.register_strategy("PHONE_STR", DefaultBinaryEncodingStrategy)
        factory.register_strategy("EMPTY", DefaultBinaryEncodingStrategy)
        factory.register_strategy("UNKNOWN", DefaultBinaryEncodingStrategy)

    def _deterministic_hash(self, input_str: str) -> int:
        """Genera un hash determinista como un entero de 32 bits para usar como semilla."""
        full_hash = int(hashlib.sha256(input_str.encode('utf-8')).hexdigest(), 16)
        return full_hash % (2**32)

    def get_binary_hv(self, key: str) -> np.ndarray:
        """
        Genera o recupera un hipervector binario determinista para una clave dada.
        Utiliza un hash de la clave para sembrar el generador de números aleatorios,
        asegurando que la misma clave siempre produzca el mismo hipervector.
        Cachea el resultado en el diccionario global hv_dict.
        """
        global hv_dict
        if key not in hv_dict:
            # Imprimir un mensaje para confirmar que se está generando un nuevo vector
            print(f"  [get_binary_hv] Generando nuevo HV determinista para la clave: '{key}'")
            
            # Usar un generador de estado aleatorio sembrado con el hash de la clave
            # Esto asegura que la generación sea determinista.
            try:
                # Usar hashlib para un hash más robusto y consistente entre ejecuciones
                import hashlib
                # El hash de Python puede variar entre sesiones, hashlib es más estable
                seed_str = str(key)
                hash_obj = hashlib.sha256(seed_str.encode('utf-8'))
                seed = int(hash_obj.hexdigest(), 16) % (2**32 - 1)
            except Exception:
                # Fallback al hash de Python si hashlib falla
                seed = hash(key) % (2**32 - 1)

            rng = np.random.RandomState(seed)
            
            # Generar el hipervector binario
            hv = rng.randint(0, 2, self.dim, dtype=np.uint8)
            hv_dict[key] = hv
        
        return hv_dict[key]

    def bind_hv(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Binding binario (XOR)."""
        return np.logical_xor(x, y).astype(np.uint8, copy=False)

    def bundle_hv(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Realiza la operación de bundling sobre una lista de vectores binarios
        de forma determinista.
        """
        if not vectors:
            return np.zeros(self.dim, dtype=np.uint8)

        # Suma los vectores componente a componente
        sum_vec = np.sum(vectors, axis=0, dtype=np.int32)

        # El umbral para la votación de mayoría
        threshold = len(vectors) / 2.0

        # Compara la suma con el umbral para obtener el resultado.
        # La comparación `> threshold` asegura que si sum_vec == threshold (un empate),
        # el resultado será False, que se convierte en 0.
        # Esto hace que el desempate sea determinista.
        bundle = (sum_vec > threshold).astype(np.uint8)

        return bundle

    def hamming_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Calcula la similitud de Hamming entre dos vectores binarios."""
        if hv1.shape != hv2.shape or len(hv1.shape) != 1:
            raise ValueError("Los vectores deben tener la misma forma y ser 1D.")
        
        hamming_dist = np.logical_xor(hv1, hv2).sum()
        return 1.0 - (hamming_dist / self.dim)

    def _thermometer(self, name: str, value: float, min_val: float, max_val: float, resolution: int = 100) -> np.ndarray:
        """Genera un vector de termómetro binario de forma determinista."""
        seed = self._deterministic_hash(name)
        rng = np.random.RandomState(seed) # RandomState es necesario para la permutación determinista con semilla
        permuted_indices = rng.permutation(self.dim)
        
        value = max(min_val, min(max_val, value)) # Clamp value
        
        proportion = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        num_bits_to_set = int(self.dim * proportion)

        vec = np.zeros(self.dim, dtype=np.uint8)
        vec[permuted_indices[:num_bits_to_set]] = 1
        return vec

    def encode_date_binary(self, value: date) -> np.ndarray:
        """
        Codifica una fecha en un vector hiperdimensional binario utilizando su número ordinal
        para garantizar la monotonicidad de la similitud.
        """
        if not isinstance(value, date):
            raise TypeError("El valor debe ser un objeto de tipo date.")

        # Usar un rango fijo para las fechas, p.ej., desde el año 1900 al 2100.
        min_date_ord = date(1900, 1, 1).toordinal()
        max_date_ord = date(2100, 12, 31).toordinal()
        
        value_ord = value.toordinal()

        # El método del termómetro ahora es monotónico por construcción.
        date_therm_vec = self._thermometer(
            name='date_ordinal', 
            value=value_ord, 
            min_val=min_date_ord, 
            max_val=max_date_ord
        )

        date_base_hv = self.get_binary_hv("DATE_ORDINAL_BASE")
        return self.bind_hv(date_base_hv, date_therm_vec)

    def encode_person_binary(self, raw_person: Dict[str, Any]) -> np.ndarray:
        """
        Codifica los datos de una persona en un hipervector binario utilizando estrategias
        basadas en tipos de datos.
        """
        person = normalize_person_data(raw_person)
        profiler = DataTypeProfiler()
        profiler.profile_record(person)

        all_field_vectors = []

        for key in sorted(person.keys()):
            value = person[key]

            # Saltar valores vacíos explícitamente, igual que en la versión bipolar
            if value is None or (isinstance(value, (str, list, dict)) and not value):
                continue
            
            data_type = profiler.get_type(key)
            
            strategy = self.strategy_factory.get_strategy(key, value, data_type)
            encoded_value = strategy.encode(key, value, profiler)

            if encoded_value is not None:
                key_hv = self.get_binary_hv(key.upper())
                bound_hv = self.bind_hv(key_hv, encoded_value)
                all_field_vectors.append(bound_hv)

        if not all_field_vectors:
            return np.zeros(self.dim, dtype=np.uint8)

        return self.bundle_hv(all_field_vectors)