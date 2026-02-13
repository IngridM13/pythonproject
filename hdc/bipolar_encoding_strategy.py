from typing import Any, Dict, List, Optional, Callable
import torch
from datetime import date, datetime
from hdc.datatype_profiler import DataTypeProfiler
from utils.person_data_normalization import normalize_person_data


class BipolarEncodingStrategy:
    """Estrategia base para codificación de diferentes tipos de datos en HDC."""

    def __init__(self, encoder):
        self.encoder = encoder

    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> torch.Tensor:
        """Método abstracto para codificar un valor basado en su tipo."""
        raise NotImplementedError("Las subclases deben implementar este método")

    def _debug_log(self, message: str):
        """Método auxiliar para mostrar mensajes de depuración."""
        print(f"  {message}")


class DefaultEncodingStrategy(BipolarEncodingStrategy):
    """Estrategia de codificación por defecto (string)."""

    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> torch.Tensor:
        self._debug_log(f"Usando codificación por defecto para {key} (tipo: {type(value).__name__})")
        return self.encoder.get_bipolar_hv(str(value))


class DateEncodingStrategy(BipolarEncodingStrategy):
    """Estrategia de codificación para fechas."""

    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> torch.Tensor:
        self._debug_log(f"Procesando fecha para {key}: {value}")
        return self.encoder.encode_date_bipolar(value)


class ListEncodingStrategy(BipolarEncodingStrategy):
    """Estrategia de codificación para listas."""

    def encode(self, key: str, value: List[Any], profiler: DataTypeProfiler) -> torch.Tensor:
        self._debug_log(f"Procesando lista para {key}, longitud: {len(value)}")
        list_acc = self.encoder.bundle_init()
        vectors_to_add = [self.encoder.get_bipolar_hv(str(v)) for v in value]
        self.encoder.bundle_add(list_acc, *vectors_to_add)
        return self.encoder.bundle_finalize(list_acc, tie_key=f"list:{key}")


class AttrsEncodingStrategy(BipolarEncodingStrategy):
    """Estrategia de codificación para diccionarios de atributos (attrs)."""

    def encode(self, key: str, value: Dict[str, List[Any]], profiler: DataTypeProfiler) -> torch.Tensor:
        attrs_acc = self.encoder.bundle_init()

        for attr_key in sorted(value.keys()):
            attr_value_list = value[attr_key]
            if not attr_value_list:
                continue  # Skip empty lists

            self._debug_log(f"Attr: {attr_key}, Type: {type(attr_value_list).__name__}, Value: {repr(attr_value_list)}")

            # Procesar la lista de valores para este atributo
            list_acc = self.encoder.bundle_init()
            vectors_to_add = [self.encoder.get_bipolar_hv(str(v)) for v in attr_value_list]
            self.encoder.bundle_add(list_acc, *vectors_to_add)
            encoded_list_hv = self.encoder.bundle_finalize(list_acc, tie_key=f"list:{attr_key}")

            # Vincular con la clave del atributo
            attr_key_hv = self.encoder.get_bipolar_hv(attr_key)
            bound_attr_hv = self.encoder.bind_hv(attr_key_hv, encoded_list_hv)
            self.encoder.bundle_add(attrs_acc, bound_attr_hv)

        print(f"[DEBUG-ENCODE] Finalizando 'attrs_bundle' (Datos presentes: {len(value) > 0}).")
        return self.encoder.bundle_finalize(attrs_acc, tie_key="attrs_bundle")


class BipolarEncodingStrategyFactory:
    """
    Factory que crea estrategias de codificación basadas en el tipo de dato.
    """

    def __init__(self, encoder):
        self.encoder = encoder
        self.strategies = {}

    def register_strategy(self, data_type: str, strategy_class: type):
        """Registra una estrategia de codificación para un tipo de dato."""
        self.strategies[data_type] = strategy_class(self.encoder)

    def get_strategy(self, key: str, value: Any, data_type: str) -> BipolarEncodingStrategy:
        """Devuelve la estrategia de codificación apropiada para el tipo de dato."""
        strategy = self.strategies.get(data_type)
        if strategy is None:
            return DefaultEncodingStrategy(self.encoder)
        return strategy


class GeneralizedBipolarHDC:
    """
    Implementación generalizada de codificación HDC bipolar con soporte para tipos de datos.
    Extiende la clase BipolarHDC original pero utiliza un sistema de estrategias para la codificación.
    """

    def __init__(self, dim=10000, seed=None):
        """
        Inicializa el codificador HDC bipolar generalizado.

        Args:
            dim: Dimensión de los hipervectores.
            seed: Semilla para la generación de números aleatorios.
        """
        # Nota: Aquí deberías inicializar la clase base o reimplementar los métodos necesarios
        # para que este ejemplo sea autónomo, estoy suponiendo que los métodos como bundle_init,
        # bundle_add, etc. estarán disponibles

        self.dim = dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize generator for reproducibility
        if seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(seed)
        else:
            self.generator = None

        # Inicializar el factory de estrategias
        self.strategy_factory = BipolarEncodingStrategyFactory(self)
        self.register_default_strategies()

    def register_default_strategies(self):
        """Registra las estrategias de codificación predeterminadas."""
        factory = self.strategy_factory
        factory.register_strategy("DATE", DateEncodingStrategy)
        factory.register_strategy("ATTRS_DICT", AttrsEncodingStrategy)
        factory.register_strategy("LIST_OF_STR", ListEncodingStrategy)

        # Puedes registrar más estrategias específicas aquí
        factory.register_strategy("CATEGORICAL_STR", DefaultEncodingStrategy)
        factory.register_strategy("TEXT_NAME", DefaultEncodingStrategy)
        factory.register_strategy("PHONE_STR", DefaultEncodingStrategy)

    def bundle_init(self):
        """Inicializa un acumulador para operaciones de bundling."""
        return torch.zeros(self.dim, dtype=torch.float32, device=self.device)

    def bundle_add(self, bundle_acc, *vectors):
        """Agrega vectores al acumulador de bundle."""
        for v in vectors:
            # Ensure v is a tensor on the correct device
            if isinstance(v, torch.Tensor):
                v_tensor = v.to(self.device)
            else:  # Handle numpy arrays or other types
                v_tensor = torch.tensor(v, device=self.device)

            bundle_acc += v_tensor

    def bundle_finalize(self, bundle_acc, tie_key=None):
        """Finaliza el bundle aplicando la función signo."""
        # Implementación básica, debería adaptarse al método original
        return torch.sign(bundle_acc).to(torch.int8)

    def bind_hv(self, hv1, hv2):
        """Vincula dos hipervectores usando multiplicación elemento a elemento."""
        # Ensure both are tensors on the correct device
        if not isinstance(hv1, torch.Tensor):
            hv1 = torch.tensor(hv1, device=self.device)
        if not isinstance(hv2, torch.Tensor):
            hv2 = torch.tensor(hv2, device=self.device)

        return hv1 * hv2

    def get_bipolar_hv(self, key):
        """Genera un hipervector bipolar determinista para una clave."""
        # Use a more straightforward way to generate bipolar values
        hash_val = hash(key) % 10000

        # Create a deterministic generator
        if self.generator is not None:
            g = self.generator.manual_seed(hash_val)
        else:
            g = torch.Generator(device='cpu').manual_seed(hash_val)

        # Generate random values between 0 and 1, then convert to -1/+1
        random_values = torch.rand(self.dim, generator=g, device='cpu')
        bipolar_vector = torch.where(
            random_values < 0.5,
            torch.tensor(-1, dtype=torch.int8, device='cpu'),
            torch.tensor(1, dtype=torch.int8, device='cpu')
        ).to(self.device)

        return bipolar_vector

    def encode_date_bipolar(self, date_value):
        """Codifica una fecha en un hipervector bipolar."""
        # Implementación usando PyTorch
        if isinstance(date_value, (date, datetime)):
            date_str = date_value.isoformat()
            return self.get_bipolar_hv(date_str)
        return self.get_bipolar_hv(str(date_value))