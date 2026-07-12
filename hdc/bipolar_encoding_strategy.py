from typing import Any, Dict, List
import torch
from hdc.datatype_profiler import DataTypeProfiler
from utils.person_data_normalization import normalize_person_data


class BipolarEncodingStrategy:
    """Estrategia base para codificación de diferentes tipos de datos en HDC."""

    def __init__(self, encoder):
        self.encoder = encoder

    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> torch.Tensor:
        """Método abstracto para codificar un valor basado en su tipo."""
        raise NotImplementedError("Las subclases deben implementar este método")


class DefaultEncodingStrategy(BipolarEncodingStrategy):
    """Estrategia de codificación por defecto (string)."""

    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> torch.Tensor:
        return self.encoder.get_bipolar_hv(str(value))


class DateEncodingStrategy(BipolarEncodingStrategy):
    """Estrategia de codificación para fechas."""

    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> torch.Tensor:
        return self.encoder.encode_date_bipolar(value)


class ListEncodingStrategy(BipolarEncodingStrategy):
    """Estrategia de codificación para listas."""

    def encode(self, key: str, value: List[Any], profiler: DataTypeProfiler) -> torch.Tensor:
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

            # Procesar la lista de valores para este atributo
            list_acc = self.encoder.bundle_init()
            vectors_to_add = [self.encoder.get_bipolar_hv(str(v)) for v in attr_value_list]
            self.encoder.bundle_add(list_acc, *vectors_to_add)
            encoded_list_hv = self.encoder.bundle_finalize(list_acc, tie_key=f"list:{attr_key}")

            # Vincular con la clave del atributo
            attr_key_hv = self.encoder.get_bipolar_hv(attr_key)
            bound_attr_hv = self.encoder.bind_hv(attr_key_hv, encoded_list_hv)
            self.encoder.bundle_add(attrs_acc, bound_attr_hv)

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