from typing import Any, List, Dict, Type
import torch
from hdc.datatype_profiler import DataTypeProfiler


class BinaryEncodingStrategy:
    """Clase base para las estrategias de codificación binaria."""

    def __init__(self, hdc_instance):
        self.hdc = hdc_instance

    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> torch.Tensor:
        raise NotImplementedError


class DefaultBinaryEncodingStrategy(BinaryEncodingStrategy):
    """Estrategia de codificación por defecto para cadenas (binario)."""

    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> torch.Tensor:
        # Devuelve un vector de ceros si el valor es None o está vacío
        if value is None or not str(value):
            return torch.zeros(self.hdc.dim, dtype=torch.uint8, device=self.hdc.device)
        return self.hdc.get_binary_hv(str(value))


class DateBinaryEncodingStrategy(BinaryEncodingStrategy):
    """Estrategia para codificar fechas en formato binario."""

    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> torch.Tensor:
        if not hasattr(value, 'year'):  # No es un objeto date/datetime
            return torch.zeros(self.hdc.dim, dtype=torch.uint8, device=self.hdc.device)

        return self.hdc.encode_date_binary(value)


class ListBinaryEncodingStrategy(BinaryEncodingStrategy):
    """Estrategia para codificar listas de cadenas (binario)."""

    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> torch.Tensor:
        if not isinstance(value, list):
            return DefaultBinaryEncodingStrategy(self.hdc).encode(key, value, profiler)

        item_vectors = []
        for item in value:
            # Para cada ítem en la lista, usamos la estrategia por defecto (tratar como string)
            item_hv = DefaultBinaryEncodingStrategy(self.hdc).encode(key, item, profiler)
            if item_hv is not None:
                item_vectors.append(item_hv)

        if not item_vectors:
            return torch.zeros(self.hdc.dim, dtype=torch.uint8, device=self.hdc.device)

        # Bundling de todos los items de la lista
        return self.hdc.bundle_hv(item_vectors)


class AttrsBinaryEncodingStrategy(BinaryEncodingStrategy):
    """Estrategia para codificar diccionarios de atributos (binario)."""

    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> torch.Tensor:
        if not isinstance(value, dict):
            return DefaultBinaryEncodingStrategy(self.hdc).encode(key, value, profiler)

        attr_vectors = []
        for sub_key, sub_value in sorted(value.items()):
            if not sub_value:
                continue

            # La sub-clave se usa para obtener su HV base
            sub_key_hv = self.hdc.get_binary_hv(sub_key.upper())

            # El sub-valor (generalmente una lista) se codifica con la estrategia de lista
            encoded_sub_value = ListBinaryEncodingStrategy(self.hdc).encode(sub_key, sub_value, profiler)

            if encoded_sub_value is not None:
                # Bind de la clave del atributo con su valor codificado
                bound_attr = self.hdc.bind_hv(sub_key_hv, encoded_sub_value)
                attr_vectors.append(bound_attr)

        if not attr_vectors:
            return torch.zeros(self.hdc.dim, dtype=torch.uint8, device=self.hdc.device)

        # Bundling de todos los atributos
        return self.hdc.bundle_hv(attr_vectors)


class BinaryHDCEncodingStrategyFactory:
    """Factory para crear y gestionar estrategias de codificación binaria."""

    def __init__(self, hdc_instance):
        self.hdc = hdc_instance
        self.strategies: Dict[str, Type[BinaryEncodingStrategy]] = {}

    def register_strategy(self, data_type: str, strategy: Type[BinaryEncodingStrategy]):
        self.strategies[data_type] = strategy

    def get_strategy(self, key: str, value: Any, data_type: str) -> BinaryEncodingStrategy:
        strategy_class = self.strategies.get(data_type)

        # Casos especiales o de fallback
        if value is None or (isinstance(value, (str, list)) and not value):
            strategy_class = self.strategies.get("EMPTY")

        if strategy_class:
            return strategy_class(self.hdc)

        # Fallback a la estrategia por defecto si no se encuentra una específica
        return DefaultBinaryEncodingStrategy(self.hdc)