from typing import Any, Dict, Optional
import numpy as np
from datetime import date, datetime
from hdc.datatype_profiler import DataTypeProfiler

class BinaryEncodingStrategy:
    """Estrategia base para codificación binaria."""
    def __init__(self, hdc_instance):
        self.hdc = hdc_instance
    
    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> np.ndarray:
        """Método base que debe ser implementado por las clases hijas."""
        raise NotImplementedError("Las subclases deben implementar este método")

class DefaultBinaryEncodingStrategy(BinaryEncodingStrategy):
    """Estrategia por defecto para codificación binaria."""
    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> np.ndarray:
        return self.hdc.get_binary_hv(str(value))

class DateBinaryEncodingStrategy(BinaryEncodingStrategy):
    """Estrategia para codificar fechas en formato binario sin periodicidad."""
    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> np.ndarray:
        if isinstance(value, (date, datetime)):
            return self.hdc.encode_date_binary(value)
        elif isinstance(value, str):
            try:
                # Intentar convertir string a fecha
                from utils.person_data_normalization import parse_date
                date_obj = parse_date(value)
                if date_obj:
                    return self.hdc.encode_date_binary(date_obj)
            except:
                pass
        
        # Si no se pudo manejar como fecha, usar codificación por defecto
        return DefaultBinaryEncodingStrategy(self.hdc).encode(key, value, profiler)

class ListBinaryEncodingStrategy(BinaryEncodingStrategy):
    """Estrategia para codificar listas en formato binario."""
    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> np.ndarray:
        if not value:  # Lista vacía
            return np.zeros(self.hdc.dim, dtype=np.uint8)
            
        list_acc = self.hdc.bundle_init()
        vectors_to_add = [self.hdc.get_binary_hv(str(v)) for v in value if v]
        
        if not vectors_to_add:
            return np.zeros(self.hdc.dim, dtype=np.uint8)
            
        self.hdc.bundle_add(list_acc, *vectors_to_add)
        return self.hdc.bundle_finalize(list_acc, num_components=len(vectors_to_add))

class AttrsBinaryEncodingStrategy(BinaryEncodingStrategy):
    """Estrategia para codificar diccionarios de atributos en formato binario."""
    def encode(self, key: str, value: Any, profiler: DataTypeProfiler) -> np.ndarray:
        if not isinstance(value, dict):
            return DefaultBinaryEncodingStrategy(self.hdc).encode(key, value, profiler)
            
        attrs_acc = self.hdc.bundle_init()
        num_components = 0
        
        for attr_key, attr_value in sorted(value.items()):
            if attr_value is None or (isinstance(attr_value, (list, str)) and not attr_value):
                continue
                
            # Obtener el tipo de dato para este atributo
            attr_type = profiler.get_type(f"{key}.{attr_key}")
            
            # Obtener la estrategia para este tipo
            strategy = self.hdc.strategy_factory.get_strategy(attr_key, attr_value, attr_type)
            encoded_attr = strategy.encode(attr_key, attr_value, profiler)
            
            # Vincular clave y valor
            attr_key_hv = self.hdc.get_binary_hv(attr_key)
            bound_attr = self.hdc.bind_hv(attr_key_hv, encoded_attr)
            
            # Añadir al acumulador
            self.hdc.bundle_add(attrs_acc, bound_attr)
            num_components += 1
            
        if num_components == 0:
            return np.zeros(self.hdc.dim, dtype=np.uint8)
            
        return self.hdc.bundle_finalize(attrs_acc, num_components=num_components)

class BinaryHDCEncodingStrategyFactory:
    """Factory para crear estrategias de codificación binaria."""
    
    def __init__(self, hdc_instance):
        self.hdc = hdc_instance
        self.strategies = {}
        
    def register_strategy(self, data_type: str, strategy_class) -> None:
        """Registra una estrategia para un tipo de dato específico."""
        self.strategies[data_type] = strategy_class
        
    def get_strategy(self, key: str, value: Any, data_type: str) -> BinaryEncodingStrategy:
        """
        Obtiene la estrategia apropiada para codificar un valor.
        
        Args:
            key: Clave del campo
            value: Valor a codificar
            data_type: Tipo de dato según el perfilador
            
        Returns:
            La estrategia apropiada para codificar el valor
        """
        strategy_class = self.strategies.get(data_type, DefaultBinaryEncodingStrategy)
        return strategy_class(self.hdc)

    # Continúa en binary_encoding_strategy.py
    class BinaryHDCEncodingStrategyFactory:
        """Factory para crear estrategias de codificación binaria."""

        def __init__(self, hdc_instance):
            self.hdc = hdc_instance
            self.strategies = {}

        def register_strategy(self, data_type: str, strategy_class) -> None:
            """Registra una estrategia para un tipo de dato específico."""
            self.strategies[data_type] = strategy_class

        def get_strategy(self, key: str, value: Any, data_type: str) -> BinaryEncodingStrategy:
            """
            Obtiene la estrategia apropiada para codificar un valor.

            Args:
                key: Clave del campo
                value: Valor a codificar
                data_type: Tipo de dato según el perfilador

            Returns:
                La estrategia apropiada para codificar el valor
            """
            strategy_class = self.strategies.get(data_type, DefaultBinaryEncodingStrategy)
            return strategy_class(self.hdc)

