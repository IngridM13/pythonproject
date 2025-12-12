from typing import Any, Dict
from datetime import date, datetime
import re


class DataTypeProfiler:
    """
    Clase extendida para perfilar y clasificar tipos de datos.
    Analiza y asigna categorías semánticas a diferentes campos de datos.
    """

    def __init__(self):
        self.type_registry = {}
        self.examples = {}

    def profile_record(self, record: Dict[str, Any]) -> None:
        """Analiza un registro completo y clasifica sus campos."""
        for key, value in record.items():
            data_type = self.classify(key, value)
            self.type_registry[key] = data_type
            self.examples[key] = value

    def get_type(self, key: str) -> str:
        """Obtiene el tipo clasificado para una clave."""
        return self.type_registry.get(key, "UNKNOWN")

    def classify(self, key: str, value: Any) -> str:
        """
        Clasifica un valor en un tipo de dato semántico.

        Args:
            key: Nombre del campo.
            value: Valor a clasificar.

        Returns:
            Categoría del tipo de dato (ej. "DATE", "TEXT_NAME", "NUMERIC", etc.)
        """
        # Tipos complejos
        if isinstance(value, dict):
            if key == "attrs" or self._is_attrs_dict(value):
                return "ATTRS_DICT"
            return "DICT"

        if isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                return "LIST_OF_STR"
            if all(isinstance(item, (int, float)) for item in value):
                return "LIST_OF_NUMBERS"
            return "LIST_MIXED"

        # Fechas
        if isinstance(value, (date, datetime)):
            return "DATE"

        # Valores numéricos
        if isinstance(value, (int, float)):
            return "NUMERIC"

        # Strings categóricos y especializados
        if isinstance(value, str):
            if self._is_empty(value):
                return "EMPTY"

            if key in ["gender", "race", "marital_status", "ethnicity", "relationship_status"]:
                return "CATEGORICAL_STR"

            if key in ["name", "firstname", "lastname", "surname"]:
                return "TEXT_NAME"

            if key in ["mobile_number", "phone", "telephone", "landline", "cell"]:
                return "PHONE_STR"

            if self._is_date_string(value):
                return "DATE_STR"

            if self._is_numeric_string(value):
                return "NUMERIC_STR"

            # Strings generales (podría refinarse más)
            return "TEXT_STR"

        # Valor nulo
        if value is None:
            return "NULL"

        # Tipo desconocido
        return f"UNKNOWN_{type(value).__name__}"

    def _is_attrs_dict(self, value: Dict) -> bool:
        """Determina si un diccionario es un diccionario de atributos (listas de valores)."""
        if not value:
            return False
        return all(isinstance(v, list) for v in value.values())

    def _is_empty(self, value: str) -> bool:
        """Verifica si una cadena está vacía o solo contiene espacios."""
        return value.strip() == ""

    def _is_date_string(self, value: str) -> bool:
        """Verifica si una cadena representa una fecha."""
        date_patterns = [
            r'^\d{4}-\d{1,2}-\d{1,2}$',  # YYYY-MM-DD
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # MM/DD/YY or MM/DD/YYYY
            r'^\d{1,2}-\d{1,2}-\d{2,4}$'  # DD-MM-YY or DD-MM-YYYY
        ]
        return any(re.match(pattern, value) for pattern in date_patterns)

    def _is_numeric_string(self, value: str) -> bool:
        """Verifica si una cadena representa un número."""
        return value.replace('.', '', 1).isdigit()

    def print_summary(self) -> None:
        """Imprime un resumen de los tipos de datos detectados."""
        print("\n=== DATA TYPE PROFILE ===")
        for key in sorted(self.type_registry.keys()):
            tipo = self.type_registry[key]
            ejemplo = repr(self.examples[key])
            # Truncar ejemplos largos
            if len(ejemplo) > 100:
                ejemplo = ejemplo[:97] + "..."
            print(f"{key:<15} -> {tipo}")
            print(f"    ejemplo: {ejemplo}")
        print("=========================\n")