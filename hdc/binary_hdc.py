import hashlib
import torch
from datetime import date
from typing import Any, List, Dict, Optional, Union, Iterable

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
from configs.settings import HDC_DIM, DEFAULT_SEED


def binary_random(d, rng):
    """Devuelve un vector binario {0,1}^d usando PyTorch."""
    return torch.randint(0, 2, (d,), dtype=torch.uint8, generator=rng)


class HyperDimensionalComputingBinary:
    """Binary HDC with vectors in {0,1} using PyTorch tensors."""

    def __init__(self, dim=HDC_DIM, seed=DEFAULT_SEED, device=None):
        self.dim = dim
        self.seed = seed

        # Set device - default to CUDA if available
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")

        if seed is not None:
            torch.manual_seed(seed)

        # Use CPU generator explicitly (PyTorch requires generators on CPU)
        self.rng = torch.Generator(device='cpu')
        if seed is not None:
            self.rng.manual_seed(seed)

        # Use instance cache rather than global
        self._hv_cache: Dict[str, torch.Tensor] = {}

        # Import the strategy factory from the existing file
        from hdc.binary_encoding_strategies import (
            BinaryHDCEncodingStrategyFactory,
            DefaultBinaryEncodingStrategy,
            DateBinaryEncodingStrategy,
            ListBinaryEncodingStrategy,
            AttrsBinaryEncodingStrategy
        )

        # Initialize the strategy factory
        self.strategy_factory = BinaryHDCEncodingStrategyFactory(self)

        # Register default strategies
        self.strategy_factory.register_strategy("DATE", DateBinaryEncodingStrategy)
        self.strategy_factory.register_strategy("ATTRS_DICT", AttrsBinaryEncodingStrategy)
        self.strategy_factory.register_strategy("LIST_OF_STR", ListBinaryEncodingStrategy)
        self.strategy_factory.register_strategy("CATEGORICAL_STR", DefaultBinaryEncodingStrategy)
        self.strategy_factory.register_strategy("TEXT_NAME", DefaultBinaryEncodingStrategy)
        self.strategy_factory.register_strategy("PHONE_STR", DefaultBinaryEncodingStrategy)
        self.strategy_factory.register_strategy("EMPTY", DefaultBinaryEncodingStrategy)
        self.strategy_factory.register_strategy("UNKNOWN", DefaultBinaryEncodingStrategy)

    def _deterministic_hash(self, input_str: str) -> int:
        """Genera un hash determinista como un entero de 32 bits para usar como semilla."""
        full_hash = int(hashlib.sha256(input_str.encode('utf-8')).hexdigest(), 16)
        return full_hash % (2 ** 32)

    def get_binary_hv(self, key: str) -> torch.Tensor:
        """
        Genera o recupera un hipervector binario determinista para una clave dada.
        Utiliza un hash de la clave para sembrar el generador de números aleatorios,
        asegurando que la misma clave siempre produzca el mismo hipervector.
        """
        key_str = str(key)

        if key_str in self._hv_cache:
            return self._hv_cache[key_str].to(self.device)

        # Imprimir un mensaje para confirmar que se está generando un nuevo vector
        print(f"  [get_binary_hv] Generando nuevo HV determinista para la clave: '{key_str}'")

        # Generar semilla determinista
        seed = self._deterministic_hash(key_str)

        # Configurar un generador de PyTorch con la semilla derivada
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)

        # Generar el hipervector binario y mover al dispositivo
        hv = torch.randint(0, 2, (self.dim,), dtype=torch.uint8, generator=rng).to(self.device)
        self._hv_cache[key_str] = hv

        return hv

    def bind_hv(self, x: Union[torch.Tensor, Any], y: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Binding binario (XOR) usando PyTorch."""
        # Asegurarse de que los inputs son tensores de PyTorch
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.uint8, device=self.device)
        else:
            x = x.to(self.device)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.uint8, device=self.device)
        else:
            y = y.to(self.device)

        return torch.logical_xor(x, y).to(torch.uint8)

    def bind_batch(self, x_batch: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Binding binario (XOR) para lotes."""
        # Ensure tensors are on the correct device
        x_batch = x_batch.to(self.device)
        y = y.to(self.device)

        # Handle broadcasting: if y is 1D, unsqueeze to add batch dimension
        if y.dim() == 1:
            y = y.unsqueeze(0).expand(x_batch.size(0), -1)

        # XOR operation
        return torch.logical_xor(x_batch, y).to(torch.uint8)

    def bundle_hv(self, vectors: List[Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Realiza la operación de bundling sobre una lista de vectores binarios
        de forma determinista usando PyTorch.

        Implementa votación mayoritaria con manejo seguro de tipos y desempate determinista.
        Los vectores de entrada deben tener valores en {0,1}.
        """
        if not vectors:
            return torch.zeros(self.dim, dtype=torch.uint8, device=self.device)

        # Convertir cada vector a tensor de PyTorch int32 para evitar overflow en la suma
        tensor_vectors = []
        for v in vectors:
            if not isinstance(v, torch.Tensor):
                tensor_vectors.append(torch.tensor(v, dtype=torch.int32, device=self.device))
            else:
                # Asegurar conversión a int32 para prevenir overflow
                tensor_vectors.append(v.to(dtype=torch.int32, device=self.device))

        # Apilar los vectores y sumar a lo largo del eje 0 (seguros en int32)
        stacked = torch.stack(tensor_vectors)
        sum_vec = torch.sum(stacked, dim=0, dtype=torch.int32)

        # El umbral para la votación de mayoría
        threshold = len(vectors) / 2.0

        # Manejo de empates: en caso de empate exacto, usar un desempate determinista
        # basado en un hash de las posiciones para evitar sesgo sistemático
        if torch.any(sum_vec == int(threshold)):
            # Crear máscara de posiciones con empate
            ties_mask = (sum_vec == int(threshold))

            # Usar un tensor determinista para desempatar
            if not hasattr(self, '_tie_breaker'):
                # Generar un vector de desempate determinista basado en la semilla
                # Este vector solo se calcula una vez y se reutiliza
                rng = torch.Generator(device='cpu')
                rng.manual_seed(self._deterministic_hash("tie_breaker"))
                self._tie_breaker = torch.randint(0, 2, (self.dim,),
                                                  dtype=torch.uint8,
                                                  generator=rng).to(self.device)

            # Resolver los empates usando el vector de desempate
            # Donde hay empates, usar el tie_breaker
            sum_vec = sum_vec.to(torch.float32)  # Convertir a float para comparación
            sum_vec[ties_mask] = threshold + (self._tie_breaker[ties_mask].to(torch.float32) - 0.5)

        # Aplicar votación mayoritaria
        bundle = (sum_vec > threshold).to(torch.uint8)

        # Verificación final: asegurar que el resultado tenga estrictamente valores en {0,1}
        # Esta verificación es técnicamente innecesaria dada la implementación anterior,
        # pero se incluye como medida de seguridad adicional
        if not torch.all((bundle == 0) | (bundle == 1)):
            raise ValueError("El resultado del bundle_hv contiene valores fuera del rango {0,1}")

        return bundle

    def bundle_batch(self, vectors_batch: torch.Tensor) -> torch.Tensor:
        """
        Versión optimizada para procesar lotes de vectores.
        vectors_batch: tensor de forma [batch_size, n_vectors, dim]
        returns: tensor de forma [batch_size, dim]
        """
        vectors_batch = vectors_batch.to(self.device)

        # Sum along vector dimension (dim=1)
        sum_vec = torch.sum(vectors_batch, dim=1, dtype=torch.int32)

        # Threshold based on number of vectors
        threshold = vectors_batch.size(1) / 2.0

        # Apply threshold
        bundle = (sum_vec > threshold).to(torch.uint8)

        return bundle

    def hamming_similarity(self, hv1: Union[torch.Tensor, Any], hv2: Union[torch.Tensor, Any]) -> float:
        """Calcula la similitud de Hamming entre dos vectores binarios usando PyTorch."""
        # Asegurarse de que los inputs son tensores de PyTorch
        if not isinstance(hv1, torch.Tensor):
            hv1 = torch.tensor(hv1, dtype=torch.uint8, device=self.device)
        else:
            hv1 = hv1.to(self.device)

        if not isinstance(hv2, torch.Tensor):
            hv2 = torch.tensor(hv2, dtype=torch.uint8, device=self.device)
        else:
            hv2 = hv2.to(self.device)

        if hv1.shape != hv2.shape:
            raise ValueError(f"Los vectores deben tener la misma forma: {hv1.shape} vs {hv2.shape}")

        # For 1D vectors
        if hv1.dim() == 1:
            hamming_dist = torch.logical_xor(hv1, hv2).sum().item()
            return 1.0 - (hamming_dist / self.dim)

        # For batched vectors, return similarities for each pair
        hamming_dist = torch.logical_xor(hv1, hv2).sum(dim=-1)
        return 1.0 - (hamming_dist / self.dim)

    def hamming_similarity_batch(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Calcula similitud de Hamming entre un vector de consulta y varios vectores clave.
        query: tensor de forma [dim]
        keys: tensor de forma [n_keys, dim]
        returns: tensor de forma [n_keys] con similitudes
        """
        query = query.to(self.device)
        keys = keys.to(self.device)

        # Ensure query is properly shaped for broadcasting
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, dim]

        # XOR to find differences
        differences = torch.logical_xor(query, keys).float()

        # Sum differences along dimension
        hamming_distances = torch.sum(differences, dim=1)

        # Calculate similarities
        similarities = 1.0 - (hamming_distances / self.dim)

        return similarities

    def _thermometer_batch(self, name: str, values: List[float], min_val: float, max_val: float) -> torch.Tensor:
        """
        Versión por lotes de thermometer para procesar múltiples valores a la vez.
        Optimizada para usar operaciones de PyTorch.
        """
        seed = self._deterministic_hash(name)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)

        # Crear permutación una sola vez
        indices = torch.arange(self.dim)
        perm = torch.randperm(self.dim, generator=rng)
        permuted_indices = indices[perm]

        # Procesar todos los valores en un batch
        batch_size = len(values)
        values_tensor = torch.tensor(values, device=self.device)

        # Clamp values
        values_tensor = torch.clamp(values_tensor, min_val, max_val)

        # Calculate proportions and bits to set
        proportions = (values_tensor - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(
            values_tensor)
        num_bits = (proportions * self.dim).int()

        # Create result tensor
        result = torch.zeros((batch_size, self.dim), dtype=torch.uint8, device=self.device)

        # Set bits for each vector in batch
        for i, n_bits in enumerate(num_bits):
            if n_bits > 0:
                result[i, permuted_indices[:n_bits]] = 1

        return result

    def encode_date_binary(self, value: Union[date, List[date]]):
        """
        Encode de fecha binario sin periodicidad, descomponiendo en:
          - year (año absoluto)
          - abs_month (meses acumulados desde 1970-01)
          - abs_day (días acumulados desde 1970-01-01)

        Usa termómetros deterministas por componente + binding XOR con roles por componente.
        Mejora similitud local con pesos discretos (duplicando el canal abs_day).
        """

        # ---------------------------
        # 0) Validación entrada
        # ---------------------------
        is_list = isinstance(value, list)

        if is_list:
            if not value:
                return torch.zeros((0, self.dim), dtype=torch.uint8, device=self.device)
            if not all(isinstance(d, date) for d in value):
                raise TypeError("Todas las entradas deben ser objetos de tipo date")
            dates = value
        else:
            if not isinstance(value, date):
                raise TypeError("El valor debe ser un objeto date o una lista de fechas")
            dates = [value]

        # ---------------------------
        # 1) Cache config (ref + rangos + roles)
        # ---------------------------
        if not hasattr(self, "_date_ref"):
            self._date_ref = date(1970, 1, 1)

        # Rango temporal para termómetros
        # Mantengo 1900-2100 como “safe default”, pero abs_* se computa respecto a 1970.
        if not hasattr(self, "_date_min_year"):
            self._date_min_year = 1900
        if not hasattr(self, "_date_max_year"):
            self._date_max_year = 2100

        if not hasattr(self, "_date_min_abs_month"):
            # meses desde 1970-01 hasta 1900-01 (negativo)
            self._date_min_abs_month = (1900 - 1970) * 12 + (1 - 1)  # -840
        if not hasattr(self, "_date_max_abs_month"):
            # meses desde 1970-01 hasta 2100-12
            self._date_max_abs_month = (2100 - 1970) * 12 + (12 - 1)  # 1571

        if not hasattr(self, "_date_min_abs_day"):
            self._date_min_abs_day = (date(1900, 1, 1) - self._date_ref).days  # negativo
        if not hasattr(self, "_date_max_abs_day"):
            self._date_max_abs_day = (date(2100, 12, 31) - self._date_ref).days

        # Roles por componente (binarios, cacheados)
        # Si get_binary_hv(key) ya es estable/determinista, esto es suficiente.
        if not hasattr(self, "_date_role_year"):
            self._date_role_year = self.get_binary_hv("ROLE::DATE::YEAR").to(self.device)
        if not hasattr(self, "_date_role_abs_month"):
            self._date_role_abs_month = self.get_binary_hv("ROLE::DATE::ABS_MONTH").to(self.device)
        if not hasattr(self, "_date_role_abs_day"):
            self._date_role_abs_day = self.get_binary_hv("ROLE::DATE::ABS_DAY").to(self.device)

        # ---------------------------
        # 2) Helpers de componentes
        # ---------------------------
        def abs_month(d: date) -> int:
            return (d.year - 1970) * 12 + (d.month - 1)

        def abs_day(d: date) -> int:
            return (d - self._date_ref).days

        # ---------------------------
        # 3) Encode batch (termómetros por componente)
        # ---------------------------
        years = [d.year for d in dates]
        months = [abs_month(d) for d in dates]
        days = [abs_day(d) for d in dates]

        # Termómetros (cada name => permutación distinta => evita colisiones entre canales)
        year_vecs = self._thermometer_batch(
            name="date_year",
            values=years,
            min_val=self._date_min_year,
            max_val=self._date_max_year
        )

        month_vecs = self._thermometer_batch(
            name="date_abs_month",
            values=months,
            min_val=self._date_min_abs_month,
            max_val=self._date_max_abs_month
        )

        day_vecs = self._thermometer_batch(
            name="date_abs_day",
            values=days,
            min_val=self._date_min_abs_day,
            max_val=self._date_max_abs_day
        )

        # Binding XOR con roles
        # bind_batch maneja broadcast de role (1D) al batch.
        bound_year = self.bind_batch(year_vecs, self._date_role_year)
        bound_month = self.bind_batch(month_vecs, self._date_role_abs_month)
        bound_day = self.bind_batch(day_vecs, self._date_role_abs_day)

        # ---------------------------
        # 4) Bundling con pesos discretos
        # ---------------------------
        # Peso actual:
        # - abs_day domina similitud local (duplicado para mas peso)
        # - abs_month contribuye a escalas medias
        # - year contribuye a escalas largas
        #
        # Para más suavidad local aún, se pueden agregar repeticiones de bound_day.
        out = []
        for i in range(len(dates)):
            hv = self.bundle_hv([bound_year[i], bound_month[i], bound_day[i], bound_day[i]])
            out.append(hv)

        batch_result = torch.stack(out)  # (n, dim)

        # ---------------------------
        # 5) Retorno
        # ---------------------------
        if is_list:
            return batch_result
        return batch_result[0]

    def encode_person_binary(self, raw_person: Dict[str, Any]) -> torch.Tensor:
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
                # Convertir a tensor si no lo es ya
                if not isinstance(encoded_value, torch.Tensor):
                    encoded_value = torch.tensor(encoded_value, dtype=torch.uint8, device=self.device)
                else:
                    encoded_value = encoded_value.to(self.device)

                key_hv = self.get_binary_hv(key.upper())
                bound_hv = self.bind_hv(key_hv, encoded_value)
                all_field_vectors.append(bound_hv)

        if not all_field_vectors:
            return torch.zeros(self.dim, dtype=torch.uint8, device=self.device)

        return self.bundle_hv(all_field_vectors)

    def encode_batch(self, people: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Codifica un lote de registros de personas en una sola operación.
        Optimizada para procesamiento en paralelo con PyTorch.

        Args:
            people: Lista de diccionarios con datos de personas

        Returns:
            Tensor de forma [len(people), dim] con los vectores codificados
        """
        if not people:
            return torch.zeros((0, self.dim), dtype=torch.uint8, device=self.device)

        # Normalizar todos los datos de personas
        normalized_people = [normalize_person_data(p) for p in people]
        batch_size = len(normalized_people)

        # Identificar todas las claves posibles (unión)
        all_keys = set()
        for p in normalized_people:
            all_keys.update(p.keys())
        sorted_keys = sorted(list(all_keys))

        # Inicializar acumulador de vectores
        all_field_vectors = []

        # Perfilador para determinar tipos de datos
        profiler = DataTypeProfiler()

        # Procesar por columnas (características)
        for key in sorted_keys:
            # Extraer valores de esta característica para todos los registros
            column_values = [p.get(key) for p in normalized_people]

            # Saltar si todos son valores vacíos
            if all(v is None or (isinstance(v, (str, list, dict)) and not v) for v in column_values):
                continue

            # Encontrar primer valor no vacío para determinar tipo
            first_valid = next(
                (v for v in column_values if not (v is None or (isinstance(v, (str, list, dict)) and not v))), None)
            if first_valid is None:
                continue

            # Determinar tipo de dato
            profiler.profile_record({key: first_valid})
            data_type = profiler.get_type(key)

            # Caso especial: procesamiento por lotes para fechas
            if data_type == "DATE" and all(hasattr(v, 'year') for v in column_values if v is not None):
                date_values = [v if v is not None else date(1900, 1, 1) for v in column_values]
                encoded_dates = self.encode_date_binary(date_values)

                # Vincular con la clave
                key_hv = self.get_binary_hv(key.upper())
                key_hv_expanded = key_hv.unsqueeze(0).expand(batch_size, -1)
                bound_vectors = self.bind_batch(encoded_dates, key_hv_expanded)

                all_field_vectors.append(bound_vectors)
                continue

            # Procesamiento secuencial para otros tipos
            encoded_batch = []
            for idx, value in enumerate(column_values):
                if value is None or (isinstance(value, (str, list, dict)) and not value):
                    encoded_batch.append(torch.zeros(self.dim, dtype=torch.uint8, device=self.device))
                    continue

                strategy = self.strategy_factory.get_strategy(key, value, data_type)
                encoded_value = strategy.encode(key, value, profiler)

                if encoded_value is not None:
                    # Convertir a tensor si no lo es ya
                    if not isinstance(encoded_value, torch.Tensor):
                        encoded_value = torch.tensor(encoded_value, dtype=torch.uint8, device=self.device)
                    else:
                        encoded_value = encoded_value.to(self.device)

                    key_hv = self.get_binary_hv(key.upper())
                    bound_hv = self.bind_hv(key_hv, encoded_value)
                    encoded_batch.append(bound_hv)
                else:
                    encoded_batch.append(torch.zeros(self.dim, dtype=torch.uint8, device=self.device))

            # Apilar vectores para esta característica
            if encoded_batch:
                stacked_batch = torch.stack(encoded_batch)
                all_field_vectors.append(stacked_batch)

        # Si no hay vectores, devolver ceros
        if not all_field_vectors:
            return torch.zeros((batch_size, self.dim), dtype=torch.uint8, device=self.device)

        # Sumar todos los vectores de características para cada registro
        # Apilar en 3D: [n_features, batch_size, dim]
        stacked_features = torch.stack(all_field_vectors)

        # Transponer para obtener [batch_size, n_features, dim]
        transposed = stacked_features.permute(1, 0, 2)

        # Hacer bundle por registro
        return self.bundle_batch(transposed)