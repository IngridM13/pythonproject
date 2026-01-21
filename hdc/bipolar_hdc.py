import datetime
import torch
import numpy as np
import hashlib
from datetime import date
from typing import Optional, Dict, Any, Iterable, List, Union

# Asumiendo que estas dependencias existen en tu proyecto
from configs.settings import HDC_DIM, DEFAULT_SEED
from hdc.datatype_profiler import DataTypeProfiler
from utils.person_data_normalization import normalize_person_data
from hdc.bipolar_encoding_strategy import (
    DefaultEncodingStrategy,
    DateEncodingStrategy,
    ListEncodingStrategy,
    AttrsEncodingStrategy,
    BipolarEncodingStrategyFactory
)


class HyperDimensionalComputingBipolar:
    """
    Bipolar HDC with vectors in {-1, +1}.
    Unified version supporting both basic operations and advanced encoding strategies.
    """

    def __init__(self, dim: int = HDC_DIM, seed: Optional[int] = DEFAULT_SEED, device: Optional[str] = None):
        self.dim = dim
        self.seed = seed
        # Configuración de dispositivo (Prioriza CUDA)
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Generador para reproducibilidad
        self.rng = torch.Generator(device='cpu')  # El generador de semillas suele estar en CPU
        if seed is not None:
            self.rng.manual_seed(seed)

        self._hv_cache: Dict[str, torch.Tensor] = {}
        self._date_thresholds: Optional[torch.Tensor] = None
        self._max_range_days: int = 365 * 200

        # Inicializar el factory de estrategias
        self.strategy_factory = BipolarEncodingStrategyFactory(self)
        self.register_default_strategies()

    def register_default_strategies(self):
        """Registra las estrategias de codificación predeterminadas."""
        factory = self.strategy_factory
        factory.register_strategy("DATE", DateEncodingStrategy)
        factory.register_strategy("ATTRS_DICT", AttrsEncodingStrategy)
        factory.register_strategy("LIST_OF_STR", ListEncodingStrategy)
        factory.register_strategy("CATEGORICAL_STR", DefaultEncodingStrategy)
        factory.register_strategy("TEXT_NAME", DefaultEncodingStrategy)
        factory.register_strategy("PHONE_STR", DefaultEncodingStrategy)

    # ---- Generación ----
    def generate_random_hdv(self, n: int = 1) -> torch.Tensor:
        """Genera 1 o n vectores bipolares {-1, +1} usando Torch."""
        # Generar bits aleatorios 0, 1
        shape = (n, self.dim) if n > 1 else (self.dim,)
        rand_bits = torch.randint(0, 2, shape, generator=self.rng, device='cpu').to(torch.int8)
        # Transformar 0 -> -1
        hdv = torch.where(rand_bits == 0, torch.tensor(-1, dtype=torch.int8), torch.tensor(1, dtype=torch.int8))
        return hdv.to(self.device)

    # ---- Operaciones Core (Compatibilidad Archivo 1 y 2) ----
    def bind_hv(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Binding bipolar: producto elemento a elemento (equivalente a XOR)."""
        return (x.to(self.device) * y.to(self.device)).to(dtype=torch.int8)

    def elementwise_product_hv(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Alias de bind_hv para compatibilidad con Archivo 1."""
        return self.bind_hv(x, y)

    def bundle_init(self) -> torch.Tensor:
        """Crea un acumulador int32 para bundling."""
        return torch.zeros(self.dim, dtype=torch.int32, device=self.device)

    def bundle_add(self, acc: torch.Tensor, *vectors: torch.Tensor,
                   weights: Optional[Iterable[int]] = None) -> torch.Tensor:
        """Suma vectores en un acumulador."""
        if weights is None:
            weights = [1] * len(vectors)
        for v, w in zip(vectors, weights):
            if w != 0:
                acc.add_(v.to(self.device, dtype=torch.int32) * int(w))
        return acc

    def bundle_finalize(self, acc: torch.Tensor, tie_key: Optional[str] = None) -> torch.Tensor:
        """Aplica la función de signo; resuelve empates de forma determinista."""
        res = torch.sign(acc).to(dtype=torch.int8)
        zeros = (res == 0)
        if torch.any(zeros):
            tb = self._tie_breaker_bipolar(tie_key or "tb", self.dim).to(device=acc.device)
            res[zeros] = tb[zeros]
        return res

    def add_hv(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Suma y recorta (clip) para mantener valores en [-1, 1]. (Archivo 1)"""
        return torch.clamp(x.to(self.device) + y.to(self.device), -1, 1)

    def xor_hv(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Operación XOR lógica mapeada a bipolar. (Archivo 1)"""
        # En bipolar, XOR es equivalente a la multiplicación
        return self.bind_hv(x, y)

    def dot_product_hv(self, x: torch.Tensor, y: torch.Tensor) -> int:
        """Producto escalar entre dos vectores."""
        return int(torch.sum(x.to(self.device).float() * y.to(self.device).float()).item())

    def shifting_hv(self, x: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Desplazamiento circular (Permutación)."""
        return torch.roll(x.to(self.device), shifts=k, dims=-1)

    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> Union[float, torch.Tensor]:
        """Similitud de coseno optimizada para vectores bipolares."""
        x_f = x.to(self.device).float()
        y_f = y.to(self.device).float()

        if x.ndim == 1 and y.ndim == 1:
            return float(torch.dot(x_f, y_f) / self.dim)

        # Soporte para lotes (batch)
        if x.ndim > 1 and y.ndim > 1:
            return (x_f @ y_f.T) / self.dim
        if x.ndim > 1:
            return torch.matmul(x_f, y_f) / self.dim

        return float(torch.dot(x_f, y_f) / self.dim)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normaliza el vector (Archivo 1)."""
        norm = torch.linalg.norm(x.float())
        return x if norm == 0 else x.float() / norm

    def bipolarize(self, vector: torch.Tensor) -> torch.Tensor:
        """Convierte cualquier vector a bipolar {-1, 1} (Archivo 1)."""
        return torch.where(vector >= 0,
                           torch.tensor(1, dtype=torch.int8, device=self.device),
                           torch.tensor(-1, dtype=torch.int8, device=self.device))

    def flip_inplace(self, v: torch.Tensor, idx: int) -> torch.Tensor:
        """Invierte el signo en el índice indicado (Archivo 1)."""
        v[idx] = -v[idx]
        return v

    # ---- Gestión de Identidad y Cache ----
    def get_bipolar_hv(self, key: Any) -> torch.Tensor:
        """Obtiene o genera un HV determinista para una clave."""
        key_str = str(key)
        if key_str in self._hv_cache:
            return self._hv_cache[key_str].to(self.device)

        seed = self._deterministic_hash(key_str)
        rng_gen = torch.Generator().manual_seed(seed)

        # Generación interna similar a generate_random_hdv pero con semilla específica
        rand_bits = torch.randint(0, 2, (self.dim,), generator=rng_gen, device='cpu').to(torch.int8)
        hv_tensor = torch.where(rand_bits == 0, torch.tensor(-1, dtype=torch.int8), torch.tensor(1, dtype=torch.int8))

        hv_tensor = hv_tensor.to(self.device)
        self._hv_cache[key_str] = hv_tensor
        return hv_tensor

    def _deterministic_hash(self, key_str: str) -> int:
        h = hashlib.md5(str(key_str).encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little") % (2 ** 32)

    def _tie_breaker_bipolar(self, key: str, dim: int) -> torch.Tensor:
        seed = self._deterministic_hash(f"tb:{key}")
        rng = torch.Generator().manual_seed(seed)
        rand_bits = torch.randint(0, 2, (dim,), generator=rng, device='cpu').to(torch.int8)
        return torch.where(rand_bits == 0, torch.tensor(-1, dtype=torch.int8), torch.tensor(1, dtype=torch.int8))

    # ---- Codificación Avanzada (Fechas, Personas, Batches) ----
    def encode_date_bipolar(self, date_obj: Union[date, List[date], None]) -> torch.Tensor:
        """Codificación de fechas basada en frecuencias (FPE)."""
        if date_obj is None:
            return torch.ones(self.dim, dtype=torch.int8, device=self.device)

        reference_date = date(1970, 1, 1)

        # Convertir a lista para procesar igual
        is_list = isinstance(date_obj, list)
        dates = date_obj if is_list else [date_obj]

        days_list = []
        for d in dates:
            if d is None:
                days_list.append(0)
            elif isinstance(d, (date, datetime.datetime)):
                days_list.append((d - reference_date).days)
            else:
                days_list.append(0)

        days_arr = torch.tensor(days_list, dtype=torch.float32, device=self.device).clamp(0, self._max_range_days)
        if not is_list:
            days_arr = days_arr.squeeze()
        else:
            days_arr = days_arr.unsqueeze(1)

        num_components = self.dim // 2
        min_freq, max_freq = 1.0 / self._max_range_days, 0.5
        freqs = torch.exp(torch.linspace(torch.log(torch.tensor(min_freq, device=self.device)),
                                         torch.log(torch.tensor(max_freq, device=self.device)),
                                         num_components, device=self.device))

        phases = 2 * torch.pi * freqs * (days_arr if not is_list else days_arr)
        sin_c, cos_c = torch.sin(phases), torch.cos(phases)

        if is_list:
            components = torch.zeros((len(dates), self.dim), dtype=torch.float32, device=self.device)
            components[:, 0::2] = sin_c
            components[:, 1::2] = cos_c
        else:
            components = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
            components[0::2] = sin_c
            components[1::2] = cos_c

        return torch.where(components >= 0,
                           torch.tensor(1, dtype=torch.int8, device=self.device),
                           torch.tensor(-1, dtype=torch.int8, device=self.device))

    def encode_person_generalized(self, raw_person: Dict[str, Any]) -> torch.Tensor:
        """Codifica una persona usando el sistema de estrategias."""
        bundle_acc = self.bundle_init()
        person = normalize_person_data(raw_person)
        profiler = DataTypeProfiler()
        profiler.profile_record(person)

        for key in sorted(person.keys()):
            value = person[key]
            if value in (None, "", []): continue

            data_type = profiler.get_type(key)
            strategy = self.strategy_factory.get_strategy(key, value, data_type)
            encoded_value = strategy.encode(key, value, profiler)

            if not isinstance(encoded_value, torch.Tensor):
                encoded_value = torch.from_numpy(encoded_value).to(self.device)

            key_hv = self.get_bipolar_hv(key)
            bound_hv = self.bind_hv(key_hv, encoded_value)
            self.bundle_add(bundle_acc, bound_hv)

        return self.bundle_finalize(bundle_acc, tie_key="person_bundle")

    def encode_batch(self, people: List[Dict[str, Any]]) -> torch.Tensor:
        """Codificación por lotes optimizada."""
        if not people:
            return torch.zeros((0, self.dim), dtype=torch.float32, device=self.device)

        normalized_people = [normalize_person_data(p) for p in people]
        N = len(normalized_people)
        all_keys = sorted({k for p in normalized_people for k in p.keys()})

        acc = torch.zeros((N, self.dim), dtype=torch.int32, device=self.device)
        profiler = DataTypeProfiler()

        for key in all_keys:
            values = [p.get(key) for p in normalized_people]
            if all(v in (None, "", []) for v in values): continue

            first_val = next((v for v in values if v not in (None, "", [])), None)
            profiler.profile_record({key: first_val})
            data_type = profiler.get_type(key)

            if data_type == "DATE":
                encoded_col = self.encode_date_bipolar(values)
            else:
                strategy = self.strategy_factory.get_strategy(key, first_val, data_type)
                col_hvs = []
                for v in values:
                    if v in (None, "", []):
                        col_hvs.append(torch.zeros(self.dim, dtype=torch.int8, device=self.device))
                    else:
                        ev = strategy.encode(key, v, profiler)
                        col_hvs.append(
                            torch.from_numpy(ev).to(self.device) if not isinstance(ev, torch.Tensor) else ev.to(
                                self.device))
                encoded_col = torch.stack(col_hvs)

            key_hv = self.get_bipolar_hv(key)
            acc += (encoded_col * key_hv.unsqueeze(0)).to(dtype=torch.int32)

        final_hvs = torch.sign(acc).to(dtype=torch.int8)
        zeros = (final_hvs == 0)
        if torch.any(zeros):
            tb = self._tie_breaker_bipolar("batch_encode", self.dim).to(device=acc.device)
            final_hvs[zeros] = tb[zeros]

        return final_hvs.float()
