import datetime

import numpy as np
import torch
from datetime import date
from configs.settings import HDC_DIM, DEFAULT_SEED
from hdc.hdc_common_operations import (
    bipolar_random, dot_product,
    elementwise_product, shifting
)
from typing import Optional, Dict, Any, Iterable, List, Union
import hashlib
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
    """Bipolar HDC with vectors in {-1,+1} (dtype=int8)."""

    def __init__(self, dim: int = HDC_DIM, seed: Optional[int] = DEFAULT_SEED):
        self.dim = dim
        self.seed = seed
        # Use torch Generator for native torch RNG
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
            
        self._hv_cache: Dict[str, torch.Tensor] = {}
        self._date_thresholds: Optional[torch.Tensor] = None
        self._max_range_days: int = 365 * 200

        # Inicializar el factory de estrategias (nuevo)
        self.strategy_factory = BipolarEncodingStrategyFactory(self)
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
    def generate_random_hdv(self, n: int = 1) -> torch.Tensor:
        """Generate 1 or n random bipolar vectors {-1,+1}, dtype=int8 (Torch Tensor)."""
        if n == 1:
            return bipolar_random(self.dim, self.rng)
            
        # Generate (n, dim)
        # Check if we can generate in batch directly or stack
        # Using stack for consistency with current RNG usage
        return torch.stack([bipolar_random(self.dim, self.rng) for _ in range(n)], dim=0)

    # ---- Core ops ----
    def bind_hv(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Bipolar binding (XOR-equivalent): element-wise product. Returns torch.Tensor."""
        # Ensure inputs are tensors
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)
            
        # Element-wise multiplication
        return (x * y).to(dtype=torch.int8)

    def bundle_init(self) -> torch.Tensor:
        """Create an int32 accumulator for bundling (Torch Tensor)."""
        return torch.zeros(self.dim, dtype=torch.int32)

    def bundle_add(self, acc: torch.Tensor, *vectors: Iterable[Union[np.ndarray, torch.Tensor]],
                   weights: Optional[Iterable[int]] = None) -> torch.Tensor:
        """Add one or more bipolar vectors into an int32 accumulator."""
        if weights is None:
            weights = [1] * len(vectors)
            
        for v, w in zip(vectors, weights):
            if w != 0:
                if not isinstance(v, torch.Tensor):
                    v = torch.from_numpy(v)
                
                # acc = acc + w * v
                acc.add_(v.to(dtype=torch.int32) * int(w))
        return acc

    def bundle_finalize(self, acc: torch.Tensor, tie_key: Optional[str] = None) -> torch.Tensor:
        """Sign of accumulator; break ties deterministically if present."""
        # torch.sign returns -1, 0, 1
        res = torch.sign(acc).to(dtype=torch.int8)
        
        zeros = (res == 0)
        if torch.any(zeros):
            # Generate tie breaker
            tb = self._tie_breaker_bipolar(tie_key or "tb", self.dim).to(device=acc.device)
            res[zeros] = tb[zeros]
            
        return res

    def dot_product_hv(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> int:
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        if isinstance(y, np.ndarray): y = torch.from_numpy(y)
        # Use torch.dot only works for 1D. For generalized, use sum(x*y)
        return int(torch.sum(x.float() * y.float()).item())

    def cosine_similarity(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Optimized Cosine similarity for bipolar {-1,+1} vectors.
        """
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        if isinstance(y, np.ndarray): y = torch.from_numpy(y)
        
        x_float = x.float()
        y_float = y.float()

        if x.ndim == 1 and y.ndim == 1:
            return float(torch.dot(x_float, y_float) / self.dim)

        # Batch usage: (N, D) @ (M, D).T -> (N, M)
        # If both are batches
        if x.ndim > 1 and y.ndim > 1:
             sim = (x_float @ y_float.T) / self.dim
             return sim
             
        # Broadcast logic handled by matmul usually, but let's be safe
        # (N, D) @ (D,) -> (N,)
        if x.ndim > 1:
            sim = torch.matmul(x_float, y_float) / self.dim
            return sim
            
        return float(torch.dot(x_float, y_float) / self.dim)


    def shifting_hv(self, x: Union[np.ndarray, torch.Tensor], k: int = 1) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return torch.roll(x, shifts=k, dims=-1)

    # ... (otros métodos como normalize, bipolarize, flip_vector_at) ...

    # ---- Deterministic HVs ----
    def get_bipolar_hv(self, key: Any) -> torch.Tensor:
        """Reproducible bipolar HV for a key; uses internal cache."""
        key_str = str(key)

        if key_str in self._hv_cache:
            return self._hv_cache[key_str]

        seed = self._deterministic_hash(key_str)
        
        # Use torch generator for deterministic creation
        rng_gen = torch.Generator().manual_seed(seed)
        
        # Generate as tensor
        hv_tensor = bipolar_random(self.dim, rng=rng_gen)
        
        self._hv_cache[key_str] = hv_tensor
        return hv_tensor

    # ---- Internals ----
    def _deterministic_hash(self, key_str: str) -> int:
        h = hashlib.md5(str(key_str).encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little") % (2 ** 32)

    def _tie_breaker_bipolar(self, key: str, dim: int) -> torch.Tensor:
        seed = self._deterministic_hash(f"tb:{key}")
        rng = torch.Generator().manual_seed(seed)
        # Generate random {0, 1} -> convert to {-1, 1}
        # bipolar_random does exactly this
        return bipolar_random(dim, rng=rng)

    # ------------------------------------------------------------------
    # 1) Inicialización de thresholds para fechas
    # ------------------------------------------------------------------
    def _init_date_thresholds(self):
        """Inicializa thresholds para el encoding escalar de fechas (Torch)."""
        if self._date_thresholds is not None:
            return

        # Usar umbrales espaciados uniformemente
        self._date_thresholds = torch.linspace(
            start=0,
            end=self._max_range_days,
            steps=self.dim,
            dtype=torch.int32
        )

    # ------------------------------------------------------------------
    # 2) Nuevo encode_date_bipolar ESCALAR (VECTORIZADO - Torch)
    # ------------------------------------------------------------------
    def encode_date_bipolar(self, date_obj: Union[date, List[date], None]) -> torch.Tensor:
        """
        Enhanced frequency-based encoding for dates using sine/cosine patterns.
        This implements a Fractional Power Encoding (FPE) approach for better discrimination
        between dates at different temporal distances.

        Supports batching: if date_obj is a list, returns tensor of shape (N, D).
        """
        # Case: Single None
        if date_obj is None:
            return torch.ones(self.dim, dtype=torch.int8)

        # Use reference date
        reference_date = date(1970, 1, 1)

        # Check for batch list
        if isinstance(date_obj, list):
            # Optimized batch processing
            days_list = []
            for d in date_obj:
                if d is None:
                    days_list.append(0)
                elif isinstance(d, (date, datetime)):
                    days_list.append((d - reference_date).days)
                else:
                    days_list.append(0)

            # Convert to tensor
            days_arr = torch.tensor(days_list, dtype=torch.float32)  # (N,)

            # Clip range (to avoid extremely large values)
            days_arr = torch.clamp(days_arr, 0, self._max_range_days)

            # Reshape for broadcasting
            days_arr = days_arr.unsqueeze(1)  # Shape: (N, 1)

            # Generate frequency components - distribute across dimension
            # Use log-spaced frequencies for better discrimination across time ranges
            num_components = self.dim // 2  # We need pairs of sin/cos

            # Log-spaced frequencies (smaller for distant dates, larger for close dates)
            min_freq = 1.0 / self._max_range_days  # Lowest frequency (cycles per day)
            max_freq = 0.5  # Highest frequency (cycles per day - Nyquist limit)

            # Generate frequencies with more components for recent dates
            freqs = torch.exp(torch.linspace(np.log(min_freq), np.log(max_freq), num_components))

            # Reshape for broadcasting - (1, num_components)
            freqs = freqs.unsqueeze(0)

            # Compute phase angles: 2π * freq * days
            # Broadcasting: (N, 1) * (1, num_components) -> (N, num_components)
            phases = 2 * np.pi * freqs * days_arr

            # Generate sin and cos components
            sin_components = torch.sin(phases)
            cos_components = torch.cos(phases)

            # Interleave sin and cos components to form the complete vector
            # Shape: (N, dim)
            components = torch.zeros((days_arr.shape[0], self.dim), dtype=torch.float32)
            components[:, 0::2] = sin_components
            components[:, 1::2] = cos_components

            # Convert to bipolar {-1, 1}
            return torch.where(components >= 0,
                               torch.tensor(1, dtype=torch.int8),
                               torch.tensor(-1, dtype=torch.int8))

        # Case: Single date
        days_since_reference = (date_obj - reference_date).days

        # Acotar al rango soportado
        t = max(0, min(days_since_reference, self._max_range_days))

        # Generate frequency components
        num_components = self.dim // 2

        # Log-spaced frequencies
        min_freq = 1.0 / self._max_range_days
        max_freq = 0.5
        freqs = torch.exp(torch.linspace(np.log(min_freq), np.log(max_freq), num_components))

        # Compute phase angles
        phases = 2 * np.pi * freqs * t

        # Generate sin and cos components
        sin_components = torch.sin(phases)
        cos_components = torch.cos(phases)

        # Interleave sin and cos components
        components = torch.zeros(self.dim, dtype=torch.float32)
        components[0::2] = sin_components
        components[1::2] = cos_components

        # Convert to bipolar
        return torch.where(components >= 0,
                           torch.tensor(1, dtype=torch.int8),
                           torch.tensor(-1, dtype=torch.int8))

    # Método legacy: redirecciona al método generalizado
    def encode_person_bipolar_datatype(self, raw_person: Dict[str, Any]) -> torch.Tensor:
        """
        Método legacy.
        """
        print("ADVERTENCIA: Usando método deprecated. Considere migrar a encode_person_generalized")
        return self.encode_person_generalized(raw_person)

    # Nuevo método generalizado basado en estrategias
    def encode_person_generalized(self, raw_person: Dict[str, Any]) -> torch.Tensor:
        """
        Codifica los datos de una persona utilizando estrategias basadas en tipos de datos.
        Returns: torch.Tensor
        """
        bundle_acc = self.bundle_init()
        person = normalize_person_data(raw_person)
        profiler = DataTypeProfiler()
        profiler.profile_record(person)

        for key in sorted(person.keys()):
            value = person[key]

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

            # Ensure encoded_value is a tensor
            if not isinstance(encoded_value, torch.Tensor):
                encoded_value = torch.from_numpy(encoded_value)

            # Vincular clave y valor codificado
            key_hv = self.get_bipolar_hv(key)
            bound_hv = self.bind_hv(key_hv, encoded_value)
            self.bundle_add(bundle_acc, bound_hv)

        # Devolver el vector final
        return self.bundle_finalize(bundle_acc, tie_key="person_bundle")

    def encode_batch(self, people: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Batched encoding with vectorization for supported types (e.g., Dates).
        Performs column-wise processing for speedup.
        
        Returns:
            torch.Tensor (N, D)
        """
        if not people:
             return torch.zeros((0, self.dim), dtype=torch.float32)

        # 1. Normalize all inputs
        normalized_people = [normalize_person_data(p) for p in people]
        N = len(normalized_people)
        
        # 2. Identify all possible keys (union of keys)
        all_keys = set()
        for p in normalized_people:
            all_keys.update(p.keys())
        sorted_keys = sorted(list(all_keys))
        
        # 3. Accumulator (N, D) - Using Torch
        acc = torch.zeros((N, self.dim), dtype=torch.int32)
        
        # 4. Process Column-wise
        profiler = DataTypeProfiler()
        
        for key in sorted_keys:
            # Extract column values
            values = [p.get(key) for p in normalized_people]
            
            # Skip if column is effectively empty
            if all(v in (None, "", []) for v in values):
                continue
                
            # Determine Strategy based on the first non-empty value
            first_val = next((v for v in values if v not in (None, "", [])), None)
            
            profiler.profile_record({key: first_val})
            data_type = profiler.get_type(key)
            
            encoded_col = None
            
            # --- Vectorized Paths ---
            if data_type == "DATE":
                # Use our new vectorized date encoder
                encoded_col = self.encode_date_bipolar(values) # Returns Tensor (N, D)
                
            # --- Iterative Paths (Fallback) ---
            else:
                strategy = self.strategy_factory.get_strategy(key, first_val, data_type)
                
                # Build (N, D) tensor iteratively
                col_hvs = []
                for v in values:
                    if v in (None, "", []):
                        col_hvs.append(torch.zeros(self.dim, dtype=torch.int8))
                    else:
                        ev = strategy.encode(key, v, profiler)
                        if not isinstance(ev, torch.Tensor):
                            ev = torch.from_numpy(ev)
                        col_hvs.append(ev)
                        
                encoded_col = torch.stack(col_hvs)
            
            # --- Binding ---
            # Get Role HV for this key
            key_hv = self.get_bipolar_hv(key) # Tensor (D,)
            
            # Broadcast Binding: (N, D) * (1, D) -> (N, D)
            bound_col = encoded_col * key_hv.unsqueeze(0)
            
            # --- Accumulation ---
            acc += bound_col.to(dtype=torch.int32)
            
        # 5. Finalize (Sign)
        final_hvs = torch.where(acc >= 0, torch.tensor(1, dtype=torch.int8), torch.tensor(-1, dtype=torch.int8))
        
        return final_hvs.float()