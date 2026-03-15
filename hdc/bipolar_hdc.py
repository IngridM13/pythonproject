import datetime
import torch
import hashlib
from datetime import date
from typing import Optional, Dict, Any, Iterable, List, Union


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
        """
        Encoding bipolar de fechas SIN periodicidad, consistente con la versión binaria:
        - Descompone en year, abs_month, abs_day (absolutos desde 1970-01-01)
        - Termómetro determinista por componente (permutación por nombre)
        - Binding con roles (multiplicación)
        - Bundling con pesos discretos (abs_day duplicado)
        """
        if date_obj is None:
            return torch.ones(self.dim, dtype=torch.int8, device=self.device)

        ref = date(1970, 1, 1)

        is_list = isinstance(date_obj, list)
        dates = date_obj if is_list else [date_obj]

        # Roles cacheados
        if not hasattr(self, "_date_role_year"):
            self._date_role_year = self.get_bipolar_hv("ROLE::DATE::YEAR")
            self._date_role_abs_month = self.get_bipolar_hv("ROLE::DATE::ABS_MONTH")
            self._date_role_abs_day = self.get_bipolar_hv("ROLE::DATE::ABS_DAY")

        # Rangos (ajustables según dataset)
        # Mantengo 1900-2100 como seguro, pero abs_* es relativo a 1970.
        years_min, years_max = 1900, 2100
        min_abs_month = (1900 - 1970) * 12 + (1 - 1)  # -840
        max_abs_month = (2100 - 1970) * 12 + (12 - 1)  # 1571
        min_abs_day = (date(1900, 1, 1) - ref).days
        max_abs_day = (date(2100, 12, 31) - ref).days

        def to_abs_month(d: date) -> int:
            return (d.year - 1970) * 12 + (d.month - 1)

        def to_abs_day(d: date) -> int:
            return (d - ref).days

        # Extraer valores (aceptar datetime.date y datetime.datetime)
        years, abs_months, abs_days = [], [], []
        for d in dates:
            if d is None:
                years.append(years_min)  # o 0, pero mejor clamp al rango
                abs_months.append(0)
                abs_days.append(0)
                continue
            if isinstance(d, (date, datetime.datetime)):
                dd = d.date() if isinstance(d, datetime.datetime) else d
                years.append(dd.year)
                abs_months.append(to_abs_month(dd))
                abs_days.append(to_abs_day(dd))
            else:
                years.append(years_min)
                abs_months.append(0)
                abs_days.append(0)

        # Termómetro bipolar determinista (usando hashing interno de esta clase)
        # y conversión a bipolar {-1, +1}
        def thermometer_bipolar_batch(name: str, vals: List[int], vmin: int, vmax: int) -> torch.Tensor:
            seed = self._deterministic_hash(name)
            rng = torch.Generator(device="cpu").manual_seed(seed)
            perm = torch.randperm(self.dim, generator=rng, device="cpu").to(self.device)

            vals_t = torch.tensor(vals, dtype=torch.int32, device=self.device)
            vals_t = torch.clamp(vals_t, vmin, vmax)
            denom = float(vmax - vmin) if vmax > vmin else 1.0
            prop = (vals_t - vmin).float() / denom
            nbits = (prop * self.dim).to(torch.int32)

            out = torch.zeros((len(vals), self.dim), dtype=torch.int8, device=self.device)
            # Set bits por fila (loop; optimizable luego)
            for i, n in enumerate(nbits.tolist()):
                if n > 0:
                    out[i, perm[:n]] = 1

            return (out * 2 - 1).to(torch.int8)

        # Batch encode componentes
        year_vecs = thermometer_bipolar_batch("date_year", years, years_min, years_max)
        month_vecs = thermometer_bipolar_batch("date_abs_month", abs_months, min_abs_month, max_abs_month)
        day_vecs = thermometer_bipolar_batch("date_abs_day", abs_days, min_abs_day, max_abs_day)

        # Binding bipolar (multiplicación)
        bound_year = self.bind_hv(year_vecs, self._date_role_year.unsqueeze(0).expand(len(dates), -1))
        bound_month = self.bind_hv(month_vecs, self._date_role_abs_month.unsqueeze(0).expand(len(dates), -1))
        bound_day = self.bind_hv(day_vecs, self._date_role_abs_day.unsqueeze(0).expand(len(dates), -1))

        # Bundling con pesos discretos: abs_day duplicado para similitud local
        acc = torch.zeros((len(dates), self.dim), dtype=torch.int32, device=self.device)
        acc += bound_year.to(torch.int32)
        acc += bound_month.to(torch.int32)
        acc += bound_day.to(torch.int32)
        acc += bound_day.to(torch.int32)  # peso x2

        res = torch.sign(acc).to(torch.int8)
        zeros = (res == 0)
        if torch.any(zeros):
            tb = self._tie_breaker_bipolar("date_bundle", self.dim).to(device=self.device)
            res[zeros] = tb.unsqueeze(0).expand(len(dates), -1)[zeros]

        return res if is_list else res[0]

    def encode_person_generalized(
        self,
        raw_person: Dict[str, Any],
        field_weights: Optional[Dict[str, int]] = None,
        excluded_fields: Optional[set] = None,
    ) -> torch.Tensor:
        """Codifica una persona usando el sistema de estrategias.

        Parameters
        ----------
        raw_person : dict
            Raw person data to encode.
        field_weights : dict, optional
            Mapping of field name -> repetition count. Each bound field vector is
            added to the accumulator ``weight`` times. Fields absent from the dict
            default to weight 1. When None, all fields have weight 1.
        excluded_fields : set, optional
            Field names to skip entirely (not bound, not bundled). When None, no
            fields are skipped.
        """
        bundle_acc = self.bundle_init()
        person = normalize_person_data(raw_person)
        profiler = DataTypeProfiler()
        profiler.profile_record(person)

        for key in sorted(person.keys()):
            # Skip excluded fields
            if excluded_fields is not None and key in excluded_fields:
                continue

            value = person[key]
            if value in (None, "", []): continue

            data_type = profiler.get_type(key)
            strategy = self.strategy_factory.get_strategy(key, value, data_type)
            encoded_value = strategy.encode(key, value, profiler)

            if not isinstance(encoded_value, torch.Tensor):
                encoded_value = torch.from_numpy(encoded_value).to(self.device)

            key_hv = self.get_bipolar_hv(key)
            bound_hv = self.bind_hv(key_hv, encoded_value)

            # Determine repetition weight (default 1)
            weight = 1
            if field_weights is not None:
                weight = field_weights.get(key, 1)

            self.bundle_add(bundle_acc, bound_hv, weights=[weight])

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
