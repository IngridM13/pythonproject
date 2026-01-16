import torch
from typing import Iterable, Mapping, Any
from configs.settings import HDC_DIM, DEFAULT_SEED
from encoding_methods.by_data_type.numbers import DecimalEncoding
# Corrected import path
from hdc.hdc_common_operations import bipolar_random


class DatetimeEncoding:
    """
    Codificación hiperdimensional para fecha/hora.

    Premisa:
    - Descomponer en componentes (año, mes, día, hora, ...).
    - Representar cada componente como entero con FPE (DecimalEncoding).
    - Usar binding para unir rol (p.ej. "mes") con valor (p.ej. 3).
    - Vector: bipolar {-1,+1}.
    - Distancia/similitud: coseno.

    Diseño:
    H(dt) = sign( sum_c bind( R_c , FPE_c(value_c) ) )
      donde:
        - R_c: hipervector de rol para el componente c (bipolar aleatorio estable).
        - FPE_c: un DecimalEncoding por componente c (permite consultas analógicas).
        - bind: multiplicación elemento a elemento (bipolar).
        - sum: superposición y luego colapso con sign (0 -> +1).

    Componentes soportados por defecto: ("year", "month", "day", "hour").
    """

    def __init__(
            self,
            D: int = HDC_DIM,
            seed: int = DEFAULT_SEED,
            components: Iterable[str] = ("year", "month", "day", "hour"),
            # Parámetros FPE por componente (puedes ajustar smoothness/x0 según dominio)
            fpe_params: Mapping[str, Mapping[str, Any]] | None = None,
            omega_spread: float = 4.0,
    ):
        """
        D: dimensión del hipervector.
        seed: semilla global (estable para roles y FPEs).
        components: iterables de nombres de componentes a usar.
        fpe_params: dict opcional con overrides por componente:
            {
              "year":  {"x0": 2000.0, "smoothness": 10.0},
              "month": {"x0": 6.5,    "smoothness": 1.0},
              "day":   {"x0": 15.5,   "smoothness": 1.0},
              "hour":  {"x0": 12.0,   "smoothness": 1.0},
              ...
            }
        omega_spread: dispersión de frecuencias para todos los FPEs.
        """
        self.D = int(D)
        self.rng = torch.Generator().manual_seed(seed)
        self.components = tuple(components)

        # Valores por defecto razonables para FPE por componente
        default_fpe_params = {
            "year": {"x0": 2000.0, "smoothness": 10.0},
            "month": {"x0": 6.5, "smoothness": 1.0},
            "day": {"x0": 15.5, "smoothness": 1.0},
            "hour": {"x0": 12.0, "smoothness": 1.0},
            "minute": {"x0": 30.0, "smoothness": 1.0},
            "second": {"x0": 30.0, "smoothness": 1.0},
            "weekday": {"x0": 3.0, "smoothness": 1.0},  # 0..6
            "yearday": {"x0": 182.5, "smoothness": 5.0},  # 1..366
        }
        if fpe_params:
            # Aplicar overrides
            for k, v in fpe_params.items():
                default_fpe_params.setdefault(k, {})
                default_fpe_params[k] = {**default_fpe_params[k], **dict(v)}

        # Hipervectores de rol (bipolares, aleatorios y estables por nombre)
        # Generamos una semilla derivada por rol para estabilidad pero sin colisiones obvias
        self._role_hv: dict[str, torch.Tensor] = {}

        for name in self.components:
            role_seed = self._stable_hash_seed(name, base=int(torch.randint(0, 2**32 - 1, (1,), generator=self.rng).item()))
            role_rng = torch.Generator().manual_seed(role_seed)
            
            # Use direct deterministic generation with the role_rng
            raw_hv = bipolar_random(self.D, rng=role_rng)
            
            self._role_hv[name] = raw_hv.float() # Use float for compatibility with FPE results

        # Un FPE dedicado por componente (permite distintas escalas)
        self._fpe: dict[str, DecimalEncoding] = {}
        for name in self.components:
            p = default_fpe_params.get(name, {"x0": 0.0, "smoothness": 1.0})
            # Derivar semilla separada para el FPE del componente
            fpe_seed = self._stable_hash_seed(f"fpe::{name}", base=seed)
            self._fpe[name] = DecimalEncoding(
                D=self.D,
                seed=fpe_seed,
                x0=float(p.get("x0", 0.0)),
                smoothness=float(p.get("smoothness", 1.0)),
                omega_spread=float(omega_spread),
            )

    # ----------------------------
    # Utilidades internas
    # ----------------------------
    @staticmethod
    def _stable_hash_seed(key: str, base: int = 0) -> int:
        """Convierte una cadena en una semilla estable (32 bits) combinada con 'base'."""
        h = 2166136261  # FNV-1a base
        for ch in key:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return int((h ^ (base & 0xFFFFFFFF)) & 0xFFFFFFFF)

    def _bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Binding bipolar: multiplicación elemento a elemento."""
        return a * b

    def _superpose(self, vecs: list[torch.Tensor]) -> torch.Tensor:
        """Superposición con colapso a {-1,+1}."""
        if not vecs:
            return torch.ones(self.D, dtype=torch.float32)
        
        # Stack efficient accumulation
        stack = torch.stack(vecs)
        acc = torch.sum(stack, dim=0)
        
        # Sign operation (preserve 0 as 1 for stability in bipolar contexts)
        out = torch.sign(acc)
        out[out == 0] = 1.0
        return out