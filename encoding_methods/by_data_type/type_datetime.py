import numpy as np
from typing import Iterable, Mapping, Any
from configs.settings import HDC_DIM, DEFAULT_SEED
from encoding_methods.by_data_type.numbers import DecimalEncoding
from hdc.ops_bipolar import HyperDimensionalComputingBipolar



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
        self.rng = np.random.default_rng(seed)
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
        self._role_hv: dict[str, np.ndarray] = {}
        hdc = HyperDimensionalComputingBipolar()

        for name in self.components:
            role_seed = self._stable_hash_seed(name, base=int(self.rng.integers(0, 2**32 - 1)))
            role_rng = np.random.default_rng(role_seed)
            self._role_hv[name] = hdc.generate_random_hdv(self.D, role_rng).astype(int)

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

    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Binding bipolar: multiplicación elemento a elemento."""
        return (a * b).astype(int)

    def _superpose(self, vecs: list[np.ndarray]) -> np.ndarray:
        """Superposición con colapso a {-1,+1}."""
        if not vecs:
            return np.ones(self.D, dtype=int)
        acc = np.sum(vecs, axis=0, dtype=int)
        out = np.sign(acc)
        out[out == 0] = 1
        return out.astype(int)

    # ----------------------------
    # Extracción de componentes
    # ----------------------------
    @staticmethod
    def _get_component_value(obj: Any, name: str) -> float | None:
        """
        Extrae el valor del componente 'name' desde distintos tipos:
        - datetime.datetime / datetime.date
        - dict-like con claves coincidentes
        Devuelve None si no se puede obtener.
        """
        # Lazy import para no forzar dependencia si no se usa
        import datetime as _dt

        # dict-like
        if isinstance(obj, Mapping):
            if name in obj:
                return float(obj[name])

        # datetime/date
        if isinstance(obj, (_dt.datetime, _dt.date)):
            if name == "year":
                return float(obj.year)
            if name == "month":
                return float(obj.month)
            if name == "day":
                return float(obj.day)
            if name == "hour" and isinstance(obj, _dt.datetime):
                return float(obj.hour)
            if name == "minute" and isinstance(obj, _dt.datetime):
                return float(obj.minute)
            if name == "second" and isinstance(obj, _dt.datetime):
                return float(obj.second)
            if name == "weekday":
                # Monday=0..Sunday=6
                return float(obj.weekday())
            if name == "yearday":
                # 1..366
                return float(obj.timetuple().tm_yday)

        # Tuplas (año, mes, día[, hora[, minuto[, segundo]]])
        if isinstance(obj, (tuple, list)):
            mapping = {}
            if len(obj) >= 1:
                mapping["year"] = obj[0]
            if len(obj) >= 2:
                mapping["month"] = obj[1]
            if len(obj) >= 3:
                mapping["day"] = obj[2]
            if len(obj) >= 4:
                mapping["hour"] = obj[3]
            if len(obj) >= 5:
                mapping["minute"] = obj[4]
            if len(obj) >= 6:
                mapping["second"] = obj[5]
            if name in mapping:
                return float(mapping[name])

        return None

    # ----------------------------
    # API pública
    # ----------------------------
    def encode_components(self, values: Mapping[str, float]) -> np.ndarray:
        """
        Codifica un dict de componentes explícitos: {"year": 2024, "month": 3, ...}
        Ignora componentes no presentes en self.components.
        """
        parts = []
        for name in self.components:
            if name not in values:
                continue
            v = float(values[name])
            role = self._role_hv[name]
            val_hv = self._fpe[name].encode(v)
            parts.append(self._bind(role, val_hv))
        return self._superpose(parts)

    def encode(self, obj: Any) -> np.ndarray:
        """
        Codifica objetos de fecha/hora o estructuras equivalentes:
        - datetime.datetime, datetime.date
        - dict con claves de componentes
        - tuplas/listas (año, mes, día[, hora[, minuto[, segundo]]])
        """
        parts = []
        for name in self.components:
            val = self._get_component_value(obj, name)
            if val is None:
                continue
            role = self._role_hv[name]
            val_hv = self._fpe[name].encode(val)
            parts.append(self._bind(role, val_hv))
        return self._superpose(parts)

    def similarity(self, a: Any, b: Any) -> float:
        """Similitud coseno entre dos fechas/horas (o representaciones compatibles)."""
        hdc = HyperDimensionalComputingBipolar()
        va = self.encode(a)
        vb = self.encode(b)
        return hdc.cosine_similarity(va, vb)

    # Accesores útiles
    def role_vector(self, component: str) -> np.ndarray:
        return self._role_hv[component]

    def value_encoder(self, component: str) -> DecimalEncoding:
        return self._fpe[component]

    def components_present(self) -> tuple[str, ...]:
        return self.components


if __name__ == "__main__":
    import datetime as dt
    from configs.settings import HDC_DIM, SEED

    print("Ejemplo: DatetimeEncoding")
    enc = DatetimeEncoding(
        D=HDC_DIM,
        seed=SEED,
        components=("year", "month", "day", "hour"),
    )


    # Fechas de prueba
    t_ref = dt.datetime(2024, 3, 15, 10, 0, 0)   # 15/03/2024 10:00
    t_next_month = dt.datetime(2024, 4, 15, 10, 0, 0)  # mes siguiente
    t_far = dt.datetime(2024, 9, 15, 10, 0, 0)   # varios meses después

    # Codificación
    h_ref = enc.encode(t_ref)
    h_next = enc.encode(t_next_month)
    h_far = enc.encode(t_far)

    # Similitudes (coseno)
    hdc = HyperDimensionalComputingBipolar()
    print("Dimensión HV:", h_ref.shape[0])
    print("sim(ref, ref)         =", hdc.cosine_similarity(h_ref, h_ref))
    print("sim(ref, mes+1)       =", hdc.cosine_similarity(h_ref, h_next))
    print("sim(ref, mes lejano)  =", hdc.cosine_similarity(h_ref, h_far))

    # También puedes codificar con diccionarios o tuplas
    # Diccionario explícito de componentes
    h_dict = enc.encode({"year": 2024, "month": 3, "day": 15, "hour": 10})
    print("sim(ref, dict mismo)  =", hdc.cosine_similarity(h_ref, h_dict))

    # Tupla (año, mes, día, hora)
    h_tuple = enc.encode((2024, 3, 15, 10))
    print("sim(ref, tupla misma) =", hdc.cosine_similarity(h_ref, h_tuple))

# Me parece que no está quedando bien. No es lo mismo una distancia de 1 dia que de 1 mes.