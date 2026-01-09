import torch
from encoding_methods.by_data_type.numbers import IntegerEncoding

class BoolEncoding:
    """
    Codificación booleana basada en IntegerEncoding con PyTorch.
    """

    def __init__(self, D=10000, flips_per_step=5, seed=0):
        """
        D: dimensión del hipervector
        flips_per_step: cantidad de bits a voltear al pasar de 0 → 1 (y viceversa)
        seed: semilla para reproducibilidad
        """
        self._enc = IntegerEncoding(
            D=D,
            flips_per_step=flips_per_step,
            n0=0,
            seed=seed,
            max_steps=1,  # sólo necesitamos representar 0 y 1
        )

    def _to_int01(self, value) -> int:
        """Normaliza el valor a 0/1. Acepta bool o int en {0,1}."""
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, int) and value in (0, 1):
            return int(value)
        raise ValueError("BoolEncoding solo acepta valores en {False, True} o {0, 1}.")

    def encode(self, value):
        """Devuelve el hipervector para False/0 o True/1."""
        return self._enc.encode(self._to_int01(value))

    def similarity(self, a, b) -> float:
        """Similitud coseno entre dos valores booleanos (o 0/1)."""
        return self._enc.similarity(self._to_int01(a), self._to_int01(b))

    def false(self):
        return torch.tensor(0)

    def true(self):
        return torch.tensor(1)

    def enc_false(self):
        """Hipervector canónico para False."""
        return self._enc.encode(0)

    def enc_true(self):
        """Hipervector canónico para True."""
        return self._enc.encode(1)