import torch
from configs.settings import HDC_DIM

class IntegerEncoding:
    """
    Linear Mapping para números enteros.
    Idea: partir de un hipervector base H0 y aplicar flips deterministas,
    en bloques, según la distancia |n - n0|. Para diferenciar signos,
    usamos dos colas disjuntas de un mismo permutado de índices:
      - pasos positivos usan perm[0:m]
      - pasos negativos usan perm[-m:]
    Asegurar max_steps * flips_per_step <= D para evitar reusar índices.
    """
    def __init__(self, D=10000, flips_per_step=5, n0=0, seed=0, max_steps=None):
        """
        D: dimensión del hipervector
        flips_per_step (k): cantidad de bits a voltear por unidad
        n0: entero de referencia donde H(n0) = H0
        seed: semilla
        max_steps: cantidad máxima de pasos desde n0 para la que se garantiza no reusar índices.
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)

        if flips_per_step <= 0:
            raise ValueError("flips_per_step debe ser > 0")
        
        self.D = int(D)
        self.k = int(flips_per_step)
        self.n0 = int(n0)

        # Generate bipolar random tensor 
        self.H0 = torch.where(
            torch.rand(self.D) > 0.5, 
            torch.tensor(1.0), 
            torch.tensor(-1.0)
        )

        # Permutación base única
        self.perm = torch.randperm(self.D)

        # Cálculo de max steps
        if max_steps is None:
            self.max_steps = self.D // (2 * self.k)
        else:
            self.max_steps = int(max_steps)

        # Pre-cortes para positivo y negativo
        self.max_m = self.max_steps * self.k  # máximo de flips por lado

        # Sanidad
        if self.max_m * 2 > self.D:
            raise ValueError(
                "max_steps*flips_per_step*2 excede D. Aumenta D o reduce flips_per_step/max_steps."
            )

    def encode(self, n: int) -> torch.Tensor:
        """
        Devuelve un hipervector bipolar para el entero n usando PyTorch.
        """
        n = int(n)
        steps = n - self.n0
        m = abs(steps) * self.k

        if m > self.max_m:
            raise ValueError(
                f"n={n} está fuera del rango soportado sin reuso de índices. "
                f"Máximo |n - n0| = {self.max_steps} (con k={self.k}, D={self.D})."
            )

        # Copia del vector base usando torch
        v = self.H0.clone()
        
        if steps > 0:
            idx = self.perm[:m]  # lado positivo
            v[idx] *= -1
        elif steps < 0:
            idx = self.perm[self.D - m:]  # lado negativo
            v[idx] *= -1

        return v

    def similarity(self, n1: int, n2: int) -> float:
        """Similitud coseno entre H(n1) y H(n2) usando PyTorch."""
        v1 = self.encode(n1)
        v2 = self.encode(n2)

        # Cosine similarity using PyTorch
        return torch.dot(v1, v2) / self.D


class DecimalEncoding:
    """
    Fractional Power Encoding (FPE) para valores decimales/reales.
    v_i(x) = sign( cos( ω_i * (x - x0) + φ_i ) ), con ω_i y φ_i aleatorios por dimensión.
    """
    def __init__(self, D=10000, seed=0, x0=0.0, smoothness=10.0, omega_spread=4.0):
        """
        D: dimensión del hipervector
        seed: semilla
        x0: centro del contexto
        smoothness: escala típica de variación
        omega_spread: factor de dispersión de frecuencias
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)

        if smoothness <= 0:
            raise ValueError("smoothness debe ser > 0")
        if omega_spread < 1.0:
            raise ValueError("omega_spread debe ser >= 1.0")

        self.D = int(D)
        self.x0 = float(x0)

        # Frecuencia base
        omega_base = torch.pi / smoothness

        # Distribuimos ω_i log-uniformemente
        log_min = torch.log(omega_base / omega_spread)
        log_max = torch.log(omega_base)

        self.omega = torch.exp(torch.empty(self.D).uniform_(log_min, log_max))
        self.phi = torch.empty(self.D).uniform_(0, 2*torch.pi)

    def encode(self, x: float) -> torch.Tensor:
        """
        Devuelve un hipervector bipolar para el real x.
        """
        x = float(x)
        arg = self.omega * (x - self.x0) + self.phi
        v = torch.sign(torch.cos(arg))
        
        # Reemplazar 0 por +1 para estabilidad
        v[v == 0] = 1
        return v.float()

    def similarity(self, x1: float, x2: float) -> float:
        """Similitud coseno entre H(x1) y H(x2)."""
        v1 = self.encode(x1)
        v2 = self.encode(x2)

        return torch.dot(v1, v2) / self.D


if __name__ == "__main__":
    # Integer Encoding testing
    int_enc = IntegerEncoding(D=HDC_DIM, flips_per_step=5, n0=0, seed=42)
    h0 = int_enc.encode(0)
    h1 = int_enc.encode(1)
    h2 = int_enc.encode(2)
    print("cos(H0,H1) =", int_enc.similarity(0, 1))
    print("cos(H0,H2) =", int_enc.similarity(0, 2))
    print("cos(H1,H2) =", int_enc.similarity(1, 2))

    # Decimal Encoding testing
    dec_enc = DecimalEncoding(D=HDC_DIM, seed=123, x0=0.0, smoothness=10.0, omega_spread=4.0)
    a = dec_enc.encode(2.5)
    b = dec_enc.encode(2.6)
    c = dec_enc.encode(9.5)
    print("cos(2.5,2.6) =", dec_enc.similarity(2.5, 2.6))
    print("cos(2.5,9.5) =", dec_enc.similarity(2.5, 9.5))