import numpy as np
from configs.settings import HDC_DIM
from hdc.hdc_common_operations import bipolar_random, flip_inplace
from sklearn.metrics.pairwise import cosine_similarity
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
                   Si None, se toma floor(D / (2*flips_per_step)) para separar positivo/negativo.
        """
        if flips_per_step <= 0:
            raise ValueError("flips_per_step debe ser > 0")
        self.D = int(D)
        self.k = int(flips_per_step)
        self.n0 = int(n0)
        self.rng = np.random.default_rng(seed)

        # Extraer esto como parámetro? -> Bernie
        self.H0 = bipolar_random(self.D, self.rng)

        # Permutación base única
        self.perm = self.rng.permutation(self.D)

        # Si no se indica, garantizamos que no se reusan índices ni en + ni en -
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

    def encode(self, n: int) -> np.ndarray:
        """
        Devuelve un hipervector bipolar para el entero n.
        Complejidad O(D) en el peor caso (por copia) + O(m) flips.
        """
        n = int(n)
        steps = n - self.n0
        m = abs(steps) * self.k

        if m > self.max_m:
            raise ValueError(
                f"n={n} está fuera del rango soportado sin reuso de índices. "
                f"Máximo |n - n0| = {self.max_steps} (con k={self.k}, D={self.D})."
            )

        v = self.H0.copy()
        if steps > 0:
            idx = self.perm[:m]  # lado positivo
            flip_inplace(v, idx)
        elif steps < 0:
            idx = self.perm[self.D - m:]  # lado negativo
            flip_inplace(v, idx)
        # steps == 0 -> H0

        return v

    def similarity(self, n1: int, n2: int) -> float:
        """Similitud coseno entre H(n1) y H(n2)."""

        # 1. Obtener los vectores 1D (shape (D,))
        v1 = self.encode(n1)
        v2 = self.encode(n2)

        # 2. Pasarlos como listas de vectores (sklearn los verá como 2D)
        #    [v1] tiene shape (1, D)
        #    [v2] tiene shape (1, D)
        sim_matrix = cosine_similarity([v1], [v2])

        # 3. Extraer el resultado (que es una matriz 1x1, ej: [[1.0]])
        return sim_matrix[0][0]

class DecimalEncoding:
    """
    Fractional Power Encoding (FPE) para valores decimales/reales.
    v_i(x) = sign( cos( ω_i * (x - x0) + φ_i ) ), con ω_i y φ_i aleatorios por dimensión.
    - 'smoothness' controla cuán “suave” es el campo: a mayor smoothness, menos flips por unidad.
    - x0 fija el centro del contexto (puede ser, por ej., el promedio esperado).
    """
    def __init__(self, D=10000, seed=0, x0=0.0, smoothness=10.0, omega_spread=4.0):
        """
        D: dimensión del hipervector
        seed: semilla
        x0: centro del contexto; valores cercanos a x0 tienden a compartir más bits
        smoothness (en unidades de x): escala típica de variación; ~ cuántas unidades
                   hay que mover x para que muchos bits cambien. (↑smoothness → ↓ω)
        omega_spread: factor de dispersión de frecuencias (>=1). Usa ω en [ω_base/omega_spread, ω_base].
        """
        if smoothness <= 0:
            raise ValueError("smoothness debe ser > 0")
        if omega_spread < 1.0:
            raise ValueError("omega_spread debe ser >= 1.0")

        self.D = int(D)
        self.rng = np.random.default_rng(seed)
        self.x0 = float(x0)

        # Frecuencia base: elegimos que ~π rad por 'smoothness' unidades
        # (aprox. medio periodo en 'smoothness', conservador para mantener suavidad).
        omega_base = np.pi / smoothness

        # Distribuimos ω_i log-uniformemente en [omega_base/omega_spread, omega_base]
        log_min = np.log(omega_base / omega_spread)
        log_max = np.log(omega_base)
        self.omega = np.exp(self.rng.uniform(log_min, log_max, size=self.D))

        # Fase aleatoria por dimensión en [0, 2π)
        self.phi = self.rng.uniform(0, 2*np.pi, size=self.D)

    def encode(self, x: float) -> np.ndarray:
        """
        Devuelve un hipervector bipolar para el real x.
        Preserva cercanía: |x1 - x2| pequeño ⇒ alta similitud esperada.
        """
        x = float(x)
        arg = self.omega * (x - self.x0) + self.phi
        v = np.sign(np.cos(arg))
        # Por estabilidad, reemplazamos 0 por +1
        v[v == 0] = 1
        return v.astype(int)

    def similarity(self, x1: float, x2: float) -> float:
        """Similitud coseno entre H(x1) y H(x2)."""
        return cosine_similarity(self.encode(x1), self.encode(x2))


if __name__ == "__main__":
    # Testing de esto
    # Integer -> Con Linear Mapping Encoding
    int_enc = IntegerEncoding(D=HDC_DIM, flips_per_step=5, n0=0, seed=42)
    h0 = int_enc.encode(0)
    h1 = int_enc.encode(1)
    h2 = int_enc.encode(2)
    print("cos(H0,H1) =", int_enc.similarity(0, 1))
    print("cos(H0,H2) =", int_enc.similarity(0, 2))
    print("cos(H1,H2) =", int_enc.similarity(1, 2))

    # Decimal -> Con Fractional Power Encoding
    dec_enc = DecimalEncoding(D=HDC_DIM, seed=123, x0=0.0, smoothness=10.0, omega_spread=4.0)
    a = dec_enc.encode(2.5)
    b = dec_enc.encode(2.6)
    c = dec_enc.encode(9.5)
    print("cos(2.5,2.6) =", dec_enc.similarity(2.5, 2.6))
    print("cos(2.5,9.5) =", dec_enc.similarity(2.5, 9.5))