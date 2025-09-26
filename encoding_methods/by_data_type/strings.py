import numpy as np
from hdc_encoding.utils import bipolar_random, cosine_similarity, binary_random


class StringEncoding:
    """
    Codificación hiperdimensional para strings:
    - Carácter → hipervector (HV) aleatorio (bipolar {-1,+1}).
    - Palabra  → combinación por permutación por posición + binding (XOR para binario, multiplicación para bipolar).
    - Texto    → superposición de n-grams con Random Indexing (RI). BEAGLE puede mapearse igual como baseline.

    Semejanza:
    - Bipolar: similitud coseno.
    - Binario: similitud por coincidencia de bits (1 - Hamming normalizada).
    """

    def __init__(
            self,
            D: int = 10000,
            seed: int = 0,
            mode: str = "bipolar",          # "bipolar" o "binary"
            ngram_n: int = 3,               # tamaño de n-gram para textos
            strategy: str = "RI",           # "RI" o "BEAGLE" (BEAGLE -> alias simple al mismo esquema aquí)
    ):
        if mode not in ("bipolar", "binary"):
            raise ValueError("mode debe ser 'bipolar' o 'binary'.")
        if ngram_n <= 0:
            raise ValueError("ngram_n debe ser > 0.")
        self.D = int(D)
        self.mode = mode
        self.ngram_n = int(ngram_n)
        self.strategy = strategy.upper()
        self.rng = np.random.default_rng(seed)

        # Permutación base para posiciones (aplicar k veces = desplazar posición k)
        self._perm = self.rng.permutation(self.D)

        # Diccionarios de base HVs
        self._char_table: dict[str, np.ndarray] = {}
        self._word_cache: dict[str, np.ndarray] = {}

        # Identidades de binding
        if self.mode == "bipolar":
            self._bind_identity = np.ones(self.D, dtype=int)  # identidad de multiplicación
        else:
            self._bind_identity = np.zeros(self.D, dtype=np.uint8)  # identidad de XOR

    # ----------------------------
    # Generación y operaciones base
    # ----------------------------
    def _random_hv(self) -> np.ndarray:
        if self.mode == "bipolar":
            return bipolar_random(self.D, self.rng).astype(int)
        else:
            # Vector binario {0,1}
            return binary_random(self.D, self.rng)

    def _permute(self, v: np.ndarray, k: int = 1) -> np.ndarray:
        """Aplica la permutación base k veces."""
        if k == 0:
            return v
        k = int(k)
        out = v
        for _ in range(k):
            out = out[self._perm]
        return out

    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Binding: XOR si binario, multiplicación si bipolar."""
        if self.mode == "bipolar":
            return a * b
        else:
            # XOR por bit
            return np.bitwise_xor(a, b)

    def _superpose(self, vecs: list[np.ndarray]) -> np.ndarray:
        """Superposición y colapso al espacio del modo."""
        if not vecs:
            # vector neutro
            return self._bind_identity.copy()
        if self.mode == "bipolar":
            acc = np.sum(vecs, axis=0, dtype=int)
            # colapso a {-1,+1}
            out = np.sign(acc)
            out[out == 0] = 1
            return out.astype(int)
        else:
            # mayoría bit a bit (ties → 1)
            acc = np.sum(vecs, axis=0, dtype=int)
            threshold = (len(vecs) + 1) // 2
            return (acc >= threshold).astype(np.uint8)

    # ----------------------------
    # Unidades de codificación
    # ----------------------------
    def encode_char(self, c: str) -> np.ndarray:
        """HV para un carácter concreto (con memoización)."""
        if len(c) != 1:
            raise ValueError("encode_char espera un único carácter.")
        if c not in self._char_table:
            self._char_table[c] = self._random_hv()
        return self._char_table[c]

    def encode_word(self, word: str) -> np.ndarray:
        """
        Palabra = binding de caracteres permutados por su posición.
        H(word) = bind( Π^0 H(c0), Π^1 H(c1), ..., Π^(L-1) H(c_{L-1}) )
        """
        if word in self._word_cache:
            return self._word_cache[word]
        if not word:
            # Palabra vacía → identidad de binding
            hv = self._bind_identity.copy()
            self._word_cache[word] = hv
            return hv

        hv = None
        for i, ch in enumerate(word):
            ch_hv = self._permute(self.encode_char(ch), i)
            hv = ch_hv if hv is None else self._bind(hv, ch_hv)

        self._word_cache[word] = hv
        return hv

    def encode_tokens_ngram(self, tokens: list[str], n: int | None = None) -> np.ndarray:
        """
        Texto como superposición de n-grams de palabras con binding+perm por posición.
        v(ngram) = bind( Π^0 H(w_i), Π^1 H(w_{i+1}), ..., Π^(n-1) H(w_{i+n-1}) )
        Resultado final = superposición de todos los n-grams y colapso al modo.
        """
        if n is None:
            n = self.ngram_n
        n = int(n)
        if n <= 0:
            raise ValueError("n debe ser > 0.")

        if len(tokens) < n:
            # Si no hay n-gram posible, degradar a superposición simple de las palabras
            return self._superpose([self.encode_word(t) for t in tokens])

        ngram_vecs = []
        for i in range(len(tokens) - n + 1):
            hv = None
            for j in range(n):
                wv = self._permute(self.encode_word(tokens[i + j]), j)
                hv = wv if hv is None else self._bind(hv, wv)
            ngram_vecs.append(hv)

        return self._superpose(ngram_vecs)

    def encode_text(self, text: str | list[str], n: int | None = None, strategy: str | None = None) -> np.ndarray:
        """
        Codifica un texto completo.
        - Si text es str, se tokeniza por espacios.
        - strategy:
            * "RI"     → n-grams por binding+perm y superposición (por defecto).
            * "BEAGLE" → baseline: mismo proceso (colocar contexto con n-grams).
        """
        if isinstance(text, str):
            tokens = [t for t in text.split() if t]
        else:
            tokens = list(text)

        if not tokens:
            # texto vacío
            return self._bind_identity.copy()

        use_strategy = (strategy or self.strategy).upper()
        if use_strategy not in ("RI", "BEAGLE"):
            raise ValueError("strategy debe ser 'RI' o 'BEAGLE'.")

        # Para este baseline, RI y BEAGLE comparten el mismo esquema de n-grams.
        return self.encode_tokens_ngram(tokens, n=n)

    # ----------------------------
    # Métricas
    # ----------------------------
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Similitud entre dos HVs.
        - Bipolar: coseno.
        - Binario: fracción de bits iguales = 1 - Hamming_normalizada.
        """
        if self.mode == "bipolar":
            return cosine_similarity(a, b)
        else:
            a = np.asarray(a).astype(np.uint8)
            b = np.asarray(b).astype(np.uint8)
            if a.shape != b.shape:
                raise ValueError("Los vectores deben tener la misma dimensión.")
            eq = np.sum(a == b)
            return float(eq) / float(a.size)

    def encode(self, s: str) -> np.ndarray:
        """Alias: si no hay espacios → palabra; si los hay → texto con n-grams."""
        if " " in s:
            return self.encode_text(s)
        return self.encode_word(s)

