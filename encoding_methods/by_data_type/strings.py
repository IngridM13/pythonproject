import torch
from hdc.bipolar_hdc import HyperDimensionalComputingBipolar
from typing import List, Union

# Create a helper instance for cosine similarity
_hdc_ops = HyperDimensionalComputingBipolar()


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
            mode: str = "bipolar",  # "bipolar" o "binary"
            ngram_n: int = 3,  # tamaño de n-gram para textos
            strategy: str = "RI",  # "RI" o "BEAGLE"
    ):
        if mode not in ("bipolar", "binary"):
            raise ValueError("mode debe ser 'bipolar' o 'binary'.")
        if ngram_n <= 0:
            raise ValueError("ngram_n debe ser > 0.")
        self.D = int(D)
        self.mode = mode
        self.ngram_n = int(ngram_n)
        self.strategy = strategy.upper()
        
        # Set device (prefer CUDA if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use a generator on CPU for compatibility
        self.rng = torch.Generator().manual_seed(seed)

        # Permutación base para posiciones - store as torch tensor
        perm_tensor = torch.randperm(self.D, generator=self.rng)
        self._perm = perm_tensor  # Keep as tensor
        
        # Store inverted permutation for faster computation of multiple shifts
        self._perm_inv = torch.argsort(self._perm)
        
        # Pre-compute common permutations (for positions 0-10)
        self._perm_cache = {}
        for i in range(11):  # 0-10
            self._perm_cache[i] = self._compute_permutation_tensor(i)

        # Diccionarios de base HVs
        self._char_table: dict[str, torch.Tensor] = {}
        self._word_cache: dict[str, torch.Tensor] = {}

        # Identidades de binding - store as torch tensor
        if self.mode == "bipolar":
            self._bind_identity = torch.ones(self.D, dtype=torch.int8, device=self.device)
        else:
            self._bind_identity = torch.zeros(self.D, dtype=torch.uint8, device=self.device)

    # Add this new method to compute permutation tensors
    def _compute_permutation_tensor(self, k: int) -> torch.Tensor:
        """
        Compute the permutation tensor for k shifts.
        This returns indices to be used with torch.index_select
        """
        if k == 0:
            return torch.arange(self.D, device=self.device)
        
        indices = torch.arange(self.D, device=self.device)
        
        # Apply permutation k times
        for _ in range(k):
            indices = torch.index_select(self._perm.to(self.device), 0, indices)
        
        return indices

    # Replace the permute method with a vectorized version

    def _permute(self, v: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Vectorized permutation using PyTorch.
        Works with both single vectors and batches of vectors.

        Args:
            v: Vector or batch of vectors to permute (PyTorch tensor)
            k: Number of shifts to apply

        Returns:
            Permuted vector(s) as a PyTorch tensor
        """
        if k == 0:
            return v

        # Ensure tensor is on the correct device
        v_tensor = v.to(self.device)

        # Get permutation indices
        if k in self._perm_cache:
            indices = self._perm_cache[k]
        else:
            indices = self._compute_permutation_tensor(k)
            # Cache if it's a reasonable size
            if k < 100:
                self._perm_cache[k] = indices

        # Apply permutation based on dimensionality
        if v_tensor.ndim == 1:
            # Single vector
            result = torch.index_select(v_tensor, 0, indices)
        else:
            # Batch of vectors
            # For batch tensors of shape (batch_size, D), we need to gather
            # Use fancy indexing: v_tensor[:, indices]
            result = v_tensor[:, indices]

        return result

    # ----------------------------
    # Generación y operaciones base
    # ----------------------------

    def _random_hv(self) -> torch.Tensor:
        """
        Genera un vector hiperdimensional aleatorio usando exclusivamente PyTorch.

        Returns:
            torch.Tensor: Un tensor de PyTorch con valores bipolar {-1, 1} o binario {0, 1}
                          según el modo configurado.

        Note:
            Esta versión rompe deliberadamente la compatibilidad con NumPy para
            forzar la migración completa a PyTorch.
        """
        if self.mode == "bipolar":
            # Generar vector bipolar directamente con PyTorch
            shape = (self.D,)
            rand_bits = torch.randint(0, 2, shape, generator=self.rng, device=self.device)
            # Transformar 0 -> -1, 1 -> 1
            hv = torch.where(rand_bits == 0,
                             torch.tensor(-1, dtype=torch.int8, device=self.device),
                             torch.tensor(1, dtype=torch.int8, device=self.device))
            return hv
        else:
            # Generar vector binario directamente con PyTorch
            hv = torch.randint(0, 2, (self.D,),
                               generator=self.rng,
                               device=self.device,
                               dtype=torch.uint8)
            return hv


    def _bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Binds two PyTorch tensors together using multiplication (for bipolar mode)
        or XOR (for binary mode).

        Args:
            a: First tensor (PyTorch tensor)
            b: Second tensor (PyTorch tensor)

        Returns:
            Result of binding operation as a PyTorch tensor
        """
        if self.mode == "bipolar":
            return a * b
        else:
            return torch.bitwise_xor(a, b)

    def _superpose(self, vecs: list[torch.Tensor]) -> torch.Tensor:
        """
        Realiza la superposición de vectores usando PyTorch.
        Para modo bipolar: realiza suma y colapsa a {-1,+1} usando sign.
        Para modo binario: realiza suma y aplica umbral mayoritario.

        Args:
            vecs: Lista de tensores PyTorch a superponer

        Returns:
            Tensor PyTorch con el resultado de la superposición
        """
        if not vecs:
            return self._bind_identity.clone()

        # Apilar los vectores eficientemente
        stack = torch.stack(vecs).to(self.device)

        # Realizar la suma a lo largo del eje 0 (a través de los vectores)
        acc = torch.sum(stack, dim=0)

        if self.mode == "bipolar":
            # Aplicar la función sign y resolver los ceros como 1
            out = torch.sign(acc)
            out[out == 0] = 1
            return out.to(torch.int8)
        else:
            # Aplicar umbral mayoritario para modo binario
            threshold = (len(vecs) + 1) // 2
            return (acc >= threshold).to(torch.uint8)

    def _superpose_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Vectorized superposition for a batch of vectors.
        
        Args:
            vectors: Tensor of shape (N, D) where N is the number of vectors
            
        Returns:
            Tensor of shape (D) representing the superposition
        """
        if vectors.shape[0] == 0:
            return self._bind_identity.clone().to(self.device)
            
        if self.mode == "bipolar":
            # Sum along batch dimension
            acc = torch.sum(vectors, dim=0)
            # Apply sign function with 1 as tie-breaker
            result = torch.sign(acc)
            result[result == 0] = 1
            return result.to(torch.int8)
        else:
            # For binary mode
            acc = torch.sum(vectors, dim=0)
            threshold = (vectors.shape[0] + 1) // 2
            return (acc >= threshold).to(torch.uint8)

    def _batch_permute_and_bind(self, vectors: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Performs permutation and binding on a batch of vectors.
        
        Args:
            vectors: Tensor of shape (batch_size, D)
            positions: Tensor of shape (batch_size,) containing positions
            
        Returns:
            A single vector representing the binding of all permuted vectors
        """
        batch_size = vectors.shape[0]
        permuted_vectors = vectors.clone()
        
        # Apply different permutations to each vector based on position
        for i in range(batch_size):
            pos = positions[i].item()
            permuted_vectors[i] = self._permute(vectors[i], pos)
        
        # Bind all vectors together
        if self.mode == "bipolar":
            # Multiply all vectors together
            bound = torch.prod(permuted_vectors, dim=0)
        else:
            # XOR all vectors together
            bound = vectors[0]
            for i in range(1, batch_size):
                bound = torch.bitwise_xor(bound, permuted_vectors[i])
        
        return bound

    # ----------------------------
    # Unidades de codificación
    # ----------------------------

    def encode_char(self, c: str) -> torch.Tensor:
        """
        Codifica un único carácter a un vector hiperdimensional.

        Args:
            c: Un único carácter a codificar

        Returns:
            Tensor PyTorch representando el hipervector del carácter

        Raises:
            ValueError: Si se proporciona más de un carácter
        """
        if len(c) != 1:
            raise ValueError("encode_char espera un único carácter.")

        # Verificar si el carácter ya está en la tabla
        if c not in self._char_table:
            # Generar un nuevo vector aleatorio para este carácter
            self._char_table[c] = self._random_hv()

        # Asegurar que el vector está en el dispositivo correcto
        char_vector = self._char_table[c].to(self.device)

        return char_vector

    def encode_word(self, word: str) -> torch.Tensor:
        """
        Codifica una palabra como un hipervector mediante la combinación de los
        hipervectores de sus caracteres con permutaciones según posición.

        Args:
            word: Palabra a codificar

        Returns:
            Tensor de PyTorch con la representación hiperdimensional de la palabra
        """
        # Verificar caché primero
        if word in self._word_cache:
            cached_hv = self._word_cache[word]
            # Si está en caché, asegurarse que esté en el dispositivo correcto
            return cached_hv.to(self.device)

        # Caso de palabra vacía
        if not word:
            hv = self._bind_identity.clone()
            self._word_cache[word] = hv
            return hv

        # Codificar la palabra caracter por caracter
        hv = None
        for i, ch in enumerate(word):
            # Codificar el caracter y aplicar permutación según posición
            ch_hv = self._permute(self.encode_char(ch), i)

            # Combinar con el vector acumulado hasta el momento
            if hv is None:
                hv = ch_hv
            else:
                hv = self._bind(hv, ch_hv)

        # Guardar en caché para uso futuro
        self._word_cache[word] = hv

        return hv

    def encode_tokens_ngram(self, tokens: list[str], n: int | None = None) -> torch.Tensor:
        """
        Codifica una lista de tokens usando técnica de n-gramas.
        Genera hipervectores para secuencias de n tokens consecutivos y los superpone.

        Args:
            tokens: Lista de tokens (palabras) a codificar
            n: Tamaño del n-grama. Si es None, usa el valor configurado en self.ngram_n

        Returns:
            Tensor de PyTorch que representa el hipervector de los n-gramas
        """
        # Configurar tamaño de n-grama
        if n is None:
            n = self.ngram_n
        n = int(n)

        # Caso especial: menos tokens que el tamaño del n-grama
        if len(tokens) < n:
            # Superponemos los vectores de cada token individual
            return self._superpose([self.encode_word(t) for t in tokens])

        # Codificación mediante n-gramas
        ngram_vecs = []
        for i in range(len(tokens) - n + 1):
            hv = None
            # Para cada n-grama, codificamos cada token con permutación según posición
            for j in range(n):
                # Codificar palabra y aplicar permutación según posición relativa
                wv = self._permute(self.encode_word(tokens[i + j]), j)

                # Acumular mediante binding
                if hv is None:
                    hv = wv
                else:
                    hv = self._bind(hv, wv)

            # Añadir el hipervector del n-grama a la lista
            ngram_vecs.append(hv)

        # Superponer todos los n-gramas para obtener la representación final
        return self._superpose(ngram_vecs)

    def encode_text(self, text: str | list[str], n: int | None = None, strategy: str | None = None) -> torch.Tensor:
        """
        Codifica un texto o lista de tokens como un hipervector utilizando PyTorch.

        Args:
            text: Texto (string) o lista de tokens a codificar
            n: Tamaño del n-grama a utilizar. Si es None, usa el valor configurado (self.ngram_n)
            strategy: Estrategia de codificación (actualmente no utilizada)

        Returns:
            Tensor de PyTorch que representa el texto codificado

        Nota:
            Si se proporciona un string, se dividirá en tokens por espacios.
            Si se proporciona una lista, se usará directamente como tokens.
        """
        # Determinar los tokens a partir del input
        if isinstance(text, str):
            # Si es una cadena, dividir por espacios y filtrar tokens vacíos
            tokens = [t for t in text.split() if t]
        else:
            # Si ya es una lista, usarla directamente
            tokens = list(text)

        # Caso especial: sin tokens
        if not tokens:
            return self._bind_identity.clone()

        # Aplicar codificación n-grama
        return self.encode_tokens_ngram(tokens, n=n)

    # ----------------------------
    # Métricas y Batch
    # ----------------------------

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """
        Calcula la similitud entre dos tensores PyTorch.

        Args:
            a: Primer tensor PyTorch
            b: Segundo tensor PyTorch

        Returns:
            Similitud como valor float entre 0 y 1
        """
        # Asegurarse que los vectores estén en el dispositivo correcto
        a = a.to(self.device)
        b = b.to(self.device)

        if self.mode == "bipolar":
            # Para vectores bipolares, usar similitud coseno optimizada
            # a y b deben convertirse a float para operaciones de punto flotante
            a_f = a.float()
            b_f = b.float()
            return float(torch.dot(a_f, b_f) / self.D)
        else:
            # Para vectores binarios, calcular coincidencia de bits
            # (1 - distancia Hamming normalizada)
            eq = torch.sum(a == b).float()
            return float(eq / a.numel())

    def batch_similarity(self, query: torch.Tensor, corpus: torch.Tensor) -> torch.Tensor:
        """
        Calculate similarities between a query vector and a corpus of vectors,
        or between all pairs of vectors in two corpuses.
        
        Args:
            query: Tensor of shape (D) or (N, D)
            corpus: Tensor of shape (M, D)
            
        Returns:
            If query is 1D: Tensor of shape (M) with similarities to each corpus vector
            If query is 2D: Tensor of shape (N, M) with all pairwise similarities
        """
        # Ensure tensors are on the correct device
        query = query.to(self.device)
        corpus = corpus.to(self.device)
        
        if self.mode == "bipolar":
            # For bipolar vectors, use optimized cosine similarity
            
            # Normalize the dimensions
            if query.ndim == 1:
                # Single query vector (D) vs corpus (M, D)
                # Compute dot product and normalize
                dots = torch.matmul(corpus, query.float())
                return dots / self.D
            else:
                # Batch queries (N, D) vs corpus (M, D)
                # Compute all pairwise similarities: (N, M)
                dots = torch.matmul(query.float(), corpus.float().t())
                return dots / self.D
        else:
            # For binary vectors, use bit agreement (1 - normalized Hamming)
            if query.ndim == 1:
                # Compute element-wise equality and mean
                matches = (corpus == query.unsqueeze(0)).float()
                return torch.mean(matches, dim=1)
            else:
                # Need to compare each query with each corpus vector
                # This is more complex for binary case
                N = query.shape[0]
                M = corpus.shape[0]
                similarities = torch.zeros((N, M), dtype=torch.float32, device=self.device)
                
                for i in range(N):
                    matches = (corpus == query[i].unsqueeze(0)).float()
                    similarities[i] = torch.mean(matches, dim=1)
                    
                return similarities

    def encode(self, s: str) -> torch.Tensor:
        """
        Codifica una cadena como vector hiperdimensional, detectando automáticamente
        si es una palabra o un texto completo.

        Args:
            s: Cadena a codificar

        Returns:
            Tensor PyTorch con la representación hiperdimensional
        """
        if " " in s:
            return self.encode_text(s)  # Asumiendo que t() es equivalente a encode_text()
        return self.encode_word(s)

    def encode_batch(self, strings: List[str]) -> torch.Tensor:
        """
        Batch encode multiple strings at once.
        Returns a tensor of shape (N, D)
        """
        if not strings:
            return torch.zeros((0, self.D), dtype=torch.float32)

        # Process in parallel based on whether strings contain spaces
        word_indices = [i for i, s in enumerate(strings) if " " not in s]
        text_indices = [i for i, s in enumerate(strings) if " " in s]

        results = torch.zeros((len(strings), self.D),
                              dtype=torch.int8 if self.mode == "bipolar" else torch.uint8)

        if word_indices:
            word_strings = [strings[i] for i in word_indices]
            word_results = self._batch_encode_words(word_strings)
            for idx, result_idx in enumerate(word_indices):
                results[result_idx] = word_results[idx]

        if text_indices:
            text_strings = [strings[i] for i in text_indices]
            text_results = self._batch_encode_texts(text_strings)
            for idx, result_idx in enumerate(text_indices):
                results[result_idx] = text_results[idx]

        return results

    def encode_batch_mixed(self, items: List[str]) -> torch.Tensor:
        """
        Enhanced batch encoding that handles a mix of words and texts efficiently.
        
        Args:
            items: List of strings (can be single words or texts with spaces)
            
        Returns:
            Tensor of shape (N, D) with encoded vectors
        """
        # This is a more descriptive alias for encode_batch
        return self.encode_batch(items)

    # Update _batch_encode_words to use the improved permutation

    def _batch_encode_words(self, words: List[str]) -> torch.Tensor:
        """
        Vectorized encoding of multiple words at once using PyTorch exclusively.

        Args:
            words: List of words to encode

        Returns:
            Tensor of shape (len(words), D) containing encoded word vectors
        """
        batch_size = len(words)

        if batch_size == 0:
            return torch.zeros((0, self.D),
                               dtype=torch.int8 if self.mode == "bipolar" else torch.uint8,
                               device=self.device)

        # Check cache first
        cached_indices = []
        uncached_indices = []
        uncached_words = []

        for i, word in enumerate(words):
            if word in self._word_cache:
                cached_indices.append(i)
            else:
                uncached_indices.append(i)
                uncached_words.append(word)

        result = torch.zeros((batch_size, self.D),
                             dtype=torch.int8 if self.mode == "bipolar" else torch.uint8,
                             device=self.device)

        # Fill in cached results
        for i, orig_idx in enumerate(cached_indices):
            word = words[orig_idx]
            # Ensure cache entry is a PyTorch tensor
            if not isinstance(self._word_cache[word], torch.Tensor):
                # Convert numpy array to tensor if necessary
                self._word_cache[word] = torch.tensor(self._word_cache[word], device=self.device)

            # Add to result
            result[orig_idx] = self._word_cache[word].to(self.device)

        # Process uncached words
        if uncached_words:
            all_chars = set()
            for word in uncached_words:
                all_chars.update(word)

            for c in all_chars:
                if c not in self._char_table:
                    self._char_table[c] = self._random_hv()
                elif not isinstance(self._char_table[c], torch.Tensor):
                    # Convert any remaining NumPy arrays in char_table
                    self._char_table[c] = torch.tensor(self._char_table[c], device=self.device)

            for i, word_idx in enumerate(uncached_indices):
                word = words[word_idx]

                if not word:
                    word_hv = self._bind_identity.clone()
                    result[word_idx] = word_hv
                    # Store in cache as PyTorch tensor
                    self._word_cache[word] = word_hv
                    continue

                word_vector = None
                for pos, char in enumerate(word):
                    # Ensure char vector is a PyTorch tensor
                    char_hv = self._char_table[char]
                    if not isinstance(char_hv, torch.Tensor):
                        # Convert any remaining NumPy arrays
                        char_hv = torch.tensor(char_hv, device=self.device)
                        self._char_table[char] = char_hv

                    # Make sure it's on the right device
                    char_tensor = char_hv.to(self.device)

                    # Use vectorized permutation
                    permuted_tensor = self._permute(char_tensor, pos)

                    if word_vector is None:
                        word_vector = permuted_tensor
                    else:
                        if self.mode == "bipolar":
                            word_vector = word_vector * permuted_tensor
                        else:
                            word_vector = torch.bitwise_xor(word_vector, permuted_tensor)

                result[word_idx] = word_vector
                # Store in cache as PyTorch tensor
                self._word_cache[word] = word_vector

        return result

    # Update _batch_encode_texts to use improved permutation
    def _batch_encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Vectorized encoding of multiple text strings at once using n-grams.
        """
        batch_size = len(texts)

        if batch_size == 0:
            return torch.zeros((0, self.D),
                           dtype=torch.int8 if self.mode == "bipolar" else torch.uint8,
                           device=self.device)

        # Initialize result tensor
        result = torch.zeros((batch_size, self.D),
                         dtype=torch.int8 if self.mode == "bipolar" else torch.uint8,
                         device=self.device)

        # Process each text in the batch
        for i, text in enumerate(texts):
            if isinstance(text, str):
                tokens = [t for t in text.split() if t]
            else:
                tokens = list(text)

            if not tokens:
                result[i] = self._bind_identity.to(self.device)
                continue

            # Encode all unique words at once
            unique_words = list(set(tokens))
            word_encodings = {}
            
            # Batch encode all words
            encoded_words = self._batch_encode_words(unique_words)
            for j, word in enumerate(unique_words):
                word_encodings[word] = encoded_words[j]

            n = self.ngram_n
            if len(tokens) < n:
                # For texts shorter than n-gram size, just superpose the word vectors
                word_vectors = torch.stack([word_encodings[token] for token in tokens])
                result[i] = self._superpose_vectors(word_vectors)
            else:
                # Process n-grams in a vectorized way
                ngram_vectors = []
                for j in range(len(tokens) - n + 1):
                    ngram_tokens = tokens[j:j+n]
                    
                    # Stack all tokens in this n-gram for vectorized permutation
                    token_vectors = torch.stack([word_encodings[token] for token in ngram_tokens])
                    
                    # Permute each token vector according to its position
                    positions = torch.arange(n, device=self.device)
                    
                    # Vectorized permutation and binding
                    ngram_vector = self._batch_permute_and_bind(token_vectors, positions)
                    ngram_vectors.append(ngram_vector)
                
                # Superpose all n-gram vectors
                if ngram_vectors:
                    result[i] = self._superpose_vectors(torch.stack(ngram_vectors))

        return result

    def encode_chars_batch(self, chars: List[str]) -> torch.Tensor:
        """
        Encode multiple characters at once, returning a tensor of shape (len(chars), D)

        Args:
            chars: List of characters to encode

        Returns:
            PyTorch tensor of shape (len(chars), D) containing character vectors
        """
        # Create result tensor
        result = torch.zeros((len(chars), self.D),
                             dtype=torch.int8 if self.mode == "bipolar" else torch.uint8,
                             device=self.device)

        # Process characters
        for i, c in enumerate(chars):
            if c not in self._char_table:
                # Generate random vector for new character using PyTorch
                if self.mode == "bipolar":
                    self._char_table[c] = torch.randint(-1, 2, (self.D,),
                                                        dtype=torch.int8,
                                                        generator=self.rng,
                                                        device=self.device)
                    # Fix zeros to maintain -1/1 distribution
                    zeros = self._char_table[c] == 0
                    if zeros.any():
                        self._char_table[c][zeros] = torch.where(
                            torch.randint(0, 2, (zeros.sum(),),
                                          generator=self.rng,
                                          device=self.device) > 0,
                            torch.tensor(1, dtype=torch.int8, device=self.device).to(self.device),
                            torch.tensor(-1, dtype=torch.int8, device=self.device).to(self.device)
                        )

                else:
                    self._char_table[c] = torch.randint(0, 2, (self.D,),
                                                        dtype=torch.uint8,
                                                        generator=self.rng,
                                                        device=self.device)

            # Get character vector (ensuring it's a PyTorch tensor)
            char_vector = self._char_table[c]
            if not isinstance(char_vector, torch.Tensor):
                # Convert any remaining NumPy arrays to PyTorch tensors
                char_vector = torch.tensor(char_vector, device=self.device)
                self._char_table[c] = char_vector  # Update cache with tensor

            # Ensure it's on the correct device
            char_vector = char_vector.to(self.device)

            # Add to result
            result[i] = char_vector

        return result

    def encode_word_optimized(self, word: str) -> torch.Tensor:
        """
        Optimized version of encode_word using vectorized operations in PyTorch.

        Args:
            word: String to encode

        Returns:
            PyTorch tensor representing the encoded word vector
        """
        # Check cache first
        if word in self._word_cache:
            vector = self._word_cache[word]
            # Ensure it's a PyTorch tensor
            if not isinstance(vector, torch.Tensor):
                # Convert NumPy array to tensor if needed
                vector = torch.tensor(vector, device=self.device)
                # Update cache with tensor version
                self._word_cache[word] = vector
            # Make sure it's on the right device
            return vector.to(self.device)

        # Handle empty string case
        if not word:
            return self._bind_identity.clone()

        # Encode all characters in batch
        chars = list(word)
        char_vectors = self.encode_chars_batch(chars)

        # Create position tensor on the appropriate device
        positions = torch.arange(len(chars), device=self.device)

        # Perform batch permute and bind
        word_vector = self._batch_permute_and_bind(char_vectors, positions)

        # Cache the result (already as a PyTorch tensor)
        self._word_cache[word] = word_vector

        return word_vector

    def encode_tokens_ngram_vectorized(self, tokens: List[str], n: int = None) -> torch.Tensor:
        """
        Vectorized version of encode_tokens_ngram
        """
        if n is None:
            n = self.ngram_n
        n = int(n)
        
        if len(tokens) < n:
            # For short texts, encode all tokens as a batch
            word_vectors = torch.stack([self.encode_word_optimized(t) for t in tokens])
            return self._superpose_vectors(word_vectors)
        
        # Process n-grams in a fully vectorized way
        ngram_vectors = []
        for i in range(len(tokens) - n + 1):
            ngram_tokens = tokens[i:i+n]
            
            # Encode all tokens in this n-gram
            token_vectors = torch.stack([self.encode_word_optimized(token) for token in ngram_tokens])
            
            # Create positions tensor
            positions = torch.arange(n, device=self.device)
            
            # Apply the batch permute and bind
            ngram_vector = self._batch_permute_and_bind(token_vectors, positions)
            ngram_vectors.append(ngram_vector)
        
        # Superpose all n-gram vectors
        return self._superpose_vectors(torch.stack(ngram_vectors))

    def encode_text_batch(self, texts: List[Union[str, List[str]]], n: int = None) -> torch.Tensor:
        """
        Encode multiple texts at once, returning a tensor of shape (len(texts), D)
        """
        if n is None:
            n = self.ngram_n
    
        result = torch.zeros((len(texts), self.D), 
                             dtype=torch.int8 if self.mode == "bipolar" else torch.uint8,
                             device=self.device)
    
        for i, text in enumerate(texts):
            if isinstance(text, str):
                tokens = [t for t in text.split() if t]
            else:
                tokens = list(text)
            
            if not tokens:
                result[i] = self._bind_identity.clone()
                continue
            
            # Use vectorized n-gram encoding
            result[i] = self.encode_tokens_ngram_vectorized(tokens, n)
        
        return result