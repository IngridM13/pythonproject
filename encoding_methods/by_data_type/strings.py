import numpy as np
import torch
from hdc.hdc_common_operations import bipolar_random, binary_random
from hdc.ops_bipolar import HyperDimensionalComputingBipolar
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
        self._char_table: dict[str, Union[np.ndarray, torch.Tensor]] = {}
        self._word_cache: dict[str, Union[np.ndarray, torch.Tensor]] = {}

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
    def _permute(self, v: Union[np.ndarray, torch.Tensor], k: int = 1) -> Union[np.ndarray, torch.Tensor]:
        """
        Vectorized permutation using PyTorch.
        Works with both single vectors and batches of vectors.
        
        Args:
            v: Vector or batch of vectors to permute (numpy or torch)
            k: Number of shifts to apply
            
        Returns:
            Permuted vector(s) in the same format as input
        """
        if k == 0:
            return v
        
        # Convert to tensor if numpy
        is_numpy = isinstance(v, np.ndarray)
        if is_numpy:
            v_tensor = torch.from_numpy(v).to(self.device)
        else:
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
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            return result.cpu().numpy()
        
        return result

    # ----------------------------
    # Generación y operaciones base
    # ----------------------------
    def _random_hv(self) -> np.ndarray:
        if self.mode == "bipolar":
            hv = bipolar_random(self.D, self.rng)
            if isinstance(hv, torch.Tensor):
                hv = hv.detach().cpu().numpy()
            return hv.astype(int)
        else:
            hv = binary_random(self.D, self.rng)
            if isinstance(hv, torch.Tensor):
                hv = hv.detach().cpu().numpy()
            return hv.astype(np.uint8)

    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.mode == "bipolar":
            return a * b
        else:
            return np.bitwise_xor(a, b)

    def _superpose(self, vecs: list[np.ndarray]) -> np.ndarray:
        if not vecs:
            return self._bind_identity.copy()
        if self.mode == "bipolar":
            acc = np.sum(vecs, axis=0, dtype=int)
            out = np.sign(acc)
            out[out == 0] = 1
            return out.astype(int)
        else:
            acc = np.sum(vecs, axis=0, dtype=int)
            threshold = (len(vecs) + 1) // 2
            return (acc >= threshold).astype(np.uint8)

    # ----------------------------
    # Unidades de codificación
    # ----------------------------
    def encode_char(self, c: str) -> np.ndarray:
        if len(c) != 1:
            raise ValueError("encode_char espera un único carácter.")
        if c not in self._char_table:
            self._char_table[c] = self._random_hv()
        return self._char_table[c]

    def encode_word(self, word: str) -> np.ndarray:
        if word in self._word_cache:
            return self._word_cache[word]
        if not word:
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
        if n is None: n = self.ngram_n
        n = int(n)
        if len(tokens) < n:
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
        if isinstance(text, str):
            tokens = [t for t in text.split() if t]
        else:
            tokens = list(text)

        if not tokens:
            return self._bind_identity.copy()

        return self.encode_tokens_ngram(tokens, n=n)

    # ----------------------------
    # Métricas y Batch
    # ----------------------------
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.mode == "bipolar":
            return _hdc_ops.cosine_similarity(a, b)
        else:
            a = np.asarray(a).astype(np.uint8)
            b = np.asarray(b).astype(np.uint8)
            eq = np.sum(a == b)
            return float(eq) / float(a.size)

    def encode(self, s: str) -> np.ndarray:
        if " " in s:
            return self.encode_text(s)
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

    # Update _batch_encode_words to use the improved permutation
    def _batch_encode_words(self, words: List[str]) -> torch.Tensor:
        """
        Vectorized encoding of multiple words at once.
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
            if isinstance(self._word_cache[word], np.ndarray):
                result[orig_idx] = torch.from_numpy(self._word_cache[word]).to(self.device)
            else:
                result[orig_idx] = self._word_cache[word].to(self.device)

        # Process uncached words
        if uncached_words:
            all_chars = set()
            for word in uncached_words:
                all_chars.update(word)

            for c in all_chars:
                if c not in self._char_table:
                    self._char_table[c] = self._random_hv()

            for i, word_idx in enumerate(uncached_indices):
                word = words[word_idx]

                if not word:
                    word_hv = self._bind_identity.clone()
                    result[word_idx] = word_hv
                    # Store in cache as numpy for consistency
                    self._word_cache[word] = word_hv.cpu().numpy()
                    continue

                word_vector = None
                for pos, char in enumerate(word):
                    char_hv = self._char_table[char]
                    if isinstance(char_hv, np.ndarray):
                        char_tensor = torch.from_numpy(char_hv).to(self.device)
                    else:
                        char_tensor = char_hv.to(self.device)

                # Use our improved vectorized permutation
                permuted_tensor = self._permute(char_tensor, pos)

                if word_vector is None:
                    word_vector = permuted_tensor
                else:
                    if self.mode == "bipolar":
                        word_vector = word_vector * permuted_tensor
                    else:
                        word_vector = torch.bitwise_xor(word_vector, permuted_tensor)

            result[word_idx] = word_vector
            # Store in cache as numpy for consistency
            self._word_cache[word] = word_vector.cpu().numpy()

        return result

    # Update _batch_encode_texts to use improved permutation
    def _batch_encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Vectorized encoding of multiple text strings at once.
        """
        batch_size = len(texts)

        if batch_size == 0:
            return torch.zeros((0, self.D),
                               dtype=torch.int8 if self.mode == "bipolar" else torch.uint8,
                               device=self.device)

        # Initialize result accumulator (int32 for summing)
        result = torch.zeros((batch_size, self.D),
                             dtype=torch.int32,
                             device=self.device)

        for i, text in enumerate(texts):
            if isinstance(text, str):
                tokens = [t for t in text.split() if t]
            else:
                tokens = list(text)

            if not tokens:
                if isinstance(self._bind_identity, np.ndarray):
                    result[i] = torch.from_numpy(self._bind_identity).to(self.device)
                else:
                    result[i] = self._bind_identity.clone()
                continue

            all_words = list(set(tokens))
            word_encodings = {}

            encoded_words = self._batch_encode_words(all_words)
            for j, word in enumerate(all_words):
                word_encodings[word] = encoded_words[j]

            n = self.ngram_n
            if len(tokens) < n:
                text_vectors = []
                for token in tokens:
                    text_vectors.append(word_encodings[token])

            text_acc = torch.sum(torch.stack(text_vectors, dim=0), dim=0)
            result[i] = text_acc
        else:
            ngram_vectors = []
            for j in range(len(tokens) - n + 1):
                ngram_tokens = tokens[j:j + n]
                ngram_vector = None

                for k, token in enumerate(ngram_tokens):
                    token_vector = word_encodings[token]
                    
                    # Use improved vectorized permutation
                    permuted_tensor = self._permute(token_vector, k)

                    if ngram_vector is None:
                        ngram_vector = permuted_tensor
                    else:
                        if self.mode == "bipolar":
                            ngram_vector = ngram_vector * permuted_tensor
                        else:
                            ngram_vector = torch.bitwise_xor(ngram_vector, permuted_tensor)

                ngram_vectors.append(ngram_vector)

            # Sum accumulation for this text
            text_acc = torch.sum(torch.stack(ngram_vectors, dim=0), dim=0)
            result[i] = text_acc

        # Finalize by applying threshold (Outside the loop!)
        if self.mode == "bipolar":
            final = torch.sign(result).to(torch.int8)
            final[final == 0] = 1
        else:
            threshold = 0.5
            final = (result > threshold).to(torch.uint8)

        return final