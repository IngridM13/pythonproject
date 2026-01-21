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
        """
        # Create result tensor
        result = torch.zeros((len(chars), self.D), 
                            dtype=torch.int8 if self.mode == "bipolar" else torch.uint8,
                            device=self.device)
        
        # Process unique characters
        unique_chars = set(chars)
        for i, c in enumerate(chars):
            if c not in self._char_table:
                # Generate random vector for new character
                if self.mode == "bipolar":
                    self._char_table[c] = bipolar_random(self.D, self.rng)
                else:
                    self._char_table[c] = binary_random(self.D, self.rng)
            
            # Get character vector (convert from numpy if needed)
            char_vector = self._char_table[c]
            if isinstance(char_vector, np.ndarray):
                char_vector = torch.from_numpy(char_vector).to(self.device)
                self._char_table[c] = char_vector  # Cache the tensor
                
            # Add to result
            result[i] = char_vector
        
        return result
    
    def encode_word_optimized(self, word: str) -> torch.Tensor:
        """
        Optimized version of encode_word using vectorized operations
        """
        if word in self._word_cache:
            vector = self._word_cache[word]
            if isinstance(vector, np.ndarray):
                vector = torch.from_numpy(vector).to(self.device)
                self._word_cache[word] = vector  # Update cache with tensor
            return vector
        
        if not word:
            return self._bind_identity.clone()
        
        # Encode all characters in batch
        chars = list(word)
        char_vectors = self.encode_chars_batch(chars)
        
        # Create position tensor
        positions = torch.arange(len(chars), device=self.device)
        
        # Perform batch permute and bind
        word_vector = self._batch_permute_and_bind(char_vectors, positions)
        
        # Cache the result
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