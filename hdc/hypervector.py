from configs.settings import HDC_DIM, DEFAULT_SEED
import numpy as np

# TODO Bernie: Yo no se si esto lo usamos Ingrid, pero por las dudas.
class HyperDimensionalContext:

    def __init__(self, vectors_dimension=HDC_DIM):
        self.vectors = {}
        self.vectors_dimension = vectors_dimension

    # Add a new vector to the context
    def add_vector(self, key, vector):
        self.vectors[key] = vector

    # Return all vectors in context
    def get_all_vectors(self):
        return self.vectors

    def get_all_vectors_as_list(self):
        return list(self.vectors.values())

    # Obtener un vector específico por su clave
    def get_vector(self, key):
        return self.vectors.get(key)

    def get_nearest_vector(self, vector):
        max_similarity = -1
        nearest_vector_key = None
        hdc = HyperDimensionalComputingBipolar(self.vectors_dimension)

        for key, stored_vector in self.vectors.items():
            similarity = hdc.cosine_similarity(vector, stored_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_vector_key = key

        return nearest_vector_key, self.vectors[nearest_vector_key]


class HyperDimensionalComputingBipolar:

    def __init__(self, dim=HDC_DIM, seed=DEFAULT_SEED):
        self.dim = dim
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate_random_hdv(self, n=1):
        hdv = np.random.randint(0, 2, size=(n, self.dim), dtype=np.int8)
        # As use bipolar, we should replace zeros with -1.
        hdv[hdv == 0] = -1
        return hdv[0]

    def add_hv(self, x, y):
        return np.clip(x + y, -1, 1)
    '''
    # No funciona bien para bipolares. En este caso se usa el producto.
    def xor_hv(self, x, y):
        return np.logical_xor(x, y)
    '''
    def dot_product_hv(self, x, y):
        return np.dot(x, y)

    def elementwise_product_hv(self, x, y):
        return x * y  # Multiplicación elemento a elemento

    ''' Éste es el binding bipolar '''
    def xor_bipolar_hv(self, x, y):
        return self.elementwise_product_hv(x, y)

    # Circle Shifting
    def shifting_hv(self, x, k=1):
        return np.roll(x, k)
    '''
    usamos uno que importamos de sklearn.metrics.pairwise
    def cosine_similarity(self, x, y):
        return self.dot_product_hv(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    '''
    def normalize(self, x):
        norm = np.linalg.norm(x)
        if norm == 0:  # Previene la división por cero
            return x
        return x / norm

# TODO: Bernie: agregate otro no para bipolares, pero para binarios.