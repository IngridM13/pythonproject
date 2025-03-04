import numpy as np

# Defininicion de la hiperdimensionalidad hypervector
DIM = 10000

# hypervector random para 1
hv_one = np.random.choice([1, -1], size=DIM)  # representación bipolar (-1, 1)

# esta funcion mapea cualquier numero a su HV basado en el HV para 1
def generate_hv(number, base_hv, dim):
    # shift circular de base_hv por el numero de posiciones
    shifted_hv = np.roll(base_hv, shift=number % dim)
    return shifted_hv

# Genera hypervector para 157
hv_157 = generate_hv(157, hv_one, DIM)

# Print
print("HV for number 1:", hv_one)
print("HV for number 157:", hv_157)

# Chequeo similaridad entre los vectores
similarity = np.dot(hv_one, hv_157) / DIM
print("Cosine similarity between HV 1 and HV 157:", similarity)
