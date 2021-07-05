# calculate Wilson loop

import numpy as np
from math import *
from .calculate_wave_functions import *

def calculate_wilson_loop(hamiltonian_function, k_min=-pi, k_max=pi, precision=100):
    k_array = np.linspace(k_min, k_max, precision)
    dim = np.array(hamiltonian_function(0)).shape[0]
    wilson_loop_array = np.ones(dim, dtype=complex)
    for i in range(dim):
        eigenvector_array = []
        for k in k_array:
            eigenvector  = calculate_eigenvector(hamiltonian_function(k))  
            if k != k_max:
                eigenvector_array.append(eigenvector[:, i])
            else:
                eigenvector_array.append(eigenvector_array[0])
        for i0 in range(precision-1):
            F = np.dot(eigenvector_array[i0+1].transpose().conj(), eigenvector_array[i0])
            wilson_loop_array[i] = np.dot(F, wilson_loop_array[i])
    return wilson_loop_array