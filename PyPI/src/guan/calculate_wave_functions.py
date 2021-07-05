# calculate wave functions

import numpy as np

def calculate_eigenvector(hamiltonian):
    eigenvalue, eigenvector = np.linalg.eig(hamiltonian) 
    eigenvector = eigenvector[:, np.argsort(np.real(eigenvalue))] 
    return eigenvector