# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com. The primary location of this package is on website https://py.guanjihuan.com.

# calculate wave functions

import numpy as np

def calculate_eigenvector(hamiltonian):
    eigenvalue, eigenvector = np.linalg.eig(hamiltonian) 
    eigenvector = eigenvector[:, np.argsort(np.real(eigenvalue))] 
    return eigenvector