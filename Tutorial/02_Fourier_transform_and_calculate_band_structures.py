import guan
import numpy as np

k_array = np.linspace(-np.pi, np.pi, 100)
hamiltonian_function = guan.one_dimensional_fourier_transform_with_k(unit_cell=0, hopping=1) # one dimensional chain
eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(k_array, hamiltonian_function)
guan.plot(k_array, eigenvalue_array, xlabel='k', ylabel='E', type='-k')