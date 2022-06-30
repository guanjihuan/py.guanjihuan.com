import guan
import numpy as np

k_array = np.linspace(-np.pi, np.pi, 100)
eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(k_array, guan.hamiltonian_of_square_lattice_in_quasi_one_dimension)
guan.plot(k_array, eigenvalue_array, xlabel='k', ylabel='E', style='-k')
eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(k_array, guan.hamiltonian_of_graphene_with_zigzag_in_quasi_one_dimension)
guan.plot(k_array, eigenvalue_array, xlabel='k', ylabel='E', style='-k')