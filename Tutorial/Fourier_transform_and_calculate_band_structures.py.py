import guan
import numpy as np
import functools

# Fourier transform / calculate band structures / plot figures
x_array = np.linspace(-np.pi, np.pi, 100)
hamiltonian_function = functools.partial(guan.one_dimensional_fourier_transform, unit_cell=0, hopping=1)
eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(x_array, hamiltonian_function)
guan.plot(x_array, eigenvalue_array, xlabel='k', ylabel='E', type='-k')