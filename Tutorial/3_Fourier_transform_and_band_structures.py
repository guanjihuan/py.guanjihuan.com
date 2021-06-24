import guan
import numpy as np
from math import *
import functools

x = np.linspace(-pi, pi, 100)
y = np.linspace(-pi, pi, 100)

hamiltonian_function = functools.partial(guan.one_dimensional_fourier_transform, unit_cell=0, hopping=1)
eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(x, hamiltonian_function)
guan.plot(x, eigenvalue_array, xlabel='k', ylabel='E', type='-o')

hamiltonian_function = functools.partial(guan.two_dimensional_fourier_transform_for_square_lattice, unit_cell=0, hopping_1=1, hopping_2=1)
eigenvalue_array = guan.calculate_eigenvalue_with_two_parameters(x, y, hamiltonian_function)
guan.plot_3d_surface(x, y, eigenvalue_array, xlabel='kx', ylabel='ky', zlabel='E')

hamiltonian_function = functools.partial(guan.three_dimensional_fourier_transform_for_cubic_lattice, k3=0, unit_cell=0, hopping_1=1, hopping_2=1, hopping_3=1)
eigenvalue_array = guan.calculate_eigenvalue_with_two_parameters(x, y, hamiltonian_function)
guan.plot_3d_surface(x, y, eigenvalue_array, xlabel='kx', ylabel='ky', zlabel='E')