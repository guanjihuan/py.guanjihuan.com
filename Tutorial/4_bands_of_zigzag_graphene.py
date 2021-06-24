import guan
import numpy as np
from math import *
import functools

x = np.linspace(-pi, pi, 100)
Ny = 10
unit_cell = guan.finite_size_along_two_directions_for_graphene(1, Ny)
hopping = guan.hopping_along_zigzag_direction_for_graphene(Ny)
hamiltonian_function = functools.partial(guan.one_dimensional_fourier_transform, unit_cell=unit_cell, hopping=hopping)
eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(x, hamiltonian_function)
guan.plot(x, eigenvalue_array, xlabel='k', ylabel='E')