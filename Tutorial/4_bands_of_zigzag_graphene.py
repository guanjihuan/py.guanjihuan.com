import gjh
import numpy as np
from math import *
import functools

x = np.linspace(-pi, pi, 100)
Ny = 10
unit_cell = gjh.finite_size_along_two_directions_for_graphene(1, Ny)
hopping = gjh.hopping_along_zigzag_direction_for_graphene(Ny)
hamiltonian_function = functools.partial(gjh.one_dimensional_fourier_transform, unit_cell=unit_cell, hopping=hopping)
eigenvalue_array = gjh.calculate_eigenvalue_with_one_parameter(x, hamiltonian_function)
gjh.plot(x, eigenvalue_array, xlabel='k', ylabel='E')