# tutorial (not for all functions)

import guan
import functools
import numpy as np
import cmath
from math import *

## test
print('test')
guan.test()

## Pauli matrix
print('Pauli matrix')
print('sigma_0:\n', guan.sigma_0(), '\n')
print('sigma_x:\n', guan.sigma_x(), '\n')
print('sigma_y:\n', guan.sigma_y(), '\n')
print('sigma_z:\n', guan.sigma_z(), '\n')

## Fourier transform / calculate band structures / plot figures
x = np.linspace(-pi, pi, 100)
hamiltonian_function = functools.partial(guan.one_dimensional_fourier_transform, unit_cell=0, hopping=1)
eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(x, hamiltonian_function)
guan.plot(x, eigenvalue_array, xlabel='k', ylabel='E', type='-k')

## Hamiltonian of finite size
print(guan.finite_size_along_one_direction(3), '\n')
print(guan.finite_size_along_two_directions_for_square_lattice(2, 2), '\n')
print(guan.finite_size_along_three_directions_for_cubic_lattice(2, 2, 2), '\n')

## Hamiltonian of models in the reciprocal space / calculate band structures / plot figures
x = np.linspace(-pi, pi, 100)
eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(x, guan.hamiltonian_of_square_lattice_in_quasi_one_dimension)
guan.plot(x, eigenvalue_array, xlabel='k', ylabel='E', type='-k')
eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(x, guan.hamiltonian_of_graphene_with_zigzag_in_quasi_one_dimension)
guan.plot(x, eigenvalue_array, xlabel='k', ylabel='E', type='-k')

## calculate density of states
hamiltonian = guan.finite_size_along_two_directions_for_square_lattice(2,2)
fermi_energy_array = np.linspace(-4, 4, 400)
total_dos_array = guan.total_density_of_states_with_fermi_energy_array(fermi_energy_array, hamiltonian, broadening=0.1)
guan.plot(fermi_energy_array, total_dos_array, xlabel='E', ylabel='Total DOS', type='-o')

fermi_energy = 0
N1 = 3
N2 = 4
hamiltonian = guan.finite_size_along_two_directions_for_square_lattice(N1,N2)
LDOS = guan.local_density_of_states_for_square_lattice(fermi_energy, hamiltonian, N1=N1, N2=N2)
print('square lattice:\n', LDOS, '\n')
h00 = guan.finite_size_along_one_direction(N2)
h01 = np.identity(N2)
LDOS = guan.local_density_of_states_for_square_lattice_using_dyson_equation(fermi_energy, h00=h00, h01=h01, N2=N2, N1=N1)
print(LDOS, '\n\n')
# guan.plot_contour(range(N1), range(N2), LDOS)

N1 = 3
N2 = 4
N3 = 5
hamiltonian = guan.finite_size_along_three_directions_for_cubic_lattice(N1, N2, N3)
LDOS = guan.local_density_of_states_for_cubic_lattice(fermi_energy, hamiltonian, N1=N1, N2=N2, N3=N3)
print('cubic lattice:\n', LDOS, '\n')
h00 = guan.finite_size_along_two_directions_for_square_lattice(N2, N3)
h01 = np.identity(N2*N3)
LDOS = guan.local_density_of_states_for_cubic_lattice_using_dyson_equation(fermi_energy, h00, h01, N3=N3, N2=N2, N1=N1)
print(LDOS)

## calculate conductance
fermi_energy_array = np.linspace(-5, 5, 400)
h00 = guan.finite_size_along_one_direction(4)
h01 = np.identity(4)
conductance_array = guan.calculate_conductance_with_fermi_energy_array(fermi_energy_array, h00, h01)
guan.plot(fermi_energy_array, conductance_array, xlabel='E', ylabel='Conductance', type='-o')

## calculate scattering matrix
fermi_energy = 0
h00 = guan.finite_size_along_one_direction(4)
h01 = np.identity(4)
guan.print_or_write_scattering_matrix(fermi_energy, h00, h01)

## calculate Chern number
def hamiltonian_function(kx, ky):  # one QAH model with chern number 2
    t1 = 1.0
    t2 = 1.0
    t3 = 0.5
    m = -1.0
    hamiltonian = np.zeros((2, 2), dtype=complex)
    hamiltonian[0, 1] = 2*t1*cos(kx)-1j*2*t1*cos(ky)
    hamiltonian[1, 0] = 2*t1*cos(kx)+1j*2*t1*cos(ky)
    hamiltonian[0, 0] = m+2*t3*sin(kx)+2*t3*sin(ky)+2*t2*cos(kx+ky)
    hamiltonian[1, 1] = -(m+2*t3*sin(kx)+2*t3*sin(ky)+2*t2*cos(kx+ky))
    return hamiltonian
chern_number = guan.calculate_chern_number_for_square_lattice(hamiltonian_function, precision=100)
print(chern_number)

## calculate Wilson loop
def hamiltonian_function(k): # SSH model
    gamma = 0.5
    lambda0 = 1
    delta = 0
    hamiltonian = np.zeros((2, 2), dtype=complex)
    hamiltonian[0,0] = delta
    hamiltonian[1,1] = -delta
    hamiltonian[0,1] = gamma+lambda0*cmath.exp(-1j*k)
    hamiltonian[1,0] = gamma+lambda0*cmath.exp(1j*k)
    return hamiltonian
wilson_loop_array = guan.calculate_wilson_loop(hamiltonian_function)
print('wilson loop =', wilson_loop_array)
p = np.log(wilson_loop_array)/2/pi/1j
print('p =', p, '\n')

## read and write
x = np.array([1, 2, 3])
y = np.array([5, 6, 7])
guan.write_one_dimensional_data(x, y, filename='one_dimensional_data')
matrix = np.zeros((3, 3))
matrix[0, 1] = 11
guan.write_two_dimensional_data(x, y, matrix, filename='two_dimensional_data')
x_read, y_read = guan.read_one_dimensional_data('one_dimensional_data')
print(x_read, '\n')
print(y_read, '\n\n')
x_read, y_read, matrix_read = guan.read_two_dimensional_data('two_dimensional_data')
print(x_read, '\n')
print(y_read, '\n')
print(matrix_read)

## download
# guan.download_with_scihub()
# guan.download_with_scihub('address')
# guan.download_with_scihub(num=3)