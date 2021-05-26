import numpy as np
import cmath
from math import *
import copy




# test

def test():
    print('\nSuccess in the installation of GJH package!\n')




# basic functions

## Pauli matrices

def sigma_0():
    return np.eye(2)

def sigma_x():
    return np.array([[0, 1],[1, 0]])

def sigma_y():
    return np.array([[0, -1j],[1j, 0]])

def sigma_z():
    return np.array([[1, 0],[0, -1]])


## Kronecker product of Pauli matrices

def sigma_00():
    return np.kron(sigma_0(), sigma_0())

def sigma_0x():
    return np.kron(sigma_0(), sigma_x())

def sigma_0y():
    return np.kron(sigma_0(), sigma_y())

def sigma_0z():
    return np.kron(sigma_0(), sigma_z())


def sigma_x0():
    return np.kron(sigma_x(), sigma_0())

def sigma_xx():
    return np.kron(sigma_x(), sigma_x())

def sigma_xy():
    return np.kron(sigma_x(), sigma_y())

def sigma_xz():
    return np.kron(sigma_x(), sigma_z())


def sigma_y0():
    return np.kron(sigma_y(), sigma_0())

def sigma_yx():
    return np.kron(sigma_y(), sigma_x())

def sigma_yy():
    return np.kron(sigma_y(), sigma_y())

def sigma_yz():
    return np.kron(sigma_y(), sigma_z())


def sigma_z0():
    return np.kron(sigma_z(), sigma_0())

def sigma_zx():
    return np.kron(sigma_z(), sigma_x())

def sigma_zy():
    return np.kron(sigma_z(), sigma_y())

def sigma_zz():
    return np.kron(sigma_z(), sigma_z())




# Hermitian Hamiltonian of tight binding model 

def finite_size_along_one_direction(N, on_site=0, hopping=1, period=0):
    on_site = np.array(on_site)
    hopping = np.array(hopping)
    if on_site.shape==():
        dim = 1
    else:
        dim = on_site.shape[0]
    hamiltonian = np.zeros((N*dim, N*dim), dtype=complex)
    for i0 in range(N):
        hamiltonian[i0*dim+0:i0*dim+dim, i0*dim+0:i0*dim+dim] = on_site
    for i0 in range(N-1):
        hamiltonian[i0*dim+0:i0*dim+dim, (i0+1)*dim+0:(i0+1)*dim+dim] = hopping
        hamiltonian[(i0+1)*dim+0:(i0+1)*dim+dim, i0*dim+0:i0*dim+dim] = hopping.transpose().conj()
    if period == 1:
        hamiltonian[(N-1)*dim+0:(N-1)*dim+dim, 0:dim] = hopping
        hamiltonian[0:dim, (N-1)*dim+0:(N-1)*dim+dim] = hopping.transpose().conj()
    return hamiltonian


def finite_size_along_two_directions_for_square_lattice(N1, N2, on_site=0, hopping_1=1, hopping_2=1, period_1=0, period_2=0):
    on_site = np.array(on_site)
    hopping_1 = np.array(hopping_1)
    hopping_2 = np.array(hopping_2)
    if on_site.shape==():
        dim = 1
    else:
        dim = on_site.shape[0]
    hamiltonian = np.zeros((N1*N2*dim, N1*N2*dim), dtype=complex)    
    for i1 in range(N1):
        for i2 in range(N2):
            hamiltonian[i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim, i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim] = on_site
    for i1 in range(N1-1):
        for i2 in range(N2):
            hamiltonian[i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim, (i1+1)*N2*dim+i2*dim+0:(i1+1)*N2*dim+i2*dim+dim] = hopping_1
            hamiltonian[(i1+1)*N2*dim+i2*dim+0:(i1+1)*N2*dim+i2*dim+dim, i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim] = hopping_1.transpose().conj()
    for i1 in range(N1):
        for i2 in range(N2-1):
            hamiltonian[i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim, i1*N2*dim+(i2+1)*dim+0:i1*N2*dim+(i2+1)*dim+dim] = hopping_2
            hamiltonian[i1*N2*dim+(i2+1)*dim+0:i1*N2*dim+(i2+1)*dim+dim, i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim] = hopping_2.transpose().conj()
    if period_1 == 1:
        for i2 in range(N2):
            hamiltonian[(N1-1)*N2*dim+i2*dim+0:(N1-1)*N2*dim+i2*dim+dim, i2*dim+0:i2*dim+dim] = hopping_1
            hamiltonian[i2*dim+0:i2*dim+dim, (N1-1)*N2*dim+i2*dim+0:(N1-1)*N2*dim+i2*dim+dim] = hopping_1.transpose().conj()
    if period_2 == 1:
        for i1 in range(N1):
            hamiltonian[i1*N2*dim+(N2-1)*dim+0:i1*N2*dim+(N2-1)*dim+dim, i1*N2*dim+0:i1*N2*dim+dim] = hopping_2
            hamiltonian[i1*N2*dim+0:i1*N2*dim+dim, i1*N2*dim+(N2-1)*dim+0:i1*N2*dim+(N2-1)*dim+dim] = hopping_2.transpose().conj()
    return hamiltonian


def finite_size_along_three_directions_for_cubic_lattice(N1, N2, N3, on_site=0, hopping_1=1, hopping_2=1, hopping_3=1, period_1=0, period_2=0, period_3=0):
    on_site = np.array(on_site)
    hopping_1 = np.array(hopping_1)
    hopping_2 = np.array(hopping_2)
    hopping_3 = np.array(hopping_3)
    if on_site.shape==():
        dim = 1
    else:
        dim = on_site.shape[0]
    hamiltonian = np.zeros((N1*N2*N3*dim, N1*N2*N3*dim), dtype=complex) 
    for i1 in range(N1):
        for i2 in range(N2):
            for i3 in range(N3):
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim, i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim] = on_site
    for i1 in range(N1-1):
        for i2 in range(N2):
            for i3 in range(N3):
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim, (i1+1)*N2*N3*dim+i2*N3*dim+i3*dim+0:(i1+1)*N2*N3*dim+i2*N3*dim+i3*dim+dim] = hopping_1
                hamiltonian[(i1+1)*N2*N3*dim+i2*N3*dim+i3*dim+0:(i1+1)*N2*N3*dim+i2*N3*dim+i3*dim+dim, i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim] = hopping_1.transpose().conj()
    for i1 in range(N1):
        for i2 in range(N2-1):
            for i3 in range(N3):
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim, i1*N2*N3*dim+(i2+1)*N3*dim+i3*dim+0:i1*N2*N3*dim+(i2+1)*N3*dim+i3*dim+dim] = hopping_2
                hamiltonian[i1*N2*N3*dim+(i2+1)*N3*dim+i3*dim+0:i1*N2*N3*dim+(i2+1)*N3*dim+i3*dim+dim, i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim] = hopping_2.transpose().conj()
    for i1 in range(N1):
        for i2 in range(N2):
            for i3 in range(N3-1):
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim, i1*N2*N3*dim+i2*N3*dim+(i3+1)*dim+0:i1*N2*N3*dim+i2*N3*dim+(i3+1)*dim+dim] = hopping_3
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+(i3+1)*dim+0:i1*N2*N3*dim+i2*N3*dim+(i3+1)*dim+dim, i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim] = hopping_3.transpose().conj()
    if period_1 == 1:
        for i2 in range(N2):
            for i3 in range(N3):
                hamiltonian[(N1-1)*N2*N3*dim+i2*N3*dim+i3*dim+0:(N1-1)*N2*N3*dim+i2*N3*dim+i3*dim+dim, i2*N3*dim+i3*dim+0:i2*N3*dim+i3*dim+dim] = hopping_1
                hamiltonian[i2*N3*dim+i3*dim+0:i2*N3*dim+i3*dim+dim, (N1-1)*N2*N3*dim+i2*N3*dim+i3*dim+0:(N1-1)*N2*N3*dim+i2*N3*dim+i3*dim+dim] = hopping_1.transpose().conj()
    if period_2 == 1:
        for i1 in range(N1):
            for i3 in range(N3):
                hamiltonian[i1*N2*N3*dim+(N2-1)*N3*dim+i3*dim+0:i1*N2*N3*dim+(N2-1)*N3*dim+i3*dim+dim, i1*N2*N3*dim+i3*dim+0:i1*N2*N3*dim+i3*dim+dim] = hopping_2
                hamiltonian[i1*N2*N3*dim+i3*dim+0:i1*N2*N3*dim+i3*dim+dim, i1*N2*N3*dim+(N2-1)*N3*dim+i3*dim+0:i1*N2*N3*dim+(N2-1)*N3*dim+i3*dim+dim] = hopping_2.transpose().conj()
    if period_3 == 1:
        for i1 in range(N1):
            for i2 in range(N2):
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+(N3-1)*dim+0:i1*N2*N3*dim+i2*N3*dim+(N3-1)*dim+dim, i1*N2*N3*dim+i2*N3*dim+0:i1*N2*N3*dim+i2*N3*dim+dim] = hopping_3
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+0:i1*N2*N3*dim+i2*N3*dim+dim, i1*N2*N3*dim+i2*N3*dim+(N3-1)*dim+0:i1*N2*N3*dim+i2*N3*dim+(N3-1)*dim+dim] = hopping_3.transpose().conj()
    return hamiltonian


def one_dimensional_fourier_transform(k, unit_cell, hopping):
    unit_cell = np.array(unit_cell)
    hopping = np.array(hopping)
    hamiltonian = unit_cell+hopping*cmath.exp(1j*k)+hopping.transpose().conj()*cmath.exp(-1j*k)
    return hamiltonian


def two_dimensional_fourier_transform_for_square_lattice(k1, k2, unit_cell, hopping_1, hopping_2):
    unit_cell = np.array(unit_cell)
    hopping_1 = np.array(hopping_1)
    hopping_2 = np.array(hopping_2)
    hamiltonian = unit_cell+hopping_1*cmath.exp(1j*k1)+hopping_1.transpose().conj()*cmath.exp(-1j*k1)+hopping_2*cmath.exp(1j*k2)+hopping_2.transpose().conj()*cmath.exp(-1j*k2)
    return hamiltonian


def three_dimensional_fourier_transform_for_cubic_lattice(k1, k2, k3, unit_cell, hopping_1, hopping_2, hopping_3):
    unit_cell = np.array(unit_cell)
    hopping_1 = np.array(hopping_1)
    hopping_2 = np.array(hopping_2)
    hopping_3 = np.array(hopping_3)
    hamiltonian = unit_cell+hopping_1*cmath.exp(1j*k1)+hopping_1.transpose().conj()*cmath.exp(-1j*k1)+hopping_2*cmath.exp(1j*k2)+hopping_2.transpose().conj()*cmath.exp(-1j*k2)+hopping_3*cmath.exp(1j*k3)+hopping_3.transpose().conj()*cmath.exp(-1j*k3)
    return hamiltonian




# Hamiltonian of graphene lattice

def hopping_along_zigzag_direction_for_graphene(N):
    hopping = np.zeros((4*N, 4*N), dtype=complex)
    for i0 in range(N):
        hopping[4*i0+1, 4*i0+0] = 1
        hopping[4*i0+2, 4*i0+3] = 1
    return hopping


def finite_size_along_two_directions_for_graphene(N1, N2, period_1=0, period_2=0):
    on_site = finite_size_along_one_direction(4)
    hopping_1 = hopping_along_zigzag_direction_for_graphene(1)
    hopping_2 = np.zeros((4, 4), dtype=complex)
    hopping_2[3, 0] = 1
    hamiltonian = finite_size_along_two_directions_for_square_lattice(N1, N2, on_site, hopping_1, hopping_2, period_1, period_2)
    return hamiltonian




# calculate band structures

def calculate_eigenvalue(hamiltonian):
    if np.array(hamiltonian).shape==():
        eigenvalue = np.real(hamiltonian)
    else:
        eigenvalue, eigenvector = np.linalg.eig(hamiltonian)
        eigenvalue = np.sort(np.real(eigenvalue))
    return eigenvalue


def calculate_eigenvalue_with_one_parameter(x, hamiltonian_function):
    dim_x = np.array(x).shape[0]
    i0 = 0
    if np.array(hamiltonian_function(0)).shape==():
        eigenvalue_array = np.zeros((dim_x, 1))
        for x0 in x:
            hamiltonian = hamiltonian_function(x0)
            eigenvalue_array[i0, 0] = np.real(hamiltonian)
            i0 += 1
    else:
        dim = np.array(hamiltonian_function(0)).shape[0]
        eigenvalue_array = np.zeros((dim_x, dim))
        for x0 in x:
            hamiltonian = hamiltonian_function(x0)
            eigenvalue, eigenvector = np.linalg.eig(hamiltonian)
            eigenvalue_array[i0, :] = np.sort(np.real(eigenvalue[:]))
            i0 += 1
    return eigenvalue_array


def calculate_eigenvalue_with_two_parameters(x, y, hamiltonian_function):  
    dim_x = np.array(x).shape[0]
    dim_y = np.array(y).shape[0]
    if np.array(hamiltonian_function(0,0)).shape==():
        eigenvalue_array = np.zeros((dim_y, dim_x, 1))
        i0 = 0
        for y0 in y:
            j0 = 0
            for x0 in x:
                hamiltonian = hamiltonian_function(x0, y0)
                eigenvalue_array[i0, j0, 0] = np.real(hamiltonian)
                j0 += 1
            i0 += 1
    else:
        dim = np.array(hamiltonian_function(0, 0)).shape[0]
        eigenvalue_array = np.zeros((dim_y, dim_x, dim))
        i0 = 0
        for y0 in y:
            j0 = 0
            for x0 in x:
                hamiltonian = hamiltonian_function(x0, y0)
                eigenvalue, eigenvector = np.linalg.eig(hamiltonian)
                eigenvalue_array[i0, j0, :] = np.sort(np.real(eigenvalue[:]))
                j0 += 1
            i0 += 1
    return eigenvalue_array




# calculate wave functions

def calculate_eigenvector(hamiltonian):
    eigenvalue, eigenvector = np.linalg.eig(hamiltonian) 
    eigenvector = eigenvector[:, np.argsort(np.real(eigenvalue))] 
    return eigenvector




# calculate Green functions

def green_function(fermi_energy, hamiltonian, broadening, self_energy=0):
    if np.array(hamiltonian).shape==():
        dim = 1
    else:
        dim = np.array(hamiltonian).shape[0]
    green = np.linalg.inv((fermi_energy+broadening*1j)*np.eye(dim)-hamiltonian-self_energy)
    return green


def green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening, self_energy=0):
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]   
    green_nn_n = np.linalg.inv((fermi_energy+broadening*1j)*np.identity(dim)-h00-np.dot(np.dot(h01.transpose().conj(), green_nn_n_minus), h01)-self_energy)
    return green_nn_n


def green_function_in_n(green_in_n_minus, h01, green_nn_n):
    green_in_n = np.dot(np.dot(green_in_n_minus, h01), green_nn_n)
    return green_in_n


def green_function_ni_n(green_nn_n, h01, green_ni_n_minus):
    h01 = np.array(h01)
    green_ni_n = np.dot(np.dot(green_nn_n, h01.transpose().conj()), green_ni_n_minus)
    return green_ni_n


def green_function_ii_n(green_ii_n_minus, green_in_n_minus, h01, green_nn_n, green_ni_n_minus):
    green_ii_n = green_ii_n_minus+np.dot(np.dot(np.dot(np.dot(green_in_n_minus, h01), green_nn_n), h01.transpose().conj()),green_ni_n_minus)
    return green_ii_n




# calculate density of states

def total_density_of_states(fermi_energy, hamiltonian, broadening=0.01):
    green = green_function(fermi_energy, hamiltonian, broadening)
    total_dos = -np.trace(np.imag(green))/pi
    return total_dos


def total_density_of_states_with_fermi_energy_array(fermi_energy_array, hamiltonian, broadening=0.01):
    dim = np.array(fermi_energy_array).shape[0]
    total_dos_array = np.zeros(dim)
    i0 = 0
    for fermi_energy in fermi_energy_array:
        total_dos_array[i0] = total_density_of_states(fermi_energy, hamiltonian, broadening)
        i0 += 1
    return total_dos_array


def local_density_of_states_for_square_lattice(fermi_energy, hamiltonian, N1, N2, internal_degree=1, broadening=0.01):
    # dim_hamiltonian = N1*N2*internal_degree
    green = green_function(fermi_energy, hamiltonian, broadening)
    local_dos = np.zeros((N2, N1))
    for i1 in range(N1):
        for i2 in range(N2):
            for i in range(internal_degree): 
                local_dos[i2, i1] = local_dos[i2, i1]-np.imag(green[i1*N2*internal_degree+i2*internal_degree+i, i1*N2*internal_degree+i2*internal_degree+i])/pi
    return local_dos


def local_density_of_states_for_cubic_lattice(fermi_energy, hamiltonian, N1, N2, N3, internal_degree=1, broadening=0.01):
    # dim_hamiltonian = N1*N2*N3*internal_degree
    green = green_function(fermi_energy, hamiltonian, broadening)
    local_dos = np.zeros((N3, N2, N1))
    for i1 in range(N1):
        for i2 in range(N2):
            for i3 in range(N3):
                for i in range(internal_degree): 
                    local_dos[i3, i2, i1] = local_dos[i3, i2, i1]-np.imag(green[i1*N2*N3*internal_degree+i2*N3*internal_degree+i3*internal_degree+i, i1*N2*N3*internal_degree+i2*N3*internal_degree+i3*internal_degree+i])/pi
    return local_dos


def local_density_of_states_for_square_lattice_using_dyson_equation(fermi_energy, h00, h01, N2, N1, internal_degree=1, broadening=0.01):
    # dim_h00 = N2*internal_degree
    local_dos = np.zeros((N2, N1))
    green_11_1 = green_function(fermi_energy, h00, broadening)
    for i1 in range(N1):
        green_nn_n_minus = green_11_1
        green_in_n_minus = green_11_1
        green_ni_n_minus = green_11_1
        green_ii_n_minus = green_11_1
        for i2_0 in range(i1):
            green_nn_n = green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
        if i1!=0:
            green_in_n_minus = green_nn_n
            green_ni_n_minus = green_nn_n
            green_ii_n_minus = green_nn_n
        for size_0 in range(N1-1-i1):
            green_nn_n = green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
            green_ii_n = green_function_ii_n(green_ii_n_minus, green_in_n_minus, h01, green_nn_n, green_ni_n_minus)
            green_ii_n_minus = green_ii_n
            green_in_n = green_function_in_n(green_in_n_minus, h01, green_nn_n)
            green_in_n_minus = green_in_n
            green_ni_n = green_function_ni_n(green_nn_n, h01, green_ni_n_minus)
            green_ni_n_minus = green_ni_n
        for i2 in range(N2):
            for i in range(internal_degree):
                local_dos[i2, i1] = local_dos[i2, i1] - np.imag(green_ii_n_minus[i2*internal_degree+i, i2*internal_degree+i])/pi
    return local_dos


def local_density_of_states_for_cubic_lattice_using_dyson_equation(fermi_energy, h00, h01, N3, N2, N1, internal_degree=1, broadening=0.01):
    # dim_h00 = N2*N3*internal_degree
    local_dos = np.zeros((N3, N2, N1))
    green_11_1 = green_function(fermi_energy, h00, broadening)
    for i1 in range(N1):
        green_nn_n_minus = green_11_1
        green_in_n_minus = green_11_1
        green_ni_n_minus = green_11_1
        green_ii_n_minus = green_11_1
        for i1_0 in range(i1):
            green_nn_n = green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
        if i1!=0:
            green_in_n_minus = green_nn_n
            green_ni_n_minus = green_nn_n
            green_ii_n_minus = green_nn_n
        for size_0 in range(N1-1-i1):
            green_nn_n = green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
            green_ii_n = green_function_ii_n(green_ii_n_minus, green_in_n_minus, h01, green_nn_n, green_ni_n_minus)
            green_ii_n_minus = green_ii_n
            green_in_n = green_function_in_n(green_in_n_minus, h01, green_nn_n)
            green_in_n_minus = green_in_n
            green_ni_n = green_function_ni_n(green_nn_n, h01, green_ni_n_minus)
            green_ni_n_minus = green_ni_n
        for i2 in range(N2):
            for i3 in range(N3):
                for i in range(internal_degree):
                    local_dos[i3, i2, i1] = local_dos[i3, i2, i1] -np.imag(green_ii_n_minus[i2*N3*internal_degree+i3*internal_degree+i, i2*N3*internal_degree+i3*internal_degree+i])/pi       
    return local_dos




# calculate conductance

def transfer_matrix(fermi_energy, h00, h01):
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    transfer = np.zeros((2*dim, 2*dim), dtype=complex)
    transfer[0:dim, 0:dim] = np.dot(np.linalg.inv(h01), fermi_energy*np.identity(dim)-h00)
    transfer[0:dim, dim:2*dim] = np.dot(-1*np.linalg.inv(h01), h01.transpose().conj())
    transfer[dim:2*dim, 0:dim] = np.identity(dim)
    transfer[dim:2*dim, dim:2*dim] = 0
    return transfer


def surface_green_function_of_lead(fermi_energy, h00, h01):
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    fermi_energy = fermi_energy+1e-9*1j
    transfer = transfer_matrix(fermi_energy, h00, h01)
    eigenvalue, eigenvector = np.linalg.eig(transfer)
    ind = np.argsort(np.abs(eigenvalue))
    temp = np.zeros((2*dim, 2*dim), dtype=complex)
    i0 = 0
    for ind0 in ind:
        temp[:, i0] = eigenvector[:, ind0]
        i0 += 1
    s1 = temp[dim:2*dim, 0:dim]
    s2 = temp[0:dim, 0:dim]
    s3 = temp[dim:2*dim, dim:2*dim]
    s4 = temp[0:dim, dim:2*dim]
    right_lead_surface = np.linalg.inv(fermi_energy*np.identity(dim)-h00-np.dot(np.dot(h01, s2), np.linalg.inv(s1)))
    left_lead_surface = np.linalg.inv(fermi_energy*np.identity(dim)-h00-np.dot(np.dot(h01.transpose().conj(), s3), np.linalg.inv(s4)))
    return right_lead_surface, left_lead_surface


def self_energy_of_lead(fermi_energy, h00, h01):
    h01 = np.array(h01)
    right_lead_surface, left_lead_surface = surface_green_function_of_lead(fermi_energy, h00, h01)
    right_self_energy = np.dot(np.dot(h01, right_lead_surface), h01.transpose().conj())
    left_self_energy = np.dot(np.dot(h01.transpose().conj(), left_lead_surface), h01)
    return right_self_energy, left_self_energy


def calculate_conductance(fermi_energy, h00, h01, length=100):
    right_self_energy, left_self_energy = self_energy_of_lead(fermi_energy, h00, h01)
    for ix in range(length):
        if ix == 0:
            green_nn_n = green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
            green_0n_n = copy.deepcopy(green_nn_n)
        elif ix != length-1:
            green_nn_n = green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0)
            green_0n_n = green_function_in_n(green_0n_n, h01, green_nn_n)
        else:
            green_nn_n = green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
            green_0n_n = green_function_in_n(green_0n_n, h01, green_nn_n)
    right_self_energy = (right_self_energy - right_self_energy.transpose().conj())*(0+1j)
    left_self_energy = (left_self_energy - left_self_energy.transpose().conj())*(0+1j)
    conductance = np.trace(np.dot(np.dot(np.dot(left_self_energy, green_0n_n), right_self_energy), green_0n_n.transpose().conj()))
    return conductance


def calculate_conductance_with_fermi_energy_array(fermi_energy_array, h00, h01, length=100):
    dim = np.array(fermi_energy_array).shape[0]
    conductance_array = np.zeros(dim)
    i0 = 0
    for fermi_energy_0 in fermi_energy_array:
        conductance_array[i0] = np.real(calculate_conductance(fermi_energy_0, h00, h01, length))
        i0 += 1
    return conductance_array




# calculate scattering matrix

def if_active_channel(k_of_channel):
    if np.abs(np.imag(k_of_channel))<1e-6:
        if_active = 1
    else:
        if_active = 0
    return if_active


def get_k_and_velocity_of_channel(fermi_energy, h00, h01):
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    transfer = transfer_matrix(fermi_energy, h00, h01)
    eigenvalue, eigenvector = np.linalg.eig(transfer)
    k_of_channel = np.log(eigenvalue)/1j
    ind = np.argsort(np.real(k_of_channel))
    k_of_channel = np.sort(k_of_channel)
    temp = np.zeros((2*dim, 2*dim), dtype=complex)
    temp2 = np.zeros((2*dim), dtype=complex)
    i0 = 0
    for ind0 in ind:
        temp[:, i0] = eigenvector[:, ind0]
        temp2[i0] = eigenvalue[ind0]
        i0 += 1
    eigenvalue = copy.deepcopy(temp2)
    temp = temp[0:dim, :]
    factor = np.zeros(2*dim, dtype=complex)
    for dim0 in range(dim):
        factor = factor+np.square(np.abs(temp[dim0, :]))
    for dim0 in range(2*dim):
        temp[:, dim0] = temp[:, dim0]/np.sqrt(factor[dim0])
    velocity_of_channel = np.zeros((2*dim), dtype=complex)
    for dim0 in range(2*dim):
        velocity_of_channel[dim0] = eigenvalue[dim0]*np.dot(np.dot(temp[0:dim, :].transpose().conj(), h01),temp[0:dim, :])[dim0, dim0]
    velocity_of_channel = -2*np.imag(velocity_of_channel)
    eigenvector = copy.deepcopy(temp) 
    return k_of_channel, velocity_of_channel, eigenvalue, eigenvector


def get_classified_k_velocity_u_and_f(fermi_energy, h00, h01):
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    k_of_channel, velocity_of_channel, eigenvalue, eigenvector = get_k_and_velocity_of_channel(fermi_energy, h00, h01)
    ind_right_active = 0; ind_right_evanescent = 0; ind_left_active = 0; ind_left_evanescent = 0
    k_right = np.zeros(dim, dtype=complex); k_left = np.zeros(dim, dtype=complex)
    velocity_right = np.zeros(dim, dtype=complex); velocity_left = np.zeros(dim, dtype=complex)
    lambda_right = np.zeros(dim, dtype=complex); lambda_left = np.zeros(dim, dtype=complex)
    u_right = np.zeros((dim, dim), dtype=complex); u_left = np.zeros((dim, dim), dtype=complex)
    for dim0 in range(2*dim):
        if_active = if_active_channel(k_of_channel[dim0])
        if if_active_channel(k_of_channel[dim0]) == 1:
            direction = np.sign(velocity_of_channel[dim0])
        else:
            direction = np.sign(np.imag(k_of_channel[dim0]))
        if direction == 1:
            if if_active == 1:  # right-moving active channel
                k_right[ind_right_active] = k_of_channel[dim0]
                velocity_right[ind_right_active] = velocity_of_channel[dim0]
                lambda_right[ind_right_active] = eigenvalue[dim0]
                u_right[:, ind_right_active] = eigenvector[:, dim0]
                ind_right_active += 1
            else:               # right-moving evanescent channel
                k_right[dim-1-ind_right_evanescent] = k_of_channel[dim0]
                velocity_right[dim-1-ind_right_evanescent] = velocity_of_channel[dim0]
                lambda_right[dim-1-ind_right_evanescent] = eigenvalue[dim0]
                u_right[:, dim-1-ind_right_evanescent] = eigenvector[:, dim0]
                ind_right_evanescent += 1
        else:
            if if_active == 1:  # left-moving active channel
                k_left[ind_left_active] = k_of_channel[dim0]
                velocity_left[ind_left_active] = velocity_of_channel[dim0]
                lambda_left[ind_left_active] = eigenvalue[dim0]
                u_left[:, ind_left_active] = eigenvector[:, dim0]
                ind_left_active += 1
            else:               # left-moving evanescent channel
                k_left[dim-1-ind_left_evanescent] = k_of_channel[dim0]
                velocity_left[dim-1-ind_left_evanescent] = velocity_of_channel[dim0]
                lambda_left[dim-1-ind_left_evanescent] = eigenvalue[dim0]
                u_left[:, dim-1-ind_left_evanescent] = eigenvector[:, dim0]
                ind_left_evanescent += 1
    lambda_matrix_right = np.diag(lambda_right)
    lambda_matrix_left = np.diag(lambda_left)
    f_right = np.dot(np.dot(u_right, lambda_matrix_right), np.linalg.inv(u_right))
    f_left = np.dot(np.dot(u_left, lambda_matrix_left), np.linalg.inv(u_left))
    return k_right, k_left, velocity_right, velocity_left, f_right, f_left, u_right, u_left, ind_right_active


def calculate_scattering_matrix(fermi_energy, h00, h01, length=100):
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    k_right, k_left, velocity_right, velocity_left, f_right, f_left, u_right, u_left, ind_right_active = get_classified_k_velocity_u_and_f(fermi_energy, h00, h01)
    right_self_energy = np.dot(h01, f_right)
    left_self_energy = np.dot(h01.transpose().conj(), np.linalg.inv(f_left))
    for i0 in range(length):
        if i0 == 0:
            green_nn_n = green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
            green_00_n = copy.deepcopy(green_nn_n)
            green_0n_n = copy.deepcopy(green_nn_n)
            green_n0_n = copy.deepcopy(green_nn_n)
        elif i0 != length-1: 
            green_nn_n = green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0) 
        else:
            green_nn_n = green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
        green_00_n = green_function_ii_n(green_00_n, green_0n_n, h01, green_nn_n, green_n0_n)
        green_0n_n = green_function_in_n(green_0n_n, h01, green_nn_n)
        green_n0_n = green_function_ni_n(green_nn_n, h01, green_n0_n)
    temp = np.dot(h01.transpose().conj(), np.linalg.inv(f_right)-np.linalg.inv(f_left))
    transmission_matrix = np.dot(np.dot(np.linalg.inv(u_right), np.dot(green_n0_n, temp)), u_right) 
    reflection_matrix = np.dot(np.dot(np.linalg.inv(u_left), np.dot(green_00_n, temp)-np.identity(dim)), u_right)
    for dim0 in range(dim):
        for dim1 in range(dim):
            if_active = if_active_channel(k_right[dim0])*if_active_channel(k_right[dim1])
            if if_active == 1:
                transmission_matrix[dim0, dim1] = np.sqrt(np.abs(velocity_right[dim0]/velocity_right[dim1])) * transmission_matrix[dim0, dim1]
                reflection_matrix[dim0, dim1] = np.sqrt(np.abs(velocity_left[dim0]/velocity_right[dim1]))*reflection_matrix[dim0, dim1]
            else:
                transmission_matrix[dim0, dim1] = 0
                reflection_matrix[dim0, dim1] = 0
    sum_of_tran_refl_array = np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])), axis=0)+np.sum(np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active])), axis=0)
    for sum_of_tran_refl in sum_of_tran_refl_array:
        if sum_of_tran_refl > 1.001:
            print('Error Alert: scattering matrix is not normalized!')
    return transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active


def print_or_write_scattering_matrix(fermi_energy, h00, h01, length=100, on_print=1, on_write=0):
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active = calculate_scattering_matrix(fermi_energy, h00, h01, length)
    if on_print == 1:
        print('\nActive channel (left or right) = ', ind_right_active)
        print('Evanescent channel (left or right) = ', dim-ind_right_active, '\n')
        print('K of right-moving active channels:\n', np.real(k_right[0:ind_right_active]))
        print('K of left-moving active channels:\n', np.real(k_left[0:ind_right_active]), '\n')
        print('Velocity of right-moving active channels:\n', np.real(velocity_right[0:ind_right_active]))
        print('Velocity of left-moving active channels:\n', np.real(velocity_left[0:ind_right_active]), '\n')
        print('Transmission matrix:\n', np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])))
        print('Reflection matrix:\n', np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active])), '\n')
        print('Total transmission of channels:\n', np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])), axis=0))
        print('Total reflection of channels:\n',np.sum(np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active])), axis=0))
        print('Sum of transmission and reflection of channels:\n', np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])), axis=0) + np.sum(np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active])), axis=0))
        print('Total conductance = ', np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active]))), '\n')
    if on_write == 1:
        with open('a.txt', 'w') as f:
            f.write('Active channel (left or right) = ' + str(ind_right_active) + '\n')
            f.write('Evanescent channel (left or right) = ' + str(dim - ind_right_active) + '\n\n')
            f.write('Channel               K                                     Velocity\n')
            for ind0 in range(ind_right_active):
                f.write('   '+str(ind0 + 1) + '   |    '+str(np.real(k_right[ind0]))+'            ' + str(np.real(velocity_right[ind0]))+'\n')
            f.write('\n')
            for ind0 in range(ind_right_active):
                f.write('  -' + str(ind0 + 1) + '   |    ' + str(np.real(k_left[ind0])) + '            ' + str(np.real(velocity_left[ind0])) + '\n')
            f.write('\nScattering matrix:\n              ')
            for ind0 in range(ind_right_active):
                f.write(str(ind0+1)+'               ')
            f.write('\n')
            for ind1 in range(ind_right_active):
                f.write('  '+str(ind1+1)+'    ')
                for ind2 in range(ind_right_active):
                    f.write('%f' % np.square(np.abs(transmission_matrix[ind1, ind2]))+'    ')
                f.write('\n')
            f.write('\n')
            for ind1 in range(ind_right_active):
                f.write(' -'+str(ind1+1)+'    ')
                for ind2 in range(ind_right_active):
                    f.write('%f' % np.square(np.abs(reflection_matrix[ind1, ind2]))+'    ')
                f.write('\n')
            f.write('\n')
            f.write('Total transmission of channels:\n'+str(np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])), axis=0))+'\n')
            f.write('Total conductance = '+str(np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active]))))+'\n')




# calculate Chern number

def calculate_chern_number_for_square_lattice(hamiltonian_function, precision=100):
    if np.array(hamiltonian_function(0, 0)).shape==():
        dim = 1
    else:
        dim = np.array(hamiltonian_function(0, 0)).shape[0]   
    delta = 2*pi/precision
    chern_number = np.zeros(dim, dtype=complex)
    for kx in np.arange(-pi, pi, delta):
        for ky in np.arange(-pi, pi, delta):
            H = hamiltonian_function(kx, ky)
            vector = calculate_eigenvector(H)
            H_delta_kx = hamiltonian_function(kx+delta, ky) 
            vector_delta_kx = calculate_eigenvector(H_delta_kx)
            H_delta_ky = hamiltonian_function(kx, ky+delta)
            vector_delta_ky = calculate_eigenvector(H_delta_ky)
            H_delta_kx_ky = hamiltonian_function(kx+delta, ky+delta)
            vector_delta_kx_ky = calculate_eigenvector(H_delta_kx_ky)
            for i in range(dim):
                vector_i = vector[:, i]
                vector_delta_kx_i = vector_delta_kx[:, i]
                vector_delta_ky_i = vector_delta_ky[:, i]
                vector_delta_kx_ky_i = vector_delta_kx_ky[:, i]
                Ux = np.dot(np.conj(vector_i), vector_delta_kx_i)/abs(np.dot(np.conj(vector_i), vector_delta_kx_i))
                Uy = np.dot(np.conj(vector_i), vector_delta_ky_i)/abs(np.dot(np.conj(vector_i), vector_delta_ky_i))
                Ux_y = np.dot(np.conj(vector_delta_ky_i), vector_delta_kx_ky_i)/abs(np.dot(np.conj(vector_delta_ky_i), vector_delta_kx_ky_i))
                Uy_x = np.dot(np.conj(vector_delta_kx_i), vector_delta_kx_ky_i)/abs(np.dot(np.conj(vector_delta_kx_i), vector_delta_kx_ky_i))
                F = cmath.log(Ux*Uy_x*(1/Ux_y)*(1/Uy))
                chern_number[i] = chern_number[i] + F
    chern_number = chern_number/(2*pi*1j)
    return chern_number




# calculate Wilson loop

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




# read and write

def read_one_dimensional_data(filename='a'): 
    f = open(filename+'.txt', 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x = np.array([])
    y = np.array([])
    for row in row_list:
        column = np.array(row.split()) 
        if column.shape[0] != 0:  
            x = np.append(x, [float(column[0])], axis=0)  
            y_row = np.zeros(dim_column-1)
            for dim0 in range(dim_column-1):
                y_row[dim0] = float(column[dim0+1])
            if np.array(y).shape[0] == 0:
                y = [y_row]
            else:
                y = np.append(y, [y_row], axis=0)
    return x, y


def read_two_dimensional_data(filename='a'): 
    f = open(filename+'.txt', 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x = np.array([])
    y = np.array([])
    matrix = np.array([])
    for i0 in range(row_list.shape[0]):
        column = np.array(row_list[i0].split()) 
        if i0 == 0:
            x_str = column[1::] 
            x = np.zeros(x_str.shape[0])
            for i00 in range(x_str.shape[0]):
                x[i00] = float(x_str[i00]) 
        elif column.shape[0] != 0: 
            y = np.append(y, [float(column[0])], axis=0)  
            matrix_row = np.zeros(dim_column-1)
            for dim0 in range(dim_column-1):
                matrix_row[dim0] = float(column[dim0+1])
            if np.array(matrix).shape[0] == 0:
                matrix = [matrix_row]
            else:
                matrix = np.append(matrix, [matrix_row], axis=0)
    return x, y, matrix


def write_one_dimensional_data(x, y, filename='a'): 
    with open(filename+'.txt', 'w') as f:
        i0 = 0
        for x0 in x:
            f.write(str(x0)+'   ')
            if len(y.shape) == 1:
                f.write(str(y[i0])+'\n')
            elif len(y.shape) == 2:
                for j0 in range(y.shape[1]):
                    f.write(str(y[i0, j0])+'   ')
                f.write('\n')
            i0 += 1


def write_two_dimensional_data(x, y, matrix, filename='a'): 
    with open(filename+'.txt', 'w') as f:
        f.write('0   ')
        for x0 in x:
            f.write(str(x0)+'   ')
        f.write('\n')
        i0 = 0
        for y0 in y:
            f.write(str(y0))
            j0 = 0
            for x0 in x:
                f.write('   '+str(matrix[i0, j0])+'   ')
                j0 += 1
            f.write('\n')
            i0 += 1




# plot figures

def plot(x, y, xlabel='x', ylabel='y', title='', filename='a', show=1, save=0, type='', y_min=None, y_max=None): 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.20, left=0.18) 
    ax.plot(x, y, type)
    ax.grid()
    ax.set_title(title, fontsize=20, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=20, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=20, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y_min=min(y)
        if y_max==None:
            y_max=max(y)
        ax.set_ylim(y_min, y_max)
    ax.tick_params(labelsize=20) 
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    if save == 1:
        plt.savefig(filename+'.jpg', dpi=300) 
    if show == 1:
        plt.show()
    plt.close('all')


def plot_3d_surface(x, y, matrix, xlabel='x', ylabel='y', zlabel='z', title='', filename='a', show=1, save=0, z_min=None, z_max=None): 
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    matrix = np.array(matrix)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.subplots_adjust(bottom=0.1, right=0.65) 
    x, y = np.meshgrid(x, y)
    if len(matrix.shape) == 2:
        surf = ax.plot_surface(x, y, matrix, cmap=cm.coolwarm, linewidth=0, antialiased=False) 
    elif len(matrix.shape) == 3:
        for i0 in range(matrix.shape[2]):
            surf = ax.plot_surface(x, y, matrix[:,:,i0], cmap=cm.coolwarm, linewidth=0, antialiased=False) 
    ax.set_title(title, fontsize=20, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=20, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=20, fontfamily='Times New Roman') 
    ax.set_zlabel(zlabel, fontsize=20, fontfamily='Times New Roman') 
    ax.zaxis.set_major_locator(LinearLocator(5)) 
    ax.zaxis.set_major_formatter('{x:.2f}')  
    if z_min!=None or z_max!=None:
        if z_min==None:
            z_min=matrix.min()
        if z_max==None:
            z_max=matrix.max()
        ax.set_zlim(z_min, z_max)
    ax.tick_params(labelsize=15) 
    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
    [label.set_fontname('Times New Roman') for label in labels] 
    cax = plt.axes([0.80, 0.15, 0.05, 0.75]) 
    cbar = fig.colorbar(surf, cax=cax)  
    cbar.ax.tick_params(labelsize=15)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
    if save == 1:
        plt.savefig(filename+'.jpg', dpi=300) 
    if show == 1:
        plt.show()
    plt.close('all')


def plot_contour(x, y, matrix, xlabel='x', ylabel='y', title='', filename='a', show=1, save=0):  
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2, right=0.75, left = 0.16) 
    x, y = np.meshgrid(x, y)
    contour = ax.contourf(x,y,matrix,cmap='jet') 
    ax.set_title(title, fontsize=20, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=20, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=20, fontfamily='Times New Roman') 
    ax.tick_params(labelsize=15) 
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels] 
    cax = plt.axes([0.78, 0.17, 0.08, 0.71])
    cbar = fig.colorbar(contour, cax=cax) 
    cbar.ax.tick_params(labelsize=15) 
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
    if save == 1:
        plt.savefig(filename+'.jpg', dpi=300) 
    if show == 1:
        plt.show()
    plt.close('all')