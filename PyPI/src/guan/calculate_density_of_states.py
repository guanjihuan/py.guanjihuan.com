# calculate density of states

import numpy as np
from math import *
from .calculate_Green_functions import *

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