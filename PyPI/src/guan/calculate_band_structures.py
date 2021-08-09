# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com. The primary location of this package is on website https://py.guanjihuan.com.

# calculate band structures

import numpy as np

def calculate_eigenvalue(hamiltonian):
    if np.array(hamiltonian).shape==():
        eigenvalue = np.real(hamiltonian)
    else:
        eigenvalue, eigenvector = np.linalg.eig(hamiltonian)
        eigenvalue = np.sort(np.real(eigenvalue))
    return eigenvalue

def calculate_eigenvalue_with_one_parameter(x_array, hamiltonian_function):
    dim_x = np.array(x_array).shape[0]
    i0 = 0
    if np.array(hamiltonian_function(0)).shape==():
        eigenvalue_array = np.zeros((dim_x, 1))
        for x0 in x_array:
            hamiltonian = hamiltonian_function(x0)
            eigenvalue_array[i0, 0] = np.real(hamiltonian)
            i0 += 1
    else:
        dim = np.array(hamiltonian_function(0)).shape[0]
        eigenvalue_array = np.zeros((dim_x, dim))
        for x0 in x_array:
            hamiltonian = hamiltonian_function(x0)
            eigenvalue, eigenvector = np.linalg.eig(hamiltonian)
            eigenvalue_array[i0, :] = np.sort(np.real(eigenvalue[:]))
            i0 += 1
    return eigenvalue_array

def calculate_eigenvalue_with_two_parameters(x_array, y_array, hamiltonian_function):  
    dim_x = np.array(x_array).shape[0]
    dim_y = np.array(y_array).shape[0]
    if np.array(hamiltonian_function(0,0)).shape==():
        eigenvalue_array = np.zeros((dim_y, dim_x, 1))
        i0 = 0
        for y0 in y_array:
            j0 = 0
            for x0 in x_array:
                hamiltonian = hamiltonian_function(x0, y0)
                eigenvalue_array[i0, j0, 0] = np.real(hamiltonian)
                j0 += 1
            i0 += 1
    else:
        dim = np.array(hamiltonian_function(0, 0)).shape[0]
        eigenvalue_array = np.zeros((dim_y, dim_x, dim))
        i0 = 0
        for y0 in y_array:
            j0 = 0
            for x0 in x_array:
                hamiltonian = hamiltonian_function(x0, y0)
                eigenvalue, eigenvector = np.linalg.eig(hamiltonian)
                eigenvalue_array[i0, j0, :] = np.sort(np.real(eigenvalue[:]))
                j0 += 1
            i0 += 1
    return eigenvalue_array