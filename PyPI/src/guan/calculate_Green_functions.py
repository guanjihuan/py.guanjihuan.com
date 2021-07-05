# calculate Green functions

import numpy as np

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