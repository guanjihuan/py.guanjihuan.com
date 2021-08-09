# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com. The primary location of this package is on website https://py.guanjihuan.com.

# calculate Chern number

import numpy as np
import cmath
from math import *
from .calculate_wave_functions import *

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
