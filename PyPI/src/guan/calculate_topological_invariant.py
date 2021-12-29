# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com/about. The primary location of this package is on website https://py.guanjihuan.com.

# calculate topological invariant

import numpy as np
import cmath
from math import *
import guan

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
            vector = guan.calculate_eigenvector(H)
            H_delta_kx = hamiltonian_function(kx+delta, ky) 
            vector_delta_kx = guan.calculate_eigenvector(H_delta_kx)
            H_delta_ky = hamiltonian_function(kx, ky+delta)
            vector_delta_ky = guan.calculate_eigenvector(H_delta_ky)
            H_delta_kx_ky = hamiltonian_function(kx+delta, ky+delta)
            vector_delta_kx_ky = guan.calculate_eigenvector(H_delta_kx_ky)
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

def calculate_chern_number_for_honeycomb_lattice(hamiltonian_function, a=1, precision=300):
    if np.array(hamiltonian_function(0, 0)).shape==():
        dim = 1
    else:
        dim = np.array(hamiltonian_function(0, 0)).shape[0]   
    chern_number = np.zeros(dim, dtype=complex)
    L1 = 4*sqrt(3)*pi/9/a
    L2 = 2*sqrt(3)*pi/9/a
    L3 = 2*pi/3/a
    delta1 = 2*L1/precision
    delta3 = 2*L3/precision
    for kx in np.arange(-L1, L1, delta1):
        for ky in np.arange(-L3, L3, delta3):
            if (-L2<=kx<=L2) or (kx>L2 and -(L1-kx)*tan(pi/3)<=ky<=(L1-kx)*tan(pi/3)) or (kx<-L2 and  -(kx-(-L1))*tan(pi/3)<=ky<=(kx-(-L1))*tan(pi/3)):
                H = hamiltonian_function(kx, ky)
                vector = guan.calculate_eigenvector(H)
                H_delta_kx = hamiltonian_function(kx+delta1, ky) 
                vector_delta_kx = guan.calculate_eigenvector(H_delta_kx)
                H_delta_ky = hamiltonian_function(kx, ky+delta3)
                vector_delta_ky = guan.calculate_eigenvector(H_delta_ky)
                H_delta_kx_ky = hamiltonian_function(kx+delta1, ky+delta3)
                vector_delta_kx_ky = guan.calculate_eigenvector(H_delta_kx_ky)
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

def calculate_wilson_loop(hamiltonian_function, k_min=-pi, k_max=pi, precision=100):
    k_array = np.linspace(k_min, k_max, precision)
    dim = np.array(hamiltonian_function(0)).shape[0]
    wilson_loop_array = np.ones(dim, dtype=complex)
    for i in range(dim):
        eigenvector_array = []
        for k in k_array:
            eigenvector  = guan.calculate_eigenvector(hamiltonian_function(k))  
            if k != k_max:
                eigenvector_array.append(eigenvector[:, i])
            else:
                eigenvector_array.append(eigenvector_array[0])
        for i0 in range(precision-1):
            F = np.dot(eigenvector_array[i0+1].transpose().conj(), eigenvector_array[i0])
            wilson_loop_array[i] = np.dot(F, wilson_loop_array[i])
    return wilson_loop_array