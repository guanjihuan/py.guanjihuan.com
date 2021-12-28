# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com/about. The primary location of this package is on website https://py.guanjihuan.com.

# Hamiltonian of models in the reciprocal space

import numpy as np
import cmath
from math import *
import guan

def hamiltonian_of_simple_chain(k):
    hamiltonian = guan.one_dimensional_fourier_transform(k, unit_cell=0, hopping=1)
    return hamiltonian

def hamiltonian_of_square_lattice(k1, k2):
    hamiltonian = guan.two_dimensional_fourier_transform_for_square_lattice(k1, k2, unit_cell=0, hopping_1=1, hopping_2=1)
    return hamiltonian

def hamiltonian_of_square_lattice_in_quasi_one_dimension(k, N=10):
    h00 = np.zeros((N, N), dtype=complex)  # hopping in a unit cell
    h01 = np.zeros((N, N), dtype=complex)  # hopping between unit cells
    for i in range(N-1):   
        h00[i, i+1] = 1
        h00[i+1, i] = 1
    for i in range(N):   
        h01[i, i] = 1
    hamiltonian = guan.one_dimensional_fourier_transform(k, unit_cell=h00, hopping=h01) 
    return hamiltonian

def hamiltonian_of_cubic_lattice(k1, k2, k3):
    hamiltonian = guan.three_dimensional_fourier_transform_for_cubic_lattice(k1, k2, k3, unit_cell=0, hopping_1=1, hopping_2=1, hopping_3=1)
    return hamiltonian

def hamiltonian_of_ssh_model(k, v=0.6, w=1):
    hamiltonian = np.zeros((2, 2), dtype=complex)
    hamiltonian[0,1] = v+w*cmath.exp(-1j*k)
    hamiltonian[1,0] = v+w*cmath.exp(1j*k)
    return hamiltonian

def hamiltonian_of_graphene(k1, k2, M=0, t=1, a=1/sqrt(3)):
    h0 = np.zeros((2, 2), dtype=complex)  # mass term
    h1 = np.zeros((2, 2), dtype=complex)  # nearest hopping
    h0[0, 0] = M     
    h0[1, 1] = -M
    h1[1, 0] = t*(cmath.exp(1j*k2*a)+cmath.exp(1j*sqrt(3)/2*k1*a-1j/2*k2*a)+cmath.exp(-1j*sqrt(3)/2*k1*a-1j/2*k2*a))   
    h1[0, 1] = h1[1, 0].conj()
    hamiltonian = h0 + h1
    return hamiltonian

def hamiltonian_of_graphene_with_zigzag_in_quasi_one_dimension(k, N=10, M=0, t=1):
    h00 = np.zeros((4*N, 4*N), dtype=complex)  # hopping in a unit cell
    h01 = np.zeros((4*N, 4*N), dtype=complex)  # hopping between unit cells
    for i in range(N):
        h00[i*4+0, i*4+0] = M
        h00[i*4+1, i*4+1] = -M
        h00[i*4+2, i*4+2] = M
        h00[i*4+3, i*4+3] = -M
        h00[i*4+0, i*4+1] = t
        h00[i*4+1, i*4+0] = t
        h00[i*4+1, i*4+2] = t
        h00[i*4+2, i*4+1] = t
        h00[i*4+2, i*4+3] = t
        h00[i*4+3, i*4+2] = t
    for i in range(N-1):
        h00[i*4+3, (i+1)*4+0] = t
        h00[(i+1)*4+0, i*4+3] = t
    for i in range(N):
        h01[i*4+1, i*4+0] = t
        h01[i*4+2, i*4+3] = t
    hamiltonian = guan.one_dimensional_fourier_transform(k, unit_cell=h00, hopping=h01) 
    return hamiltonian

def hamiltonian_of_haldane_model(k1, k2, M=2/3, t1=1, t2=1/3, phi=pi/4, a=1/sqrt(3)):
    h0 = np.zeros((2, 2), dtype=complex)  # mass term
    h1 = np.zeros((2, 2), dtype=complex)  # nearest hopping
    h2 = np.zeros((2, 2), dtype=complex)  # next nearest hopping
    h0[0, 0] = M
    h0[1, 1] = -M
    h1[1, 0] = t1*(cmath.exp(1j*k2*a)+cmath.exp(1j*sqrt(3)/2*k1*a-1j/2*k2*a)+cmath.exp(-1j*sqrt(3)/2*k1*a-1j/2*k2*a))
    h1[0, 1] = h1[1, 0].conj()
    h2[0, 0] = t2*cmath.exp(-1j*phi)*(cmath.exp(1j*sqrt(3)*k1*a)+cmath.exp(-1j*sqrt(3)/2*k1*a+1j*3/2*k2*a)+cmath.exp(-1j*sqrt(3)/2*k1*a-1j*3/2*k2*a))
    h2[1, 1] = t2*cmath.exp(1j*phi)*(cmath.exp(1j*sqrt(3)*k1*a)+cmath.exp(-1j*sqrt(3)/2*k1*a+1j*3/2*k2*a)+cmath.exp(-1j*sqrt(3)/2*k1*a-1j*3/2*k2*a))
    hamiltonian = h0 + h1 + h2 + h2.transpose().conj()
    return hamiltonian

def hamiltonian_of_haldane_model_in_quasi_one_dimension(k, N=10, M=2/3, t1=1, t2=1/3, phi=pi/4):
    h00 = np.zeros((4*N, 4*N), dtype=complex)  # hopping in a unit cell
    h01 = np.zeros((4*N, 4*N), dtype=complex)  # hopping between unit cells
    for i in range(N):
        h00[i*4+0, i*4+0] = M
        h00[i*4+1, i*4+1] = -M
        h00[i*4+2, i*4+2] = M
        h00[i*4+3, i*4+3] = -M
        h00[i*4+0, i*4+1] = t1
        h00[i*4+1, i*4+0] = t1
        h00[i*4+1, i*4+2] = t1
        h00[i*4+2, i*4+1] = t1
        h00[i*4+2, i*4+3] = t1
        h00[i*4+3, i*4+2] = t1
        h00[i*4+0, i*4+2] = t2*cmath.exp(-1j*phi)
        h00[i*4+2, i*4+0] = h00[i*4+0, i*4+2].conj()
        h00[i*4+1, i*4+3] = t2*cmath.exp(-1j*phi)
        h00[i*4+3, i*4+1] = h00[i*4+1, i*4+3].conj()
    for i in range(N-1):
        h00[i*4+3, (i+1)*4+0] = t1
        h00[(i+1)*4+0, i*4+3] = t1
        h00[i*4+2, (i+1)*4+0] = t2*cmath.exp(1j*phi)
        h00[(i+1)*4+0, i*4+2] = h00[i*4+2, (i+1)*4+0].conj()
        h00[i*4+3, (i+1)*4+1] = t2*cmath.exp(1j*phi)
        h00[(i+1)*4+1, i*4+3] = h00[i*4+3, (i+1)*4+1].conj()
    for i in range(N):
        h01[i*4+1, i*4+0] = t1
        h01[i*4+2, i*4+3] = t1
        h01[i*4+0, i*4+0] = t2*cmath.exp(1j*phi)
        h01[i*4+1, i*4+1] = t2*cmath.exp(-1j*phi)
        h01[i*4+2, i*4+2] = t2*cmath.exp(1j*phi)
        h01[i*4+3, i*4+3] = t2*cmath.exp(-1j*phi)
        h01[i*4+1, i*4+3] = t2*cmath.exp(1j*phi)
        h01[i*4+2, i*4+0] = t2*cmath.exp(-1j*phi)
        if i != 0:
            h01[i*4+1, (i-1)*4+3] = t2*cmath.exp(1j*phi)
    for i in range(N-1):
        h01[i*4+2, (i+1)*4+0] = t2*cmath.exp(-1j*phi)
    hamiltonian = h00 + h01*cmath.exp(1j*k) + h01.transpose().conj()*cmath.exp(-1j*k)
    return hamiltonian