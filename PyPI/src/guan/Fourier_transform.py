# GUAN is an open-source python package developed and maintained by https://www.guanjihuan.com. The primary location of this package is on website https://py.guanjihuan.com.

# Fourier_transform

import numpy as np
import cmath

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