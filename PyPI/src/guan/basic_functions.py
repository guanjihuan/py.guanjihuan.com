# GUAN is an open-source python package developed and maintained by https://www.guanjihuan.com. The primary location of this package is on website https://py.guanjihuan.com.

# basic functions

import numpy as np

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