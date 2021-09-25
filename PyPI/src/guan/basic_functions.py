# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com. The primary location of this package is on website https://py.guanjihuan.com.

# basic functions

import numpy as np

## test

def test():
    print('\nSuccess in the installation of Guan package!\n')

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

## calculate reciprocal lattice vectors

def calculate_one_dimensional_reciprocal_lattice_vector(a1):
    b1 = 2*np.pi/a1
    return b1

def calculate_two_dimensional_reciprocal_lattice_vectors(a1, a2):
    a1 = np.array(a1)
    a2 = np.array(a2)
    a1 = np.append(a1, 0)
    a2 = np.append(a2, 0)
    a3 = np.array([0, 0, 1])
    b1 = 2*np.pi*np.cross(a2, a3)/np.dot(a1, np.cross(a2, a3))
    b2 = 2*np.pi*np.cross(a3, a1)/np.dot(a1, np.cross(a2, a3))
    b1 = np.delete(b1, 2)
    b2 = np.delete(b2, 2)
    return b1, b2

def calculate_three_dimensional_reciprocal_lattice_vectors(a1, a2, a3):
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)
    b1 = 2*np.pi*np.cross(a2, a3)/np.dot(a1, np.cross(a2, a3))
    b2 = 2*np.pi*np.cross(a3, a1)/np.dot(a1, np.cross(a2, a3))
    b3 = 2*np.pi*np.cross(a1, a2)/np.dot(a1, np.cross(a2, a3))
    return b1, b2, b3

def calculate_one_dimensional_reciprocal_lattice_vector_with_sympy(a1):
    import sympy
    b1 = 2*sympy.pi/a1
    return b1

def calculate_two_dimensional_reciprocal_lattice_vectors_with_sympy(a1, a2):
    import sympy
    a1 = sympy.Matrix(1, 3, [a1[0], a1[1], 0])
    a2 = sympy.Matrix(1, 3, [a2[0], a2[1], 0])
    a3 = sympy.Matrix(1, 3, [0, 0, 1])
    cross_a2_a3 = a2.cross(a3)
    cross_a3_a1 = a3.cross(a1)
    b1 = 2*sympy.pi*cross_a2_a3/a1.dot(cross_a2_a3)
    b2 = 2*sympy.pi*cross_a3_a1/a1.dot(cross_a2_a3)
    b1 = sympy.Matrix(1, 2, [b1[0], b1[1]])
    b2 = sympy.Matrix(1, 2, [b2[0], b2[1]])
    return b1, b2

def calculate_three_dimensional_reciprocal_lattice_vectors_with_sympy(a1, a2, a3):
    import sympy
    cross_a2_a3 = a2.cross(a3)
    cross_a3_a1 = a3.cross(a1)
    cross_a1_a2 = a1.cross(a2)
    b1 = 2*sympy.pi*cross_a2_a3/a1.dot(cross_a2_a3)
    b2 = 2*sympy.pi*cross_a3_a1/a1.dot(cross_a2_a3)
    b3 = 2*sympy.pi*cross_a1_a2/a1.dot(cross_a2_a3)
    return b1, b2, b3