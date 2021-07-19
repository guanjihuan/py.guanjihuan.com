# GUAN is an open-source python package developed and maintained by https://www.guanjihuan.com. The primary location of this package is on website https://py.guanjihuan.com.

# calculate reciprocal lattice vectors

import numpy as np
import sympy
from math import *

def calculate_one_dimensional_reciprocal_lattice_vector(a1):
    b1 = 2*pi/a1
    return b1

def calculate_two_dimensional_reciprocal_lattice_vectors(a1, a2):
    a1 = np.array(a1)
    a2 = np.array(a2)
    a1 = np.append(a1, 0)
    a2 = np.append(a2, 0)
    a3 = np.array([0, 0, 1])
    b1 = 2*pi*np.cross(a2, a3)/np.dot(a1, np.cross(a2, a3))
    b2 = 2*pi*np.cross(a3, a1)/np.dot(a1, np.cross(a2, a3))
    b1 = np.delete(b1, 2)
    b2 = np.delete(b2, 2)
    return b1, b2

def calculate_three_dimensional_reciprocal_lattice_vectors(a1, a2, a3):
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)
    b1 = 2*pi*np.cross(a2, a3)/np.dot(a1, np.cross(a2, a3))
    b2 = 2*pi*np.cross(a3, a1)/np.dot(a1, np.cross(a2, a3))
    b3 = 2*pi*np.cross(a1, a2)/np.dot(a1, np.cross(a2, a3))
    return b1, b2, b3

def calculate_one_dimensional_reciprocal_lattice_vector_with_sympy(a1):
    b1 = 2*sympy.pi/a1
    return b1

def calculate_two_dimensional_reciprocal_lattice_vectors_with_sympy(a1, a2):
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
    cross_a2_a3 = a2.cross(a3)
    cross_a3_a1 = a3.cross(a1)
    cross_a1_a2 = a1.cross(a2)
    b1 = 2*sympy.pi*cross_a2_a3/a1.dot(cross_a2_a3)
    b2 = 2*sympy.pi*cross_a3_a1/a1.dot(cross_a2_a3)
    b3 = 2*sympy.pi*cross_a1_a2/a1.dot(cross_a2_a3)
    return b1, b2, b3