# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com/about. The primary location of this package is on website https://py.guanjihuan.com.

# With this package, you can calculate band structures, density of states, quantum transport and topological invariant of tight-binding models by invoking the functions you need. Other frequently used functions are also integrated in this package, such as file reading/writing, figure plotting, data processing.

# The current version is guan-0.0.135, updated on September 12, 2022.

# Installation: pip install --upgrade guan

# Modules:

# # Module 1: basic functions
# # Module 2: Fourier transform
# # Module 3: Hamiltonian of finite size systems
# # Module 4: Hamiltonian of models in the reciprocal space
# # Module 5: band structures and wave functions
# # Module 6: Green functions
# # Module 7: density of states
# # Module 8: quantum transport
# # Module 9: topological invariant
# # Module 10: read and write
# # Module 11: plot figures
# # Module 12: data processing
# # Module 13: others



# import necessary packages

import numpy as np
import math
import cmath
import copy
import guan



# Module 1: basic functions

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
    return np.kron(guan.sigma_0(), guan.sigma_0())

def sigma_0x():
    return np.kron(guan.sigma_0(), guan.sigma_x())

def sigma_0y():
    return np.kron(guan.sigma_0(), guan.sigma_y())

def sigma_0z():
    return np.kron(guan.sigma_0(), guan.sigma_z())

def sigma_x0():
    return np.kron(guan.sigma_x(), guan.sigma_0())

def sigma_xx():
    return np.kron(guan.sigma_x(), guan.sigma_x())

def sigma_xy():
    return np.kron(guan.sigma_x(), guan.sigma_y())

def sigma_xz():
    return np.kron(guan.sigma_x(), guan.sigma_z())

def sigma_y0():
    return np.kron(guan.sigma_y(), guan.sigma_0())

def sigma_yx():
    return np.kron(guan.sigma_y(), guan.sigma_x())

def sigma_yy():
    return np.kron(guan.sigma_y(), guan.sigma_y())

def sigma_yz():
    return np.kron(guan.sigma_y(), guan.sigma_z())

def sigma_z0():
    return np.kron(guan.sigma_z(), guan.sigma_0())

def sigma_zx():
    return np.kron(guan.sigma_z(), guan.sigma_x())

def sigma_zy():
    return np.kron(guan.sigma_z(), guan.sigma_y())

def sigma_zz():
    return np.kron(guan.sigma_z(), guan.sigma_z())










# Module 2: Fourier_transform

# Fourier transform for discrete lattices

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

def one_dimensional_fourier_transform_with_k(unit_cell, hopping):
    import functools
    hamiltonian_function = functools.partial(guan.one_dimensional_fourier_transform, unit_cell=unit_cell, hopping=hopping)
    return hamiltonian_function

def two_dimensional_fourier_transform_for_square_lattice_with_k1_k2(unit_cell, hopping_1, hopping_2):
    import functools
    hamiltonian_function = functools.partial(guan.two_dimensional_fourier_transform_for_square_lattice, unit_cell=unit_cell, hopping_1=hopping_1, hopping_2=hopping_2)
    return hamiltonian_function

def three_dimensional_fourier_transform_for_cubic_lattice_with_k1_k2_k3(unit_cell, hopping_1, hopping_2, hopping_3):
    import functools
    hamiltonian_function = functools.partial(guan.three_dimensional_fourier_transform_for_cubic_lattice, unit_cell=unit_cell, hopping_1=hopping_1, hopping_2=hopping_2, hopping_3=hopping_3)
    return hamiltonian_function

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










# Module 3: Hamiltonian of finite size systems

def hamiltonian_of_finite_size_system_along_one_direction(N, on_site=0, hopping=1, period=0):
    on_site = np.array(on_site)
    hopping = np.array(hopping)
    if on_site.shape==():
        dim = 1
    else:
        dim = on_site.shape[0]
    hamiltonian = np.zeros((N*dim, N*dim), dtype=complex)
    for i0 in range(N):
        hamiltonian[i0*dim+0:i0*dim+dim, i0*dim+0:i0*dim+dim] = on_site
    for i0 in range(N-1):
        hamiltonian[i0*dim+0:i0*dim+dim, (i0+1)*dim+0:(i0+1)*dim+dim] = hopping
        hamiltonian[(i0+1)*dim+0:(i0+1)*dim+dim, i0*dim+0:i0*dim+dim] = hopping.transpose().conj()
    if period == 1:
        hamiltonian[(N-1)*dim+0:(N-1)*dim+dim, 0:dim] = hopping
        hamiltonian[0:dim, (N-1)*dim+0:(N-1)*dim+dim] = hopping.transpose().conj()
    return hamiltonian

def hamiltonian_of_finite_size_system_along_two_directions_for_square_lattice(N1, N2, on_site=0, hopping_1=1, hopping_2=1, period_1=0, period_2=0):
    on_site = np.array(on_site)
    hopping_1 = np.array(hopping_1)
    hopping_2 = np.array(hopping_2)
    if on_site.shape==():
        dim = 1
    else:
        dim = on_site.shape[0]
    hamiltonian = np.zeros((N1*N2*dim, N1*N2*dim), dtype=complex)    
    for i1 in range(N1):
        for i2 in range(N2):
            hamiltonian[i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim, i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim] = on_site
    for i1 in range(N1-1):
        for i2 in range(N2):
            hamiltonian[i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim, (i1+1)*N2*dim+i2*dim+0:(i1+1)*N2*dim+i2*dim+dim] = hopping_1
            hamiltonian[(i1+1)*N2*dim+i2*dim+0:(i1+1)*N2*dim+i2*dim+dim, i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim] = hopping_1.transpose().conj()
    for i1 in range(N1):
        for i2 in range(N2-1):
            hamiltonian[i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim, i1*N2*dim+(i2+1)*dim+0:i1*N2*dim+(i2+1)*dim+dim] = hopping_2
            hamiltonian[i1*N2*dim+(i2+1)*dim+0:i1*N2*dim+(i2+1)*dim+dim, i1*N2*dim+i2*dim+0:i1*N2*dim+i2*dim+dim] = hopping_2.transpose().conj()
    if period_1 == 1:
        for i2 in range(N2):
            hamiltonian[(N1-1)*N2*dim+i2*dim+0:(N1-1)*N2*dim+i2*dim+dim, i2*dim+0:i2*dim+dim] = hopping_1
            hamiltonian[i2*dim+0:i2*dim+dim, (N1-1)*N2*dim+i2*dim+0:(N1-1)*N2*dim+i2*dim+dim] = hopping_1.transpose().conj()
    if period_2 == 1:
        for i1 in range(N1):
            hamiltonian[i1*N2*dim+(N2-1)*dim+0:i1*N2*dim+(N2-1)*dim+dim, i1*N2*dim+0:i1*N2*dim+dim] = hopping_2
            hamiltonian[i1*N2*dim+0:i1*N2*dim+dim, i1*N2*dim+(N2-1)*dim+0:i1*N2*dim+(N2-1)*dim+dim] = hopping_2.transpose().conj()
    return hamiltonian

def hamiltonian_of_finite_size_system_along_three_directions_for_cubic_lattice(N1, N2, N3, on_site=0, hopping_1=1, hopping_2=1, hopping_3=1, period_1=0, period_2=0, period_3=0):
    on_site = np.array(on_site)
    hopping_1 = np.array(hopping_1)
    hopping_2 = np.array(hopping_2)
    hopping_3 = np.array(hopping_3)
    if on_site.shape==():
        dim = 1
    else:
        dim = on_site.shape[0]
    hamiltonian = np.zeros((N1*N2*N3*dim, N1*N2*N3*dim), dtype=complex) 
    for i1 in range(N1):
        for i2 in range(N2):
            for i3 in range(N3):
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim, i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim] = on_site
    for i1 in range(N1-1):
        for i2 in range(N2):
            for i3 in range(N3):
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim, (i1+1)*N2*N3*dim+i2*N3*dim+i3*dim+0:(i1+1)*N2*N3*dim+i2*N3*dim+i3*dim+dim] = hopping_1
                hamiltonian[(i1+1)*N2*N3*dim+i2*N3*dim+i3*dim+0:(i1+1)*N2*N3*dim+i2*N3*dim+i3*dim+dim, i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim] = hopping_1.transpose().conj()
    for i1 in range(N1):
        for i2 in range(N2-1):
            for i3 in range(N3):
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim, i1*N2*N3*dim+(i2+1)*N3*dim+i3*dim+0:i1*N2*N3*dim+(i2+1)*N3*dim+i3*dim+dim] = hopping_2
                hamiltonian[i1*N2*N3*dim+(i2+1)*N3*dim+i3*dim+0:i1*N2*N3*dim+(i2+1)*N3*dim+i3*dim+dim, i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim] = hopping_2.transpose().conj()
    for i1 in range(N1):
        for i2 in range(N2):
            for i3 in range(N3-1):
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim, i1*N2*N3*dim+i2*N3*dim+(i3+1)*dim+0:i1*N2*N3*dim+i2*N3*dim+(i3+1)*dim+dim] = hopping_3
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+(i3+1)*dim+0:i1*N2*N3*dim+i2*N3*dim+(i3+1)*dim+dim, i1*N2*N3*dim+i2*N3*dim+i3*dim+0:i1*N2*N3*dim+i2*N3*dim+i3*dim+dim] = hopping_3.transpose().conj()
    if period_1 == 1:
        for i2 in range(N2):
            for i3 in range(N3):
                hamiltonian[(N1-1)*N2*N3*dim+i2*N3*dim+i3*dim+0:(N1-1)*N2*N3*dim+i2*N3*dim+i3*dim+dim, i2*N3*dim+i3*dim+0:i2*N3*dim+i3*dim+dim] = hopping_1
                hamiltonian[i2*N3*dim+i3*dim+0:i2*N3*dim+i3*dim+dim, (N1-1)*N2*N3*dim+i2*N3*dim+i3*dim+0:(N1-1)*N2*N3*dim+i2*N3*dim+i3*dim+dim] = hopping_1.transpose().conj()
    if period_2 == 1:
        for i1 in range(N1):
            for i3 in range(N3):
                hamiltonian[i1*N2*N3*dim+(N2-1)*N3*dim+i3*dim+0:i1*N2*N3*dim+(N2-1)*N3*dim+i3*dim+dim, i1*N2*N3*dim+i3*dim+0:i1*N2*N3*dim+i3*dim+dim] = hopping_2
                hamiltonian[i1*N2*N3*dim+i3*dim+0:i1*N2*N3*dim+i3*dim+dim, i1*N2*N3*dim+(N2-1)*N3*dim+i3*dim+0:i1*N2*N3*dim+(N2-1)*N3*dim+i3*dim+dim] = hopping_2.transpose().conj()
    if period_3 == 1:
        for i1 in range(N1):
            for i2 in range(N2):
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+(N3-1)*dim+0:i1*N2*N3*dim+i2*N3*dim+(N3-1)*dim+dim, i1*N2*N3*dim+i2*N3*dim+0:i1*N2*N3*dim+i2*N3*dim+dim] = hopping_3
                hamiltonian[i1*N2*N3*dim+i2*N3*dim+0:i1*N2*N3*dim+i2*N3*dim+dim, i1*N2*N3*dim+i2*N3*dim+(N3-1)*dim+0:i1*N2*N3*dim+i2*N3*dim+(N3-1)*dim+dim] = hopping_3.transpose().conj()
    return hamiltonian

def hamiltonian_of_finite_size_ssh_model(N, v=0.6, w=1, onsite_1=0, onsite_2=0, period=1):
    hamiltonian = np.zeros((2*N, 2*N))
    for i in range(N):
        hamiltonian[i*2+0, i*2+0] = onsite_1
        hamiltonian[i*2+1, i*2+1] = onsite_2
        hamiltonian[i*2+0, i*2+1] = v
        hamiltonian[i*2+1, i*2+0] = v
    for i in range(N-1):
        hamiltonian[i*2+1, (i+1)*2+0] = w
        hamiltonian[(i+1)*2+0, i*2+1] = w
    if period==1:
        hamiltonian[0, 2*N-1] = w
        hamiltonian[2*N-1, 0] = w
    return hamiltonian

def get_hopping_term_of_graphene_ribbon_along_zigzag_direction(N, eta=0):
    hopping = np.zeros((4*N, 4*N), dtype=complex)
    for i0 in range(N):
        hopping[4*i0+0, 4*i0+0] = eta
        hopping[4*i0+1, 4*i0+1] = eta
        hopping[4*i0+2, 4*i0+2] = eta
        hopping[4*i0+3, 4*i0+3] = eta
        hopping[4*i0+1, 4*i0+0] = 1
        hopping[4*i0+2, 4*i0+3] = 1
    return hopping

def hamiltonian_of_finite_size_system_along_two_directions_for_graphene(N1, N2, period_1=0, period_2=0):
    on_site = guan.hamiltonian_of_finite_size_system_along_one_direction(4)
    hopping_1 = guan.get_hopping_term_of_graphene_ribbon_along_zigzag_direction(1)
    hopping_2 = np.zeros((4, 4), dtype=complex)
    hopping_2[3, 0] = 1
    hamiltonian = guan.hamiltonian_of_finite_size_system_along_two_directions_for_square_lattice(N1, N2, on_site, hopping_1, hopping_2, period_1, period_2)
    return hamiltonian

def get_onsite_and_hopping_terms_of_2d_effective_graphene_along_one_direction(qy, t=1, staggered_potential=0, eta=0, valley_index=0):
    constant = -np.sqrt(3)/2
    h00 = np.zeros((2, 2), dtype=complex)
    h00[0, 0] = staggered_potential
    h00[1, 1] = -staggered_potential
    h00[0, 1] = -1j*constant*t*np.sin(qy)
    h00[1, 0] = 1j*constant*t*np.sin(qy)
    h01 = np.zeros((2, 2), dtype=complex)
    h01[0, 0] = eta
    h01[1, 1] = eta
    if valley_index == 0:
        h01[0, 1] = constant*t*(-1j/2)
        h01[1, 0] = constant*t*(-1j/2)
    else:
        h01[0, 1] = constant*t*(1j/2)
        h01[1, 0] = constant*t*(1j/2)
    return h00, h01

def get_onsite_and_hopping_terms_of_bhz_model(A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01, a=1):
    E_s = C+M-4*(D+B)/(a**2)
    E_p = C-M-4*(D-B)/(a**2)
    V_ss = (D+B)/(a**2)
    V_pp = (D-B)/(a**2)
    V_sp = -1j*A/(2*a)
    H0 = np.zeros((4, 4), dtype=complex)
    H1 = np.zeros((4, 4), dtype=complex)
    H2 = np.zeros((4, 4), dtype=complex)
    H0[0, 0] = E_s
    H0[1, 1] = E_p
    H0[2, 2] = E_s
    H0[3, 3] = E_p
    H1[0, 0] = V_ss
    H1[1, 1] = V_pp
    H1[2, 2] = V_ss
    H1[3, 3] = V_pp
    H1[0, 1] = V_sp
    H1[1, 0] = -np.conj(V_sp)
    H1[2, 3] = np.conj(V_sp)
    H1[3, 2] = -V_sp
    H2[0, 0] = V_ss
    H2[1, 1] = V_pp
    H2[2, 2] = V_ss
    H2[3, 3] = V_pp
    H2[0, 1] = 1j*V_sp
    H2[1, 0] = 1j*np.conj(V_sp)
    H2[2, 3] = -1j*np.conj(V_sp)
    H2[3, 2] = -1j*V_sp
    return H0, H1, H2

def get_onsite_and_hopping_terms_of_half_bhz_model_for_spin_up(A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01, a=1):
    E_s = C+M-4*(D+B)/(a**2)
    E_p = C-M-4*(D-B)/(a**2)
    V_ss = (D+B)/(a**2)
    V_pp = (D-B)/(a**2)
    V_sp = -1j*A/(2*a)
    H0 = np.zeros((2, 2), dtype=complex)
    H1 = np.zeros((2, 2), dtype=complex)
    H2 = np.zeros((2, 2), dtype=complex)
    H0[0, 0] = E_s
    H0[1, 1] = E_p
    H1[0, 0] = V_ss
    H1[1, 1] = V_pp
    H1[0, 1] = V_sp
    H1[1, 0] = -np.conj(V_sp)
    H2[0, 0] = V_ss
    H2[1, 1] = V_pp
    H2[0, 1] = 1j*V_sp
    H2[1, 0] = 1j*np.conj(V_sp)
    return H0, H1, H2

def get_onsite_and_hopping_terms_of_half_bhz_model_for_spin_down(A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01, a=1):
    E_s = C+M-4*(D+B)/(a**2)
    E_p = C-M-4*(D-B)/(a**2)
    V_ss = (D+B)/(a**2)
    V_pp = (D-B)/(a**2)
    V_sp = -1j*A/(2*a)
    H0 = np.zeros((2, 2), dtype=complex)
    H1 = np.zeros((2, 2), dtype=complex)
    H2 = np.zeros((2, 2), dtype=complex)
    H0[0, 0] = E_s
    H0[1, 1] = E_p
    H1[0, 0] = V_ss
    H1[1, 1] = V_pp
    H1[0, 1] = np.conj(V_sp)
    H1[1, 0] = -V_sp
    H2[0, 0] = V_ss
    H2[1, 1] = V_pp
    H2[0, 1] = -1j*np.conj(V_sp)
    H2[1, 0] = -1j*V_sp
    return H0, H1, H2












# Module 4: Hamiltonian of models in the reciprocal space

def hamiltonian_of_simple_chain(k):
    hamiltonian = guan.one_dimensional_fourier_transform(k, unit_cell=0, hopping=1)
    return hamiltonian

def hamiltonian_of_square_lattice(k1, k2):
    hamiltonian = guan.two_dimensional_fourier_transform_for_square_lattice(k1, k2, unit_cell=0, hopping_1=1, hopping_2=1)
    return hamiltonian

def hamiltonian_of_square_lattice_in_quasi_one_dimension(k, N=10, period=0):
    h00 = np.zeros((N, N), dtype=complex)  # hopping in a unit cell
    h01 = np.zeros((N, N), dtype=complex)  # hopping between unit cells
    for i in range(N-1):   
        h00[i, i+1] = 1
        h00[i+1, i] = 1
    if period == 1:
        h00[N-1, 0] = 1
        h00[0, N-1] = 1
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

def hamiltonian_of_graphene(k1, k2, staggered_potential=0, t=1, a=1/math.sqrt(3)):
    h0 = np.zeros((2, 2), dtype=complex)  # mass term
    h1 = np.zeros((2, 2), dtype=complex)  # nearest hopping
    h0[0, 0] = staggered_potential     
    h0[1, 1] = -staggered_potential
    h1[1, 0] = t*(cmath.exp(1j*k2*a)+cmath.exp(1j*math.sqrt(3)/2*k1*a-1j/2*k2*a)+cmath.exp(-1j*math.sqrt(3)/2*k1*a-1j/2*k2*a))   
    h1[0, 1] = h1[1, 0].conj()
    hamiltonian = h0 + h1
    return hamiltonian

def effective_hamiltonian_of_graphene(qx, qy, t=1, staggered_potential=0, valley_index=0):
    hamiltonian = np.zeros((2, 2), dtype=complex)
    hamiltonian[0, 0] = staggered_potential
    hamiltonian[1, 1] = -staggered_potential
    constant = -np.sqrt(3)/2
    if valley_index == 0:
        hamiltonian[0, 1] = constant*t*(qx-1j*qy)
        hamiltonian[1, 0] = constant*t*(qx+1j*qy)
    else:
        hamiltonian[0, 1] = constant*t*(-qx-1j*qy)
        hamiltonian[1, 0] = constant*t*(-qx+1j*qy)
    return hamiltonian

def effective_hamiltonian_of_graphene_after_discretization(qx, qy, t=1, staggered_potential=0, valley_index=0):
    hamiltonian = np.zeros((2, 2), dtype=complex)
    hamiltonian[0, 0] = staggered_potential
    hamiltonian[1, 1] = -staggered_potential
    constant = -np.sqrt(3)/2
    if valley_index == 0:
        hamiltonian[0, 1] = constant*t*(np.sin(qx)-1j*np.sin(qy))
        hamiltonian[1, 0] = constant*t*(np.sin(qx)+1j*np.sin(qy))
    else:
        hamiltonian[0, 1] = constant*t*(-np.sin(qx)-1j*np.sin(qy))
        hamiltonian[1, 0] = constant*t*(-np.sin(qx)+1j*np.sin(qy))
    return hamiltonian

def hamiltonian_of_graphene_with_zigzag_in_quasi_one_dimension(k, N=10, M=0, t=1, period=0):
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
    if period == 1:
        h00[(N-1)*4+3, 0] = t
        h00[0, (N-1)*4+3] = t
    for i in range(N):
        h01[i*4+1, i*4+0] = t
        h01[i*4+2, i*4+3] = t
    hamiltonian = guan.one_dimensional_fourier_transform(k, unit_cell=h00, hopping=h01) 
    return hamiltonian

def hamiltonian_of_haldane_model(k1, k2, M=2/3, t1=1, t2=1/3, phi=math.pi/4, a=1/math.sqrt(3)):
    h0 = np.zeros((2, 2), dtype=complex)  # mass term
    h1 = np.zeros((2, 2), dtype=complex)  # nearest hopping
    h2 = np.zeros((2, 2), dtype=complex)  # next nearest hopping
    h0[0, 0] = M
    h0[1, 1] = -M
    h1[1, 0] = t1*(cmath.exp(1j*k2*a)+cmath.exp(1j*math.sqrt(3)/2*k1*a-1j/2*k2*a)+cmath.exp(-1j*math.sqrt(3)/2*k1*a-1j/2*k2*a))
    h1[0, 1] = h1[1, 0].conj()
    h2[0, 0] = t2*cmath.exp(-1j*phi)*(cmath.exp(1j*math.sqrt(3)*k1*a)+cmath.exp(-1j*math.sqrt(3)/2*k1*a+1j*3/2*k2*a)+cmath.exp(-1j*math.sqrt(3)/2*k1*a-1j*3/2*k2*a))
    h2[1, 1] = t2*cmath.exp(1j*phi)*(cmath.exp(1j*math.sqrt(3)*k1*a)+cmath.exp(-1j*math.sqrt(3)/2*k1*a+1j*3/2*k2*a)+cmath.exp(-1j*math.sqrt(3)/2*k1*a-1j*3/2*k2*a))
    hamiltonian = h0 + h1 + h2 + h2.transpose().conj()
    return hamiltonian

def hamiltonian_of_haldane_model_in_quasi_one_dimension(k, N=10, M=2/3, t1=1, t2=1/3, phi=math.pi/4, period=0):
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
    if period == 1:
        h00[(N-1)*4+3, 0] = t1
        h00[0, (N-1)*4+3] = t1
        h00[(N-1)*4+2, 0] = t2*cmath.exp(1j*phi)
        h00[0, (N-1)*4+2] = h00[(N-1)*4+2, 0].conj()
        h00[(N-1)*4+3, 1] = t2*cmath.exp(1j*phi)
        h00[1, (N-1)*4+3] = h00[(N-1)*4+3, 1].conj()
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

def hamiltonian_of_one_QAH_model(k1, k2, t1=1, t2=1, t3=0.5, m=-1):
    hamiltonian = np.zeros((2, 2), dtype=complex)
    hamiltonian[0, 1] = 2*t1*math.cos(k1)-1j*2*t1*math.cos(k2)
    hamiltonian[1, 0] = 2*t1*math.cos(k1)+1j*2*t1*math.cos(k2)
    hamiltonian[0, 0] = m+2*t3*math.sin(k1)+2*t3*math.sin(k2)+2*t2*math.cos(k1+k2)
    hamiltonian[1, 1] = -(m+2*t3*math.sin(k1)+2*t3*math.sin(k2)+2*t2*math.cos(k1+k2))
    return hamiltonian

def hamiltonian_of_bhz_model(kx, ky, A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01):
    hamiltonian = np.zeros((4, 4), dtype=complex)
    varepsilon = C-2*D*(2-math.cos(kx)-math.cos(ky))
    d3 = -2*B*(2-(M/2/B)-math.cos(kx)-math.cos(ky))
    d1_d2 = A*(math.sin(kx)+1j*math.sin(ky))
    hamiltonian[0, 0] = varepsilon+d3
    hamiltonian[1, 1] = varepsilon-d3
    hamiltonian[0, 1] = np.conj(d1_d2)
    hamiltonian[1, 0] = d1_d2
    hamiltonian[2, 2] = varepsilon+d3
    hamiltonian[3, 3] = varepsilon-d3
    hamiltonian[2, 3] = -d1_d2 
    hamiltonian[3, 2] = -np.conj(d1_d2)
    return hamiltonian

def hamiltonian_of_half_bhz_model_for_spin_up(kx, ky, A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01):
    hamiltonian = np.zeros((2, 2), dtype=complex)
    varepsilon = C-2*D*(2-math.cos(kx)-math.cos(ky))
    d3 = -2*B*(2-(M/2/B)-math.cos(kx)-math.cos(ky))
    d1_d2 = A*(math.sin(kx)+1j*math.sin(ky))
    hamiltonian[0, 0] = varepsilon+d3
    hamiltonian[1, 1] = varepsilon-d3
    hamiltonian[0, 1] = np.conj(d1_d2)
    hamiltonian[1, 0] = d1_d2
    return hamiltonian

def hamiltonian_of_half_bhz_model_for_spin_down(kx, ky, A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01):
    hamiltonian = np.zeros((2, 2), dtype=complex)
    varepsilon = C-2*D*(2-math.cos(kx)-math.cos(ky))
    d3 = -2*B*(2-(M/2/B)-math.cos(kx)-math.cos(ky))
    d1_d2 = A*(math.sin(kx)+1j*math.sin(ky))
    hamiltonian[0, 0] = varepsilon+d3
    hamiltonian[1, 1] = varepsilon-d3
    hamiltonian[0, 1] = -d1_d2 
    hamiltonian[1, 0] = -np.conj(d1_d2)
    return hamiltonian

def hamiltonian_of_bbh_model(kx, ky, gamma_x=0.5, gamma_y=0.5, lambda_x=1, lambda_y=1):
    # label of atoms in a unit cell
    # (2) —— (0)
    #  |      |
    # (1) —— (3)   
    hamiltonian = np.zeros((4, 4), dtype=complex)
    hamiltonian[0, 2] = gamma_x+lambda_x*cmath.exp(1j*kx)
    hamiltonian[1, 3] = gamma_x+lambda_x*cmath.exp(-1j*kx)
    hamiltonian[0, 3] = gamma_y+lambda_y*cmath.exp(1j*ky)
    hamiltonian[1, 2] = -gamma_y-lambda_y*cmath.exp(-1j*ky)
    hamiltonian[2, 0] = np.conj(hamiltonian[0, 2])
    hamiltonian[3, 1] = np.conj(hamiltonian[1, 3])
    hamiltonian[3, 0] = np.conj(hamiltonian[0, 3])
    hamiltonian[2, 1] = np.conj(hamiltonian[1, 2]) 
    return hamiltonian

def hamiltonian_of_kagome_lattice(kx, ky, t=1):
    k1_dot_a1 = kx
    k2_dot_a2 = kx/2+ky*math.sqrt(3)/2
    k3_dot_a3 = -kx/2+ky*math.sqrt(3)/2
    hamiltonian = np.zeros((3, 3), dtype=complex)
    hamiltonian[0, 1] = 2*math.cos(k1_dot_a1)
    hamiltonian[0, 2] = 2*math.cos(k2_dot_a2)
    hamiltonian[1, 2] = 2*math.cos(k3_dot_a3)
    hamiltonian = hamiltonian + hamiltonian.transpose().conj()
    hamiltonian = -t*hamiltonian
    return hamiltonian









# Module 5: band structures and wave functions

## band structures

def calculate_eigenvalue(hamiltonian):
    if np.array(hamiltonian).shape==():
        eigenvalue = np.real(hamiltonian)
    else:
        eigenvalue, eigenvector = np.linalg.eigh(hamiltonian)
    return eigenvalue

def calculate_eigenvalue_with_one_parameter(x_array, hamiltonian_function, print_show=0):
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
            if print_show==1:
                print(x0)
            hamiltonian = hamiltonian_function(x0)
            eigenvalue, eigenvector = np.linalg.eigh(hamiltonian)
            eigenvalue_array[i0, :] = eigenvalue
            i0 += 1
    return eigenvalue_array

def calculate_eigenvalue_with_two_parameters(x_array, y_array, hamiltonian_function, print_show=0, print_show_more=0):  
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
            if print_show==1:
                print(y0)
            for x0 in x_array:
                if print_show_more==1:
                    print(x0)
                hamiltonian = hamiltonian_function(x0, y0)
                eigenvalue, eigenvector = np.linalg.eigh(hamiltonian)
                eigenvalue_array[i0, j0, :] = eigenvalue
                j0 += 1
            i0 += 1
    return eigenvalue_array

## wave functions

def calculate_eigenvector(hamiltonian):
    eigenvalue, eigenvector = np.linalg.eigh(hamiltonian) 
    return eigenvector

## find vector with the same gauge

def find_vector_with_the_same_gauge_with_binary_search(vector_target, vector_ref, show_error=1, show_times=0, show_phase=0, n_test=1000, precision=1e-6):
    phase_1_pre = 0
    phase_2_pre = np.pi
    for i0 in range(n_test):
        test_1 = np.sum(np.abs(vector_target*cmath.exp(1j*phase_1_pre) - vector_ref))
        test_2 = np.sum(np.abs(vector_target*cmath.exp(1j*phase_2_pre) - vector_ref))
        if test_1 < precision:
            phase = phase_1_pre
            if show_times==1:
                print('Binary search times=', i0)
            break
        if i0 == n_test-1:
            phase = phase_1_pre
            if show_error==1:
                print('Gauge not found with binary search times=', i0)
        if test_1 < test_2:
            if i0 == 0:
                phase_1 = phase_1_pre-(phase_2_pre-phase_1_pre)/2
                phase_2 = phase_1_pre+(phase_2_pre-phase_1_pre)/2
            else:
                phase_1 = phase_1_pre
                phase_2 = phase_1_pre+(phase_2_pre-phase_1_pre)/2
        else:
            if i0 == 0:
                phase_1 = phase_2_pre-(phase_2_pre-phase_1_pre)/2
                phase_2 = phase_2_pre+(phase_2_pre-phase_1_pre)/2
            else:
                phase_1 = phase_2_pre-(phase_2_pre-phase_1_pre)/2
                phase_2 = phase_2_pre 
        phase_1_pre = phase_1
        phase_2_pre = phase_2
    vector_target = vector_target*cmath.exp(1j*phase)
    if show_phase==1:
        print('Phase=', phase)   
    return vector_target

def find_vector_with_fixed_gauge_by_making_one_component_real(vector, precision=0.005, index=None):
    vector = np.array(vector)
    if index == None:
        index = np.argmax(np.abs(vector))
    sign_pre = np.sign(np.imag(vector[index]))
    for phase in np.arange(0, 2*np.pi, precision):
        sign =  np.sign(np.imag(vector[index]*cmath.exp(1j*phase)))
        if np.abs(np.imag(vector[index]*cmath.exp(1j*phase))) < 1e-9 or sign == -sign_pre:
            break
        sign_pre = sign
    vector = vector*cmath.exp(1j*phase)
    if np.real(vector[index]) < 0:
        vector = -vector
    return vector 

def find_vector_array_with_fixed_gauge_by_making_one_component_real(vector_array, precision=0.005):
    vector_sum = 0
    Num_k = np.array(vector_array).shape[0]
    for i0 in range(Num_k):
        vector_sum += np.abs(vector_array[i0])
    index = np.argmax(np.abs(vector_sum))
    for i0 in range(Num_k):
        vector_array[i0] = guan.find_vector_with_fixed_gauge_by_making_one_component_real(vector_array[i0], precision=precision, index=index)
    return vector_array

def rotation_of_degenerate_vectors(vector1, vector2, index1=None, index2=None, precision=0.01, criterion=0.01, show_theta=0):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    if index1 == None:
        index1 = np.argmax(np.abs(vector1))
    if index2 == None:
        index2 = np.argmax(np.abs(vector2))
    if np.abs(vector1[index2])>criterion or np.abs(vector2[index1])>criterion:
        for theta in np.arange(0, 2*math.pi, precision):
            if show_theta==1:
                print(theta)
            for phi1 in np.arange(0, 2*math.pi, precision):
                for phi2 in np.arange(0, 2*math.pi, precision):
                    vector1_test = cmath.exp(1j*phi1)*vector1*math.cos(theta)+cmath.exp(1j*phi2)*vector2*math.sin(theta)
                    vector2_test = -cmath.exp(-1j*phi2)*vector1*math.sin(theta)+cmath.exp(-1j*phi1)*vector2*math.cos(theta)
                    if np.abs(vector1_test[index2])<criterion and np.abs(vector2_test[index1])<criterion:
                        vector1 = vector1_test
                        vector2 = vector2_test
                        break
                if np.abs(vector1_test[index2])<criterion and np.abs(vector2_test[index1])<criterion:
                    break
            if np.abs(vector1_test[index2])<criterion and np.abs(vector2_test[index1])<criterion:
                break
    return vector1, vector2

def rotation_of_degenerate_vectors_array(vector1_array, vector2_array, precision=0.01, criterion=0.01, show_theta=0):
    Num_k = np.array(vector1_array).shape[0]
    vector1_sum = 0
    for i0 in range(Num_k):
        vector1_sum += np.abs(vector1_array[i0])
    index1 = np.argmax(np.abs(vector1_sum))
    vector2_sum = 0
    for i0 in range(Num_k):
        vector2_sum += np.abs(vector2_array[i0])
    index2 = np.argmax(np.abs(vector2_sum))
    for i0 in range(Num_k):
        vector1_array[i0], vector2_array[i0] = guan.rotation_of_degenerate_vectors(vector1=vector1_array[i0], vector2=vector2_array[i0], index1=index1, index2=index2, precision=precision, criterion=criterion, show_theta=show_theta)
    return vector1_array, vector2_array







# Module 6: Green functions

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

def transfer_matrix(fermi_energy, h00, h01):
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    transfer = np.zeros((2*dim, 2*dim), dtype=complex)
    transfer[0:dim, 0:dim] = np.dot(np.linalg.inv(h01), fermi_energy*np.identity(dim)-h00)
    transfer[0:dim, dim:2*dim] = np.dot(-1*np.linalg.inv(h01), h01.transpose().conj())
    transfer[dim:2*dim, 0:dim] = np.identity(dim)
    transfer[dim:2*dim, dim:2*dim] = 0
    return transfer

def surface_green_function_of_lead(fermi_energy, h00, h01):
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    fermi_energy = fermi_energy+1e-9*1j
    transfer = transfer_matrix(fermi_energy, h00, h01)
    eigenvalue, eigenvector = np.linalg.eig(transfer)
    ind = np.argsort(np.abs(eigenvalue))
    temp = np.zeros((2*dim, 2*dim), dtype=complex)
    i0 = 0
    for ind0 in ind:
        temp[:, i0] = eigenvector[:, ind0]
        i0 += 1
    s1 = temp[dim:2*dim, 0:dim]
    s2 = temp[0:dim, 0:dim]
    s3 = temp[dim:2*dim, dim:2*dim]
    s4 = temp[0:dim, dim:2*dim]
    right_lead_surface = np.linalg.inv(fermi_energy*np.identity(dim)-h00-np.dot(np.dot(h01, s2), np.linalg.inv(s1)))
    left_lead_surface = np.linalg.inv(fermi_energy*np.identity(dim)-h00-np.dot(np.dot(h01.transpose().conj(), s3), np.linalg.inv(s4)))
    return right_lead_surface, left_lead_surface

def self_energy_of_lead(fermi_energy, h00, h01):
    h01 = np.array(h01)
    right_lead_surface, left_lead_surface = guan.surface_green_function_of_lead(fermi_energy, h00, h01)
    right_self_energy = np.dot(np.dot(h01, right_lead_surface), h01.transpose().conj())
    left_self_energy = np.dot(np.dot(h01.transpose().conj(), left_lead_surface), h01)
    gamma_right = (right_self_energy - right_self_energy.transpose().conj())*1j
    gamma_left = (left_self_energy - left_self_energy.transpose().conj())*1j
    return right_self_energy, left_self_energy, gamma_right, gamma_left

def self_energy_of_lead_with_h_LC_and_h_CR(fermi_energy, h00, h01, h_LC, h_CR):
    h_LC = np.array(h_LC)
    h_CR = np.array(h_CR)
    right_lead_surface, left_lead_surface = guan.surface_green_function_of_lead(fermi_energy, h00, h01)
    right_self_energy = np.dot(np.dot(h_CR, right_lead_surface), h_CR.transpose().conj())
    left_self_energy = np.dot(np.dot(h_LC.transpose().conj(), left_lead_surface), h_LC)
    gamma_right = (right_self_energy - right_self_energy.transpose().conj())*1j
    gamma_left = (left_self_energy - left_self_energy.transpose().conj())*1j
    return right_self_energy, left_self_energy, gamma_right, gamma_left

def self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00, h01, h_lead_to_center):
    h_lead_to_center = np.array(h_lead_to_center)
    right_lead_surface, left_lead_surface = guan.surface_green_function_of_lead(fermi_energy, h00, h01)
    self_energy = np.dot(np.dot(h_lead_to_center.transpose().conj(), right_lead_surface), h_lead_to_center)
    gamma = (self_energy - self_energy.transpose().conj())*1j
    return self_energy, gamma

def green_function_with_leads(fermi_energy, h00, h01, h_LC, h_CR, center_hamiltonian):
    dim = np.array(center_hamiltonian).shape[0]
    right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead_with_h_LC_and_h_CR(fermi_energy, h00, h01, h_LC, h_CR)
    green = np.linalg.inv(fermi_energy*np.identity(dim)-center_hamiltonian-left_self_energy-right_self_energy)
    return green, gamma_right, gamma_left

def electron_correlation_function_green_n_for_local_current(fermi_energy, h00, h01, h_LC, h_CR, center_hamiltonian):
    right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead_with_h_LC_and_h_CR(fermi_energy, h00, h01, h_LC, h_CR)
    green = guan.green_function(fermi_energy, center_hamiltonian, broadening=0, self_energy=left_self_energy+right_self_energy)
    G_n = np.imag(np.dot(np.dot(green, gamma_left), green.transpose().conj()))
    return G_n










# Module 7: density of states

def total_density_of_states(fermi_energy, hamiltonian, broadening=0.01):
    green = guan.green_function(fermi_energy, hamiltonian, broadening)
    total_dos = -np.trace(np.imag(green))/math.pi
    return total_dos

def total_density_of_states_with_fermi_energy_array(fermi_energy_array, hamiltonian, broadening=0.01, print_show=0):
    dim = np.array(fermi_energy_array).shape[0]
    total_dos_array = np.zeros(dim)
    i0 = 0
    for fermi_energy in fermi_energy_array:
        if print_show == 1:
            print(fermi_energy)
        total_dos_array[i0] = total_density_of_states(fermi_energy, hamiltonian, broadening)
        i0 += 1
    return total_dos_array

def local_density_of_states_for_square_lattice(fermi_energy, hamiltonian, N1, N2, internal_degree=1, broadening=0.01):
    # dim_hamiltonian = N1*N2*internal_degree
    green = guan.green_function(fermi_energy, hamiltonian, broadening)
    local_dos = np.zeros((N2, N1))
    for i1 in range(N1):
        for i2 in range(N2):
            for i in range(internal_degree): 
                local_dos[i2, i1] = local_dos[i2, i1]-np.imag(green[i1*N2*internal_degree+i2*internal_degree+i, i1*N2*internal_degree+i2*internal_degree+i])/math.pi
    return local_dos

def local_density_of_states_for_cubic_lattice(fermi_energy, hamiltonian, N1, N2, N3, internal_degree=1, broadening=0.01):
    # dim_hamiltonian = N1*N2*N3*internal_degree
    green = guan.green_function(fermi_energy, hamiltonian, broadening)
    local_dos = np.zeros((N3, N2, N1))
    for i1 in range(N1):
        for i2 in range(N2):
            for i3 in range(N3):
                for i in range(internal_degree): 
                    local_dos[i3, i2, i1] = local_dos[i3, i2, i1]-np.imag(green[i1*N2*N3*internal_degree+i2*N3*internal_degree+i3*internal_degree+i, i1*N2*N3*internal_degree+i2*N3*internal_degree+i3*internal_degree+i])/math.pi
    return local_dos

def local_density_of_states_for_square_lattice_using_dyson_equation(fermi_energy, h00, h01, N2, N1, internal_degree=1, broadening=0.01):
    # dim_h00 = N2*internal_degree
    local_dos = np.zeros((N2, N1))
    green_11_1 = guan.green_function(fermi_energy, h00, broadening)
    for i1 in range(N1):
        green_nn_n_minus = green_11_1
        green_in_n_minus = green_11_1
        green_ni_n_minus = green_11_1
        green_ii_n_minus = green_11_1
        for i2_0 in range(i1):
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
        if i1!=0:
            green_in_n_minus = green_nn_n
            green_ni_n_minus = green_nn_n
            green_ii_n_minus = green_nn_n
        for size_0 in range(N1-1-i1):
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
            green_ii_n = guan.green_function_ii_n(green_ii_n_minus, green_in_n_minus, h01, green_nn_n, green_ni_n_minus)
            green_ii_n_minus = green_ii_n
            green_in_n = guan.green_function_in_n(green_in_n_minus, h01, green_nn_n)
            green_in_n_minus = green_in_n
            green_ni_n = guan.green_function_ni_n(green_nn_n, h01, green_ni_n_minus)
            green_ni_n_minus = green_ni_n
        for i2 in range(N2):
            for i in range(internal_degree):
                local_dos[i2, i1] = local_dos[i2, i1] - np.imag(green_ii_n_minus[i2*internal_degree+i, i2*internal_degree+i])/math.pi
    return local_dos

def local_density_of_states_for_cubic_lattice_using_dyson_equation(fermi_energy, h00, h01, N3, N2, N1, internal_degree=1, broadening=0.01):
    # dim_h00 = N2*N3*internal_degree
    local_dos = np.zeros((N3, N2, N1))
    green_11_1 = guan.green_function(fermi_energy, h00, broadening)
    for i1 in range(N1):
        green_nn_n_minus = green_11_1
        green_in_n_minus = green_11_1
        green_ni_n_minus = green_11_1
        green_ii_n_minus = green_11_1
        for i1_0 in range(i1):
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
        if i1!=0:
            green_in_n_minus = green_nn_n
            green_ni_n_minus = green_nn_n
            green_ii_n_minus = green_nn_n
        for size_0 in range(N1-1-i1):
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
            green_ii_n = guan.green_function_ii_n(green_ii_n_minus, green_in_n_minus, h01, green_nn_n, green_ni_n_minus)
            green_ii_n_minus = green_ii_n
            green_in_n = guan.green_function_in_n(green_in_n_minus, h01, green_nn_n)
            green_in_n_minus = green_in_n
            green_ni_n = guan.green_function_ni_n(green_nn_n, h01, green_ni_n_minus)
            green_ni_n_minus = green_ni_n
        for i2 in range(N2):
            for i3 in range(N3):
                for i in range(internal_degree):
                    local_dos[i3, i2, i1] = local_dos[i3, i2, i1] -np.imag(green_ii_n_minus[i2*N3*internal_degree+i3*internal_degree+i, i2*N3*internal_degree+i3*internal_degree+i])/math.pi       
    return local_dos

def local_density_of_states_for_square_lattice_with_self_energy_using_dyson_equation(fermi_energy, h00, h01, N2, N1, right_self_energy, left_self_energy, internal_degree=1, broadening=0.01):
    # dim_h00 = N2*internal_degree
    local_dos = np.zeros((N2, N1))
    green_11_1 = guan.green_function(fermi_energy, h00+left_self_energy, broadening)
    for i1 in range(N1):
        green_nn_n_minus = green_11_1
        green_in_n_minus = green_11_1
        green_ni_n_minus = green_11_1
        green_ii_n_minus = green_11_1
        for i2_0 in range(i1):
            if i2_0 == N1-1-1:
                green_nn_n = guan.green_function_nn_n(fermi_energy, h00+right_self_energy, h01, green_nn_n_minus, broadening)
            else:
                green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
        if i1!=0:
            green_in_n_minus = green_nn_n
            green_ni_n_minus = green_nn_n
            green_ii_n_minus = green_nn_n
        for size_0 in range(N1-1-i1):
            if size_0 == N1-1-i1-1:
                green_nn_n = guan.green_function_nn_n(fermi_energy, h00+right_self_energy, h01, green_nn_n_minus, broadening)
            else:
                green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
            green_ii_n = guan.green_function_ii_n(green_ii_n_minus, green_in_n_minus, h01, green_nn_n, green_ni_n_minus)
            green_ii_n_minus = green_ii_n
            green_in_n = guan.green_function_in_n(green_in_n_minus, h01, green_nn_n)
            green_in_n_minus = green_in_n
            green_ni_n = guan.green_function_ni_n(green_nn_n, h01, green_ni_n_minus)
            green_ni_n_minus = green_ni_n
        for i2 in range(N2):
            for i in range(internal_degree):
                local_dos[i2, i1] = local_dos[i2, i1] - np.imag(green_ii_n_minus[i2*internal_degree+i, i2*internal_degree+i])/math.pi
    return local_dos










# Module 8: quantum transport

## conductance

def calculate_conductance(fermi_energy, h00, h01, length=100):
    right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead(fermi_energy, h00, h01)
    for ix in range(length):
        if ix == 0:
            green_nn_n = guan.green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
            green_0n_n = copy.deepcopy(green_nn_n)
        elif ix != length-1:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
        else:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
    conductance = np.trace(np.dot(np.dot(np.dot(gamma_left, green_0n_n), gamma_right), green_0n_n.transpose().conj()))
    return conductance

def calculate_conductance_with_fermi_energy_array(fermi_energy_array, h00, h01, length=100, print_show=0):
    dim = np.array(fermi_energy_array).shape[0]
    conductance_array = np.zeros(dim)
    i0 = 0
    for fermi_energy in fermi_energy_array:
        if print_show == 1:
            print(fermi_energy)
        conductance_array[i0] = np.real(guan.calculate_conductance(fermi_energy, h00, h01, length))
        i0 += 1
    return conductance_array

def calculate_conductance_with_barrier(fermi_energy, h00, h01, length=100, barrier_length=20, barrier_potential=1):
    right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead(fermi_energy, h00, h01)
    dim = np.array(h00).shape[0]
    for ix in range(length):
        if ix == 0:
            green_nn_n = guan.green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
            green_0n_n = copy.deepcopy(green_nn_n)
        elif int(length/2-barrier_length/2)<=ix<int(length/2+barrier_length/2):
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00+barrier_potential*np.identity(dim), h01, green_nn_n, broadening=0) 
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
        elif ix != length-1:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
        else:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
    conductance = np.trace(np.dot(np.dot(np.dot(gamma_left, green_0n_n), gamma_right), green_0n_n.transpose().conj()))
    return conductance

def calculate_conductance_with_disorder(fermi_energy, h00, h01, disorder_intensity=2.0, disorder_concentration=1.0, length=100):
    right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead(fermi_energy, h00, h01)
    dim = np.array(h00).shape[0]
    for ix in range(length):
        disorder = np.zeros((dim, dim))
        for dim0 in range(dim):
            if np.random.uniform(0, 1)<=disorder_concentration:
                disorder[dim0, dim0] = np.random.uniform(-disorder_intensity, disorder_intensity)
        if ix == 0:
            green_nn_n = guan.green_function(fermi_energy, h00+disorder, broadening=0, self_energy=left_self_energy)
            green_0n_n = copy.deepcopy(green_nn_n)
        elif ix != length-1:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00+disorder, h01, green_nn_n, broadening=0)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
        else:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00+disorder, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
    conductance = np.trace(np.dot(np.dot(np.dot(gamma_left, green_0n_n), gamma_right), green_0n_n.transpose().conj()))
    return conductance

def calculate_conductance_with_disorder_intensity_array(fermi_energy, h00, h01, disorder_intensity_array, disorder_concentration=1.0, length=100, calculation_times=1, print_show=0):
    dim = np.array(disorder_intensity_array).shape[0]
    conductance_array = np.zeros(dim)
    i0 = 0
    for disorder_intensity in disorder_intensity_array:
        if print_show == 1:
            print(disorder_intensity)
        for times in range(calculation_times):
            conductance_array[i0] = conductance_array[i0]+np.real(guan.calculate_conductance_with_disorder(fermi_energy, h00, h01, disorder_intensity=disorder_intensity, disorder_concentration=disorder_concentration, length=length))
        i0 += 1
    conductance_array = conductance_array/calculation_times
    return conductance_array

def calculate_conductance_with_disorder_concentration_array(fermi_energy, h00, h01, disorder_concentration_array, disorder_intensity=2.0, length=100, calculation_times=1, print_show=0):
    dim = np.array(disorder_concentration_array).shape[0]
    conductance_array = np.zeros(dim)
    i0 = 0
    for disorder_concentration in disorder_concentration_array:
        if print_show == 1:
            print(disorder_concentration)
        for times in range(calculation_times):
            conductance_array[i0] = conductance_array[i0]+np.real(guan.calculate_conductance_with_disorder(fermi_energy, h00, h01, disorder_intensity=disorder_intensity, disorder_concentration=disorder_concentration, length=length))
        i0 += 1
    conductance_array = conductance_array/calculation_times
    return conductance_array

def calculate_conductance_with_scattering_length_array(fermi_energy, h00, h01, length_array, disorder_intensity=2.0, disorder_concentration=1.0, calculation_times=1, print_show=0):
    dim = np.array(length_array).shape[0]
    conductance_array = np.zeros(dim)
    i0 = 0
    for length in length_array:
        if print_show == 1:
            print(length)
        for times in range(calculation_times):
            conductance_array[i0] = conductance_array[i0]+np.real(guan.calculate_conductance_with_disorder(fermi_energy, h00, h01, disorder_intensity=disorder_intensity, disorder_concentration=disorder_concentration, length=length))
        i0 += 1
    conductance_array = conductance_array/calculation_times
    return conductance_array

## multi-terminal transmission

def get_gamma_array_and_green_for_six_terminal_transmission(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10):
    #   ---------------- Geometry ----------------
    #               lead2         lead3
    #   lead1(L)                          lead4(R)  
    #               lead6         lead5 

    # h00 and h01 in leads
    h00_for_lead_1 = h00_for_lead_4
    h00_for_lead_2 = h00_for_lead_2
    h00_for_lead_3 = h00_for_lead_2
    h00_for_lead_5 = h00_for_lead_2
    h00_for_lead_6 = h00_for_lead_2
    h00_for_lead_4 = h00_for_lead_4
    h01_for_lead_1 = h01_for_lead_4.transpose().conj()
    h01_for_lead_2 = h01_for_lead_2
    h01_for_lead_3 = h01_for_lead_2
    h01_for_lead_4 = h01_for_lead_4
    h01_for_lead_5 = h01_for_lead_2.transpose().conj()
    h01_for_lead_6 = h01_for_lead_2.transpose().conj()
    # hopping matrix from lead to center
    h_lead1_to_center = np.zeros((internal_degree*width, internal_degree*width*length), dtype=complex)
    h_lead2_to_center = np.zeros((internal_degree*width, internal_degree*width*length), dtype=complex)
    h_lead3_to_center = np.zeros((internal_degree*width, internal_degree*width*length), dtype=complex)
    h_lead4_to_center = np.zeros((internal_degree*width, internal_degree*width*length), dtype=complex)
    h_lead5_to_center = np.zeros((internal_degree*width, internal_degree*width*length), dtype=complex)
    h_lead6_to_center = np.zeros((internal_degree*width, internal_degree*width*length), dtype=complex)
    move = moving_step_of_leads # the step of leads 2,3,6,5 moving to center
    h_lead1_to_center[0:internal_degree*width, 0:internal_degree*width] = h01_for_lead_1.transpose().conj()
    h_lead4_to_center[0:internal_degree*width, internal_degree*width*(length-1):internal_degree*width*length] = h01_for_lead_4.transpose().conj()
    for i0 in range(width):
        begin_index = internal_degree*i0+0
        end_index = internal_degree*i0+internal_degree
        h_lead2_to_center[begin_index:end_index, internal_degree*(width*(move+i0)+(width-1))+0:internal_degree*(width*(move+i0)+(width-1))+internal_degree] = h01_for_lead_2.transpose().conj()[begin_index:end_index, begin_index:end_index]
        h_lead3_to_center[begin_index:end_index, internal_degree*(width*(length-move-1-i0)+(width-1))+0:internal_degree*(width*(length-move-1-i0)+(width-1))+internal_degree] = h01_for_lead_3.transpose().conj()[begin_index:end_index, begin_index:end_index]
        h_lead5_to_center[begin_index:end_index, internal_degree*(width*(length-move-1-i0)+0)+0:internal_degree*(width*(length-move-1-i0)+0)+internal_degree] = h01_for_lead_5.transpose().conj()[begin_index:end_index, begin_index:end_index]
        h_lead6_to_center[begin_index:end_index, internal_degree*(width*(i0+move)+0)+0:internal_degree*(width*(i0+move)+0)+internal_degree] = h01_for_lead_6.transpose().conj()[begin_index:end_index, begin_index:end_index]
    # self energy    
    self_energy1, gamma1 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_1, h01_for_lead_1, h_lead1_to_center)
    self_energy2, gamma2 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_2, h01_for_lead_1, h_lead2_to_center)
    self_energy3, gamma3 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_3, h01_for_lead_1, h_lead3_to_center)
    self_energy4, gamma4 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_4, h01_for_lead_1, h_lead4_to_center)
    self_energy5, gamma5 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_5, h01_for_lead_1, h_lead5_to_center)
    self_energy6, gamma6 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_6, h01_for_lead_1, h_lead6_to_center)
    gamma_array = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6]
    # Green function
    green = np.linalg.inv(fermi_energy*np.eye(internal_degree*width*length)-center_hamiltonian-self_energy1-self_energy2-self_energy3-self_energy4-self_energy5-self_energy6)
    return gamma_array, green

def calculate_six_terminal_transmission_matrix(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10):
    gamma_array, green = guan.get_gamma_array_and_green_for_six_terminal_transmission(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width, length, internal_degree, moving_step_of_leads)
    # Transmission
    transmission_matrix = np.zeros((6, 6), dtype=complex)
    channel_lead_4 = guan.calculate_conductance(fermi_energy, h00_for_lead_4, h01_for_lead_4, length=3)
    channel_lead_2 = guan.calculate_conductance(fermi_energy, h00_for_lead_2, h01_for_lead_2, length=3)
    for i0 in range(6):
        for j0 in range(6):
            if j0!=i0:
                transmission_matrix[i0, j0] = np.trace(np.dot(np.dot(np.dot(gamma_array[i0], green), gamma_array[j0]), green.transpose().conj()))
    for i0 in range(6):
        if i0 == 0 or i0 == 3:
            transmission_matrix[i0, i0] = channel_lead_4
        else:
            transmission_matrix[i0, i0] = channel_lead_2
    for i0 in range(6):
        for j0 in range(6):
            if j0!=i0:
                transmission_matrix[i0, i0] = transmission_matrix[i0, i0]-transmission_matrix[i0, j0]
    transmission_matrix = np.real(transmission_matrix)
    return transmission_matrix

def calculate_six_terminal_transmissions_from_lead_1(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10):
    gamma_array, green = guan.get_gamma_array_and_green_for_six_terminal_transmission(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width, length, internal_degree, moving_step_of_leads)
    # Transmission
    transmission_12 = np.real(np.trace(np.dot(np.dot(np.dot(gamma_array[0], green), gamma_array[1]), green.transpose().conj())))
    transmission_13 = np.real(np.trace(np.dot(np.dot(np.dot(gamma_array[0], green), gamma_array[2]), green.transpose().conj())))
    transmission_14 = np.real(np.trace(np.dot(np.dot(np.dot(gamma_array[0], green), gamma_array[3]), green.transpose().conj())))
    transmission_15 = np.real(np.trace(np.dot(np.dot(np.dot(gamma_array[0], green), gamma_array[4]), green.transpose().conj())))
    transmission_16 = np.real(np.trace(np.dot(np.dot(np.dot(gamma_array[0], green), gamma_array[5]), green.transpose().conj())))
    return transmission_12, transmission_13, transmission_14, transmission_15, transmission_16

## scattering matrix

def if_active_channel(k_of_channel):
    if np.abs(np.imag(k_of_channel))<1e-6:
        if_active = 1
    else:
        if_active = 0
    return if_active

def get_k_and_velocity_of_channel(fermi_energy, h00, h01):
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    transfer = guan.transfer_matrix(fermi_energy, h00, h01)
    eigenvalue, eigenvector = np.linalg.eig(transfer)
    k_of_channel = np.log(eigenvalue)/1j
    ind = np.argsort(np.real(k_of_channel))
    k_of_channel = np.sort(k_of_channel)
    temp = np.zeros((2*dim, 2*dim), dtype=complex)
    temp2 = np.zeros((2*dim), dtype=complex)
    i0 = 0
    for ind0 in ind:
        temp[:, i0] = eigenvector[:, ind0]
        temp2[i0] = eigenvalue[ind0]
        i0 += 1
    eigenvalue = copy.deepcopy(temp2)
    temp = temp[0:dim, :]
    factor = np.zeros(2*dim)
    for dim0 in range(dim):
        factor = factor+np.square(np.abs(temp[dim0, :]))
    for dim0 in range(2*dim):
        temp[:, dim0] = temp[:, dim0]/math.sqrt(factor[dim0])
    velocity_of_channel = np.zeros((2*dim), dtype=complex)
    for dim0 in range(2*dim):
        velocity_of_channel[dim0] = eigenvalue[dim0]*np.dot(np.dot(temp[0:dim, :].transpose().conj(), h01),temp[0:dim, :])[dim0, dim0]
    velocity_of_channel = -2*np.imag(velocity_of_channel)
    eigenvector = copy.deepcopy(temp) 
    return k_of_channel, velocity_of_channel, eigenvalue, eigenvector

def get_classified_k_velocity_u_and_f(fermi_energy, h00, h01):
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    k_of_channel, velocity_of_channel, eigenvalue, eigenvector = guan.get_k_and_velocity_of_channel(fermi_energy, h00, h01)
    ind_right_active = 0; ind_right_evanescent = 0; ind_left_active = 0; ind_left_evanescent = 0
    k_right = np.zeros(dim, dtype=complex); k_left = np.zeros(dim, dtype=complex)
    velocity_right = np.zeros(dim, dtype=complex); velocity_left = np.zeros(dim, dtype=complex)
    lambda_right = np.zeros(dim, dtype=complex); lambda_left = np.zeros(dim, dtype=complex)
    u_right = np.zeros((dim, dim), dtype=complex); u_left = np.zeros((dim, dim), dtype=complex)
    for dim0 in range(2*dim):
        if_active = guan.if_active_channel(k_of_channel[dim0])
        if guan.if_active_channel(k_of_channel[dim0]) == 1:
            direction = np.sign(velocity_of_channel[dim0])
        else:
            direction = np.sign(np.imag(k_of_channel[dim0]))
        if direction == 1:
            if if_active == 1:  # right-moving active channel
                k_right[ind_right_active] = k_of_channel[dim0]
                velocity_right[ind_right_active] = velocity_of_channel[dim0]
                lambda_right[ind_right_active] = eigenvalue[dim0]
                u_right[:, ind_right_active] = eigenvector[:, dim0]
                ind_right_active += 1
            else:               # right-moving evanescent channel
                k_right[dim-1-ind_right_evanescent] = k_of_channel[dim0]
                velocity_right[dim-1-ind_right_evanescent] = velocity_of_channel[dim0]
                lambda_right[dim-1-ind_right_evanescent] = eigenvalue[dim0]
                u_right[:, dim-1-ind_right_evanescent] = eigenvector[:, dim0]
                ind_right_evanescent += 1
        else:
            if if_active == 1:  # left-moving active channel
                k_left[ind_left_active] = k_of_channel[dim0]
                velocity_left[ind_left_active] = velocity_of_channel[dim0]
                lambda_left[ind_left_active] = eigenvalue[dim0]
                u_left[:, ind_left_active] = eigenvector[:, dim0]
                ind_left_active += 1
            else:               # left-moving evanescent channel
                k_left[dim-1-ind_left_evanescent] = k_of_channel[dim0]
                velocity_left[dim-1-ind_left_evanescent] = velocity_of_channel[dim0]
                lambda_left[dim-1-ind_left_evanescent] = eigenvalue[dim0]
                u_left[:, dim-1-ind_left_evanescent] = eigenvector[:, dim0]
                ind_left_evanescent += 1
    lambda_matrix_right = np.diag(lambda_right)
    lambda_matrix_left = np.diag(lambda_left)
    f_right = np.dot(np.dot(u_right, lambda_matrix_right), np.linalg.inv(u_right))
    f_left = np.dot(np.dot(u_left, lambda_matrix_left), np.linalg.inv(u_left))
    return k_right, k_left, velocity_right, velocity_left, f_right, f_left, u_right, u_left, ind_right_active

def calculate_scattering_matrix(fermi_energy, h00, h01, length=100):
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    k_right, k_left, velocity_right, velocity_left, f_right, f_left, u_right, u_left, ind_right_active = guan.get_classified_k_velocity_u_and_f(fermi_energy, h00, h01)
    right_self_energy = np.dot(h01, f_right)
    left_self_energy = np.dot(h01.transpose().conj(), np.linalg.inv(f_left))
    for i0 in range(length):
        if i0 == 0:
            green_nn_n = guan.green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
            green_00_n = copy.deepcopy(green_nn_n)
            green_0n_n = copy.deepcopy(green_nn_n)
            green_n0_n = copy.deepcopy(green_nn_n)
        elif i0 != length-1: 
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0) 
        else:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
        green_00_n = guan.green_function_ii_n(green_00_n, green_0n_n, h01, green_nn_n, green_n0_n)
        green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
        green_n0_n = guan.green_function_ni_n(green_nn_n, h01, green_n0_n)
    temp = np.dot(h01.transpose().conj(), np.linalg.inv(f_right)-np.linalg.inv(f_left))
    transmission_matrix = np.dot(np.dot(np.linalg.inv(u_right), np.dot(green_n0_n, temp)), u_right) 
    reflection_matrix = np.dot(np.dot(np.linalg.inv(u_left), np.dot(green_00_n, temp)-np.identity(dim)), u_right)
    for dim0 in range(dim):
        for dim1 in range(dim):
            if_active = guan.if_active_channel(k_right[dim0])*guan.if_active_channel(k_right[dim1])
            if if_active == 1:
                transmission_matrix[dim0, dim1] = math.sqrt(np.abs(velocity_right[dim0]/velocity_right[dim1])) * transmission_matrix[dim0, dim1]
                reflection_matrix[dim0, dim1] = math.sqrt(np.abs(velocity_left[dim0]/velocity_right[dim1]))*reflection_matrix[dim0, dim1]
            else:
                transmission_matrix[dim0, dim1] = 0
                reflection_matrix[dim0, dim1] = 0
    sum_of_tran_refl_array = np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])), axis=0)+np.sum(np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active])), axis=0)
    for sum_of_tran_refl in sum_of_tran_refl_array:
        if sum_of_tran_refl > 1.001:
            print('Error Alert: scattering matrix is not normalized!')
    return transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active

def information_of_scattering_matrix(transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active):
    if np.array(transmission_matrix).shape==():
        dim = 1
    else:
        dim = np.array(transmission_matrix).shape[0]
    number_of_active_channels = ind_right_active
    number_of_evanescent_channels = dim-ind_right_active
    k_of_right_moving_active_channels = np.real(k_right[0:ind_right_active])
    k_of_left_moving_active_channels = np.real(k_left[0:ind_right_active])
    velocity_of_right_moving_active_channels = np.real(velocity_right[0:ind_right_active])
    velocity_of_left_moving_active_channels = np.real(velocity_left[0:ind_right_active])
    transmission_matrix_for_active_channels = np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active]))
    reflection_matrix_for_active_channels = np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active]))
    total_transmission_of_channels = np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])), axis=0)
    total_conductance = np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])))
    total_reflection_of_channels = np.sum(np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active])), axis=0)
    sum_of_transmission_and_reflection_of_channels = np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])), axis=0) + np.sum(np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active])), axis=0)
    return number_of_active_channels, number_of_evanescent_channels, k_of_right_moving_active_channels, k_of_left_moving_active_channels, velocity_of_right_moving_active_channels, velocity_of_left_moving_active_channels, transmission_matrix_for_active_channels, reflection_matrix_for_active_channels, total_transmission_of_channels, total_conductance, total_reflection_of_channels, sum_of_transmission_and_reflection_of_channels

def print_or_write_scattering_matrix(fermi_energy, h00, h01, length=100, print_show=1, write_file=0, filename='a', file_format='.txt'):
    number_of_active_channels, number_of_evanescent_channels, k_of_right_moving_active_channels, k_of_left_moving_active_channels, velocity_of_right_moving_active_channels, velocity_of_left_moving_active_channels, transmission_matrix_for_active_channels, reflection_matrix_for_active_channels, total_transmission_of_channels, total_conductance, total_reflection_of_channels, sum_of_transmission_and_reflection_of_channels = guan.information_of_scattering_matrix(fermi_energy, h00, h01, length)
    if print_show == 1:
        print('\nActive channel (left or right) = ', number_of_active_channels)
        print('Evanescent channel (left or right) = ', number_of_evanescent_channels, '\n')
        print('K of right-moving active channels:\n', k_of_right_moving_active_channels)
        print('K of left-moving active channels:\n', k_of_left_moving_active_channels, '\n')
        print('Velocity of right-moving active channels:\n', velocity_of_right_moving_active_channels)
        print('Velocity of left-moving active channels:\n', velocity_of_left_moving_active_channels, '\n')
        print('Transmission matrix:\n', transmission_matrix_for_active_channels)
        print('Reflection matrix:\n', reflection_matrix_for_active_channels, '\n')
        print('Total transmission of channels:\n', total_transmission_of_channels)
        print('Total conductance = ', total_conductance, '\n')
        print('Total reflection of channels:\n', total_reflection_of_channels)
        print('Sum of transmission and reflection of channels:\n', sum_of_transmission_and_reflection_of_channels, '\n')
    if write_file == 1:
        with open(filename+file_format, 'w') as f:
            f.write('Active channel (left or right) = ' + str(number_of_active_channels) + '\n')
            f.write('Evanescent channel (left or right) = ' + str(number_of_evanescent_channels) + '\n\n')
            f.write('Channel               K                                     Velocity\n')
            for ind0 in range(number_of_active_channels):
                f.write('   '+str(ind0 + 1) + '   |    '+str(k_of_right_moving_active_channels[ind0])+'            ' + str(velocity_of_right_moving_active_channels[ind0])+'\n')
            f.write('\n')
            for ind0 in range(number_of_active_channels):
                f.write('  -' + str(ind0 + 1) + '   |    ' + str(k_of_left_moving_active_channels[ind0]) + '            ' + str(velocity_of_left_moving_active_channels[ind0]) + '\n')
            f.write('\nScattering matrix:\n              ')
            for ind0 in range(number_of_active_channels):
                f.write(str(ind0+1)+'               ')
            f.write('\n')
            for ind1 in range(number_of_active_channels):
                f.write('  '+str(ind1+1)+'    ')
                for ind2 in range(number_of_active_channels):
                    f.write('%f' % transmission_matrix_for_active_channels[ind1, ind2]+'    ')
                f.write('\n')
            f.write('\n')
            for ind1 in range(number_of_active_channels):
                f.write(' -'+str(ind1+1)+'    ')
                for ind2 in range(number_of_active_channels):
                    f.write('%f' % reflection_matrix_for_active_channels[ind1, ind2]+'    ')
                f.write('\n')
            f.write('\n')
            f.write('Total transmission of channels:\n'+str(total_transmission_of_channels)+'\n')
            f.write('Total conductance = '+str(total_conductance)+'\n')










# Module 9: topological invariant

def calculate_chern_number_for_square_lattice_with_efficient_method(hamiltonian_function, precision=100, print_show=0):
    if np.array(hamiltonian_function(0, 0)).shape==():
        dim = 1
    else:
        dim = np.array(hamiltonian_function(0, 0)).shape[0]   
    delta = 2*math.pi/precision
    chern_number = np.zeros(dim, dtype=complex)
    for kx in np.arange(-math.pi, math.pi, delta):
        if print_show == 1:
            print(kx)
        for ky in np.arange(-math.pi, math.pi, delta):
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
    chern_number = chern_number/(2*math.pi*1j)
    return chern_number

def calculate_chern_number_for_square_lattice_with_efficient_method_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], precision=100, print_show=0): 
    delta = 2*math.pi/precision
    chern_number = 0
    for kx in np.arange(-math.pi, math.pi, delta):
        if print_show == 1:
            print(kx)
        for ky in np.arange(-math.pi, math.pi, delta):
            H = hamiltonian_function(kx, ky)
            eigenvalue, vector = np.linalg.eigh(H) 
            H_delta_kx = hamiltonian_function(kx+delta, ky) 
            eigenvalue, vector_delta_kx = np.linalg.eigh(H_delta_kx) 
            H_delta_ky = hamiltonian_function(kx, ky+delta)
            eigenvalue, vector_delta_ky = np.linalg.eigh(H_delta_ky) 
            H_delta_kx_ky = hamiltonian_function(kx+delta, ky+delta)
            eigenvalue, vector_delta_kx_ky = np.linalg.eigh(H_delta_kx_ky)
            dim = len(index_of_bands)
            det_value = 1
            # first dot
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[dim1, dim2] = np.dot(np.conj(vector[:, dim1]), vector_delta_kx[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # second dot
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[dim1, dim2] = np.dot(np.conj(vector_delta_kx[:, dim1]), vector_delta_kx_ky[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # third dot
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[dim1, dim2] = np.dot(np.conj(vector_delta_kx_ky[:, dim1]), vector_delta_ky[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # four dot
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[dim1, dim2] = np.dot(np.conj(vector_delta_ky[:, dim1]), vector[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value= det_value*dot_matrix
            chern_number += cmath.log(det_value)
    chern_number = chern_number/(2*math.pi*1j)
    return chern_number

def calculate_chern_number_for_square_lattice_with_wilson_loop(hamiltonian_function, precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0):
    delta = 2*math.pi/precision_of_plaquettes
    chern_number = 0
    for kx in np.arange(-math.pi, math.pi, delta):
        if print_show == 1:
            print(kx)
        for ky in np.arange(-math.pi, math.pi, delta):
            vector_array = []
            # line_1
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta/precision_of_wilson_loop*i0, ky) 
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_2
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta, ky+delta/precision_of_wilson_loop*i0)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_3
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta-delta/precision_of_wilson_loop*i0, ky+delta)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_4
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx, ky+delta-delta/precision_of_wilson_loop*i0)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            wilson_loop = 1
            for i0 in range(len(vector_array)-1):
                wilson_loop = wilson_loop*np.dot(vector_array[i0].transpose().conj(), vector_array[i0+1])
            wilson_loop = wilson_loop*np.dot(vector_array[len(vector_array)-1].transpose().conj(), vector_array[0])
            arg = np.log(np.diagonal(wilson_loop))/1j
            chern_number = chern_number + arg
    chern_number = chern_number/(2*math.pi)
    return chern_number

def calculate_chern_number_for_square_lattice_with_wilson_loop_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0):
    delta = 2*math.pi/precision_of_plaquettes
    chern_number = 0
    for kx in np.arange(-math.pi, math.pi, delta):
        if print_show == 1:
            print(kx)
        for ky in np.arange(-math.pi, math.pi, delta):
            vector_array = []
            # line_1
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta/precision_of_wilson_loop*i0, ky) 
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_2
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta, ky+delta/precision_of_wilson_loop*i0)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_3
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta-delta/precision_of_wilson_loop*i0, ky+delta)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_4
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx, ky+delta-delta/precision_of_wilson_loop*i0)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)           
            wilson_loop = 1
            dim = len(index_of_bands)
            for i0 in range(len(vector_array)-1):
                dot_matrix = np.zeros((dim , dim), dtype=complex)
                i01 = 0
                for dim1 in index_of_bands:
                    i02 = 0
                    for dim2 in index_of_bands:
                        dot_matrix[i01, i02] = np.dot(vector_array[i0][:, dim1].transpose().conj(), vector_array[i0+1][:, dim2])
                        i02 += 1
                    i01 += 1
                det_value = np.linalg.det(dot_matrix)
                wilson_loop = wilson_loop*det_value
            dot_matrix_plus = np.zeros((dim , dim), dtype=complex)
            i01 = 0
            for dim1 in index_of_bands:
                i02 = 0
                for dim2 in index_of_bands:
                    dot_matrix_plus[i01, i02] = np.dot(vector_array[len(vector_array)-1][:, dim1].transpose().conj(), vector_array[0][:, dim2])
                    i02 += 1
                i01 += 1
            det_value = np.linalg.det(dot_matrix_plus)
            wilson_loop = wilson_loop*det_value
            arg = np.log(wilson_loop)/1j
            chern_number = chern_number + arg
    chern_number = chern_number/(2*math.pi)
    return chern_number


def calculate_berry_curvature_with_efficient_method(hamiltonian_function, k_min=-math.pi, k_max=math.pi, precision=100, print_show=0):
    if np.array(hamiltonian_function(0, 0)).shape==():
        dim = 1
    else:
        dim = np.array(hamiltonian_function(0, 0)).shape[0]   
    delta = (k_max-k_min)/precision
    k_array = np.arange(k_min, k_max, delta)
    berry_curvature_array = np.zeros((k_array.shape[0], k_array.shape[0], dim), dtype=complex)
    i0 = 0
    for kx in k_array:
        if print_show == 1:
            print(kx)
        j0 = 0
        for ky in k_array:
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
                berry_curvature = cmath.log(Ux*Uy_x*(1/Ux_y)*(1/Uy))/delta/delta*1j
                berry_curvature_array[j0, i0, i] = berry_curvature
            j0 += 1
        i0 += 1
    return k_array, berry_curvature_array

def calculate_berry_curvature_with_efficient_method_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], k_min=-math.pi, k_max=math.pi, precision=100, print_show=0):
    delta = (k_max-k_min)/precision
    k_array = np.arange(k_min, k_max, delta)
    berry_curvature_array = np.zeros((k_array.shape[0], k_array.shape[0]), dtype=complex)
    i00 = 0
    for kx in np.arange(k_min, k_max, delta):
        if print_show == 1:
            print(kx)
        j00 = 0
        for ky in np.arange(k_min, k_max, delta):
            H = hamiltonian_function(kx, ky)
            eigenvalue, vector = np.linalg.eigh(H) 
            H_delta_kx = hamiltonian_function(kx+delta, ky) 
            eigenvalue, vector_delta_kx = np.linalg.eigh(H_delta_kx) 
            H_delta_ky = hamiltonian_function(kx, ky+delta)
            eigenvalue, vector_delta_ky = np.linalg.eigh(H_delta_ky) 
            H_delta_kx_ky = hamiltonian_function(kx+delta, ky+delta)
            eigenvalue, vector_delta_kx_ky = np.linalg.eigh(H_delta_kx_ky)
            dim = len(index_of_bands)
            det_value = 1
            # first dot
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[dim1, dim2] = np.dot(np.conj(vector[:, dim1]), vector_delta_kx[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # second dot
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[dim1, dim2] = np.dot(np.conj(vector_delta_kx[:, dim1]), vector_delta_kx_ky[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # third dot
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[dim1, dim2] = np.dot(np.conj(vector_delta_kx_ky[:, dim1]), vector_delta_ky[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # four dot
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[dim1, dim2] = np.dot(np.conj(vector_delta_ky[:, dim1]), vector[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value= det_value*dot_matrix
            berry_curvature = cmath.log(det_value)/delta/delta*1j
            berry_curvature_array[j00, i00] = berry_curvature
            j00 += 1
        i00 += 1
    return k_array, berry_curvature_array

def calculate_berry_curvature_with_wilson_loop(hamiltonian_function, k_min=-math.pi, k_max=math.pi, precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0):
    if np.array(hamiltonian_function(0, 0)).shape==():
        dim = 1
    else:
        dim = np.array(hamiltonian_function(0, 0)).shape[0]   
    delta = (k_max-k_min)/precision_of_plaquettes
    k_array = np.arange(k_min, k_max, delta)
    berry_curvature_array = np.zeros((k_array.shape[0], k_array.shape[0], dim), dtype=complex)
    i00 = 0
    for kx in k_array:
        if print_show == 1:
            print(kx)
        j00 = 0
        for ky in k_array:
            vector_array = []
            # line_1
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta/precision_of_wilson_loop*i0, ky) 
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_2
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta, ky+delta/precision_of_wilson_loop*i0)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_3
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta-delta/precision_of_wilson_loop*i0, ky+delta)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_4
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx, ky+delta-delta/precision_of_wilson_loop*i0)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            wilson_loop = 1
            for i0 in range(len(vector_array)-1):
                wilson_loop = wilson_loop*np.dot(vector_array[i0].transpose().conj(), vector_array[i0+1])
            wilson_loop = wilson_loop*np.dot(vector_array[len(vector_array)-1].transpose().conj(), vector_array[0])
            berry_curvature = np.log(np.diagonal(wilson_loop))/delta/delta*1j
            berry_curvature_array[j00, i00, :]=berry_curvature
            j00 += 1
        i00 += 1
    return k_array, berry_curvature_array

def calculate_berry_curvature_with_wilson_loop_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], k_min=-math.pi, k_max=math.pi, precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0):
    delta = (k_max-k_min)/precision_of_plaquettes
    k_array = np.arange(k_min, k_max, delta)
    berry_curvature_array = np.zeros((k_array.shape[0], k_array.shape[0]), dtype=complex)
    i000 = 0
    for kx in k_array:
        if print_show == 1:
            print(kx)
        j000 = 0
        for ky in k_array:
            vector_array = []
            # line_1
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta/precision_of_wilson_loop*i0, ky) 
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_2
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta, ky+delta/precision_of_wilson_loop*i0)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_3
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx+delta-delta/precision_of_wilson_loop*i0, ky+delta)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)
            # line_4
            for i0 in range(precision_of_wilson_loop):
                H_delta = hamiltonian_function(kx, ky+delta-delta/precision_of_wilson_loop*i0)  
                eigenvalue, eigenvector = np.linalg.eig(H_delta)
                vector_delta = eigenvector[:, np.argsort(np.real(eigenvalue))]
                vector_array.append(vector_delta)           
            wilson_loop = 1
            dim = len(index_of_bands)
            for i0 in range(len(vector_array)-1):
                dot_matrix = np.zeros((dim , dim), dtype=complex)
                i01 = 0
                for dim1 in index_of_bands:
                    i02 = 0
                    for dim2 in index_of_bands:
                        dot_matrix[i01, i02] = np.dot(vector_array[i0][:, dim1].transpose().conj(), vector_array[i0+1][:, dim2])
                        i02 += 1
                    i01 += 1
                det_value = np.linalg.det(dot_matrix)
                wilson_loop = wilson_loop*det_value
            dot_matrix_plus = np.zeros((dim , dim), dtype=complex)
            i01 = 0
            for dim1 in index_of_bands:
                i02 = 0
                for dim2 in index_of_bands:
                    dot_matrix_plus[i01, i02] = np.dot(vector_array[len(vector_array)-1][:, dim1].transpose().conj(), vector_array[0][:, dim2])
                    i02 += 1
                i01 += 1
            det_value = np.linalg.det(dot_matrix_plus)
            wilson_loop = wilson_loop*det_value
            berry_curvature = np.log(wilson_loop)/delta/delta*1j
            berry_curvature_array[j000, i000]=berry_curvature
            j000 += 1
        i000 += 1
    return k_array, berry_curvature_array

def calculate_chern_number_for_honeycomb_lattice(hamiltonian_function, a=1, precision=300, print_show=0):
    if np.array(hamiltonian_function(0, 0)).shape==():
        dim = 1
    else:
        dim = np.array(hamiltonian_function(0, 0)).shape[0]   
    chern_number = np.zeros(dim, dtype=complex)
    L1 = 4*math.sqrt(3)*math.pi/9/a
    L2 = 2*math.sqrt(3)*math.pi/9/a
    L3 = 2*math.pi/3/a
    delta1 = 2*L1/precision
    delta3 = 2*L3/precision
    for kx in np.arange(-L1, L1, delta1):
        if print_show == 1:
            print(kx)
        for ky in np.arange(-L3, L3, delta3):
            if (-L2<=kx<=L2) or (kx>L2 and -(L1-kx)*math.tan(math.pi/3)<=ky<=(L1-kx)*math.tan(math.pi/3)) or (kx<-L2 and  -(kx-(-L1))*math.tan(math.pi/3)<=ky<=(kx-(-L1))*math.tan(math.pi/3)):
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
    chern_number = chern_number/(2*math.pi*1j)
    return chern_number

def calculate_wilson_loop(hamiltonian_function, k_min=-math.pi, k_max=math.pi, precision=100, print_show=0):
    k_array = np.linspace(k_min, k_max, precision)
    dim = np.array(hamiltonian_function(0)).shape[0]
    wilson_loop_array = np.ones(dim, dtype=complex)
    for i in range(dim):
        if print_show == 1:
            print(i)
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







# Module 10: read and write

def read_one_dimensional_data(filename='a', file_format='.txt'): 
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    for row in row_list:
        column = np.array(row.split()) 
        if column.shape[0] != 0:  
            x_array = np.append(x_array, [float(column[0])], axis=0)  
            y_row = np.zeros(dim_column-1)
            for dim0 in range(dim_column-1):
                y_row[dim0] = float(column[dim0+1])
            if np.array(y_array).shape[0] == 0:
                y_array = [y_row]
            else:
                y_array = np.append(y_array, [y_row], axis=0)
    return x_array, y_array

def read_one_dimensional_complex_data(filename='a', file_format='.txt'): 
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    for row in row_list:
        column = np.array(row.split()) 
        if column.shape[0] != 0:  
            x_array = np.append(x_array, [complex(column[0])], axis=0)  
            y_row = np.zeros(dim_column-1, dtype=complex)
            for dim0 in range(dim_column-1):
                y_row[dim0] = complex(column[dim0+1])
            if np.array(y_array).shape[0] == 0:
                y_array = [y_row]
            else:
                y_array = np.append(y_array, [y_row], axis=0)
    return x_array, y_array

def read_two_dimensional_data(filename='a', file_format='.txt'): 
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    matrix = np.array([])
    for i0 in range(row_list.shape[0]):
        column = np.array(row_list[i0].split()) 
        if i0 == 0:
            x_str = column[1::] 
            x_array = np.zeros(x_str.shape[0])
            for i00 in range(x_str.shape[0]):
                x_array[i00] = float(x_str[i00]) 
        elif column.shape[0] != 0: 
            y_array = np.append(y_array, [float(column[0])], axis=0)  
            matrix_row = np.zeros(dim_column-1)
            for dim0 in range(dim_column-1):
                matrix_row[dim0] = float(column[dim0+1])
            if np.array(matrix).shape[0] == 0:
                matrix = [matrix_row]
            else:
                matrix = np.append(matrix, [matrix_row], axis=0)
    return x_array, y_array, matrix

def read_two_dimensional_complex_data(filename='a', file_format='.txt'): 
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    matrix = np.array([])
    for i0 in range(row_list.shape[0]):
        column = np.array(row_list[i0].split()) 
        if i0 == 0:
            x_str = column[1::] 
            x_array = np.zeros(x_str.shape[0], dtype=complex)
            for i00 in range(x_str.shape[0]):
                x_array[i00] = complex(x_str[i00]) 
        elif column.shape[0] != 0: 
            y_array = np.append(y_array, [complex(column[0])], axis=0)  
            matrix_row = np.zeros(dim_column-1, dtype=complex)
            for dim0 in range(dim_column-1):
                matrix_row[dim0] = complex(column[dim0+1])
            if np.array(matrix).shape[0] == 0:
                matrix = [matrix_row]
            else:
                matrix = np.append(matrix, [matrix_row], axis=0)
    return x_array, y_array, matrix

def write_one_dimensional_data(x_array, y_array, filename='a', file_format='.txt'): 
    with open(filename+file_format, 'w') as file:
        guan.write_one_dimensional_data_without_opening_file(x_array, y_array, file)

def write_one_dimensional_data_without_opening_file(x_array, y_array, file): 
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    i0 = 0
    for x0 in x_array:
        file.write(str(x0)+'   ')
        if len(y_array.shape) == 1:
            file.write(str(y_array[i0])+'\n')
        elif len(y_array.shape) == 2:
            for j0 in range(y_array.shape[1]):
                file.write(str(y_array[i0, j0])+'   ')
            file.write('\n')
        i0 += 1

def write_two_dimensional_data(x_array, y_array, matrix, filename='a', file_format='.txt'): 
    with open(filename+file_format, 'w') as file:
        guan.write_two_dimensional_data_without_opening_file(x_array, y_array, matrix, file)

def write_two_dimensional_data_without_opening_file(x_array, y_array, matrix, file): 
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    matrix = np.array(matrix)
    file.write('0   ')
    for x0 in x_array:
        file.write(str(x0)+'   ')
    file.write('\n')
    i0 = 0
    for y0 in y_array:
        file.write(str(y0))
        j0 = 0
        for x0 in x_array:
            file.write('   '+str(matrix[i0, j0])+'   ')
            j0 += 1
        file.write('\n')
        i0 += 1

def print_array(array, show_index=0, index_type=0):
    if show_index==0:
        for i0 in array:
            print(i0)
    else:
        if index_type==0:
            index = 0
            for i0 in array:
                print(index, i0)
                index += 1
        else:
            index = 0
            for i0 in array:
                index += 1
                print(index, i0)









# Module 11: plot figures

def import_plt_and_start_fig_ax(adjust_bottom=0.2, adjust_left=0.2, labelsize=20):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=adjust_bottom, left=adjust_left)
    ax.grid()
    ax.tick_params(labelsize=labelsize) 
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    return plt, fig, ax

def plot_without_starting_fig(plt, fig, ax, x_array, y_array, xlabel='x', ylabel='y', title='', fontsize=20, style='', y_min=None, y_max=None, linewidth=None, markersize=None): 
    ax.plot(x_array, y_array, style, linewidth=linewidth, markersize=markersize)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y_min=min(y_array)
        if y_max==None:
            y_max=max(y_array)
        ax.set_ylim(y_min, y_max)

def plot(x_array, y_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style='', y_min=None, y_max=None, linewidth=None, markersize=None, adjust_bottom=0.2, adjust_left=0.2): 
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize)
    ax.plot(x_array, y_array, style, linewidth=linewidth, markersize=markersize)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y_min=min(y_array)
        if y_max==None:
            y_max=max(y_array)
        ax.set_ylim(y_min, y_max)
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')

def plot_two_array(x_array, y1_array, y2_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, markersize_1=None, markersize_2=None, adjust_bottom=0.2, adjust_left=0.2): 
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize) 
    ax.plot(x_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y1_min=min(y1_array)
            y2_min=min(y2_array)
            y_min=min([y1_min, y2_min])
        if y_max==None:
            y1_max=max(y1_array)
            y2_max=max(y2_array)
            y_max=max([y1_max, y2_max])
        ax.set_ylim(y_min, y_max)
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')

def plot_two_array_with_two_horizontal_array(x1_array, x2_array, y1_array, y2_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, markersize_1=None, markersize_2=None, adjust_bottom=0.2, adjust_left=0.2): 
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize) 
    ax.plot(x1_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x2_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y1_min=min(y1_array)
            y2_min=min(y2_array)
            y_min=min([y1_min, y2_min])
        if y_max==None:
            y1_max=max(y1_array)
            y2_max=max(y2_array)
            y_max=max([y1_max, y2_max])
        ax.set_ylim(y_min, y_max)
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')

def plot_three_array(x_array, y1_array, y2_array, y3_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', style_3='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, linewidth_3=None,markersize_1=None, markersize_2=None, markersize_3=None, adjust_bottom=0.2, adjust_left=0.2): 
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize) 
    ax.plot(x_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    ax.plot(x_array, y3_array, style_3, linewidth=linewidth_3, markersize=markersize_3)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y1_min=min(y1_array)
            y2_min=min(y2_array)
            y3_min=min(y3_array)
            y_min=min([y1_min, y2_min, y3_min])
        if y_max==None:
            y1_max=max(y1_array)
            y2_max=max(y2_array)
            y3_max=max(y3_array)
            y_max=max([y1_max, y2_max, y3_max])
        ax.set_ylim(y_min, y_max)
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')

def plot_three_array_with_three_horizontal_array(x1_array, x2_array, x3_array, y1_array, y2_array, y3_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', style_3='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, linewidth_3=None,markersize_1=None, markersize_2=None, markersize_3=None, adjust_bottom=0.2, adjust_left=0.2): 
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize) 
    ax.plot(x1_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x2_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    ax.plot(x3_array, y3_array, style_3, linewidth=linewidth_3, markersize=markersize_3)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y1_min=min(y1_array)
            y2_min=min(y2_array)
            y3_min=min(y3_array)
            y_min=min([y1_min, y2_min, y3_min])
        if y_max==None:
            y1_max=max(y1_array)
            y2_max=max(y2_array)
            y3_max=max(y3_array)
            y_max=max([y1_max, y2_max, y3_max])
        ax.set_ylim(y_min, y_max)
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')

def plot_3d_surface(x_array, y_array, matrix, xlabel='x', ylabel='y', zlabel='z', title='', fontsize=20, labelsize=15, show=1, save=0, filename='a', file_format='.jpg', dpi=300, z_min=None, z_max=None, rcount=100, ccount=100): 
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    matrix = np.array(matrix)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.subplots_adjust(bottom=0.1, right=0.65) 
    x_array, y_array = np.meshgrid(x_array, y_array)
    if len(matrix.shape) == 2:
        surf = ax.plot_surface(x_array, y_array, matrix, rcount=rcount, ccount=ccount, cmap=cm.coolwarm, linewidth=0, antialiased=False) 
    elif len(matrix.shape) == 3:
        for i0 in range(matrix.shape[2]):
            surf = ax.plot_surface(x_array, y_array, matrix[:,:,i0], rcount=rcount, ccount=ccount, cmap=cm.coolwarm, linewidth=0, antialiased=False) 
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_zlabel(zlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.zaxis.set_major_locator(LinearLocator(5)) 
    ax.zaxis.set_major_formatter('{x:.2f}')  
    if z_min!=None or z_max!=None:
        if z_min==None:
            z_min=matrix.min()
        if z_max==None:
            z_max=matrix.max()
        ax.set_zlim(z_min, z_max)
    ax.tick_params(labelsize=labelsize) 
    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
    [label.set_fontname('Times New Roman') for label in labels] 
    cax = plt.axes([0.8, 0.1, 0.05, 0.8]) 
    cbar = fig.colorbar(surf, cax=cax)  
    cbar.ax.tick_params(labelsize=labelsize)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')

def plot_contour(x_array, y_array, matrix, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=15, show=1, save=0, filename='a', file_format='.jpg', dpi=300):  
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2, right=0.75, left=0.2) 
    x_array, y_array = np.meshgrid(x_array, y_array)
    contour = ax.contourf(x_array,y_array,matrix,cmap='jet') 
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.tick_params(labelsize=labelsize) 
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels] 
    cax = plt.axes([0.8, 0.2, 0.05, 0.68])
    cbar = fig.colorbar(contour, cax=cax) 
    cbar.ax.tick_params(labelsize=labelsize) 
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')

def draw_dots_and_lines(coordinate_array, draw_dots=1, draw_lines=1, max_distance=1.1, line_style='-k', linewidth=1, dot_style='ro', markersize=3, show=1, save=0, filename='a', file_format='.eps', dpi=300):
    import matplotlib.pyplot as plt
    coordinate_array = np.array(coordinate_array)
    print(coordinate_array.shape)
    x_range = max(coordinate_array[:, 0])-min(coordinate_array[:, 0])
    y_range = max(coordinate_array[:, 1])-min(coordinate_array[:, 1])
    fig, ax = plt.subplots(figsize=(6*x_range/y_range,6))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.axis('off')
    if draw_lines==1:
        for i1 in range(coordinate_array.shape[0]):
            for i2 in range(coordinate_array.shape[0]):
                if np.sqrt((coordinate_array[i1, 0] - coordinate_array[i2, 0])**2+(coordinate_array[i1, 1] - coordinate_array[i2, 1])**2) < max_distance:
                    ax.plot([coordinate_array[i1, 0], coordinate_array[i2, 0]], [coordinate_array[i1, 1], coordinate_array[i2, 1]], line_style, linewidth=linewidth)
    if draw_dots==1:
        for i in range(coordinate_array.shape[0]):
            ax.plot(coordinate_array[i, 0], coordinate_array[i, 1], dot_style, markersize=markersize)
    if show==1:
        plt.show()
    if save==1:
        if file_format=='.eps':
            plt.savefig(filename+file_format)
        else:
            plt.savefig(filename+file_format, dpi=dpi)

def combine_two_images(image_path_array, figsize=(16,8), show=0, save=1, filename='a', file_format='.jpg', dpi=300):
    num = np.array(image_path_array).shape[0]
    if num != 2:
        print('Error: The number of images should be two!')
    else:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0) 
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        image_1 = mpimg.imread(image_path_array[0])
        image_2 = mpimg.imread(image_path_array[1])
        ax1.imshow(image_1)
        ax2.imshow(image_2)
        ax1.axis('off')
        ax2.axis('off')
        if show == 1:
            plt.show()
        if save == 1:
            plt.savefig(filename+file_format, dpi=dpi)
        plt.close('all')

def combine_three_images(image_path_array, figsize=(16,5), show=0, save=1, filename='a', file_format='.jpg', dpi=300):
    num = np.array(image_path_array).shape[0]
    if num != 3:
        print('Error: The number of images should be three!')
    else:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0) 
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        image_1 = mpimg.imread(image_path_array[0])
        image_2 = mpimg.imread(image_path_array[1])
        image_3 = mpimg.imread(image_path_array[2])
        ax1.imshow(image_1)
        ax2.imshow(image_2)
        ax3.imshow(image_3)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        if show == 1:
            plt.show()
        if save == 1:
            plt.savefig(filename+file_format, dpi=dpi)
        plt.close('all')

def combine_four_images(image_path_array, figsize=(16,16), show=0, save=1, filename='a', file_format='.jpg', dpi=300):
    num = np.array(image_path_array).shape[0]
    if num != 4:
        print('Error: The number of images should be four!')
    else:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0) 
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        image_1 = mpimg.imread(image_path_array[0])
        image_2 = mpimg.imread(image_path_array[1])
        image_3 = mpimg.imread(image_path_array[2])
        image_4 = mpimg.imread(image_path_array[3])
        ax1.imshow(image_1)
        ax2.imshow(image_2)
        ax3.imshow(image_3)
        ax4.imshow(image_4)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')
        if show == 1:
            plt.show()
        if save == 1:
            plt.savefig(filename+file_format, dpi=dpi)
        plt.close('all')

def make_gif(image_path_array, filename='a', duration=0.1):
    import imageio
    images = []
    for image_path in image_path_array:
        im = imageio.imread(image_path)
        images.append(im)
    imageio.mimsave(filename+'.gif', images, 'GIF', duration=duration)











# Module 12: data processing

def preprocess_for_parallel_calculations(parameter_array_all, cpus=1, task_index=0):
    num_all = np.array(parameter_array_all).shape[0]
    if num_all%cpus == 0:
        num_parameter = int(num_all/cpus) 
        parameter_array = parameter_array_all[task_index*num_parameter:(task_index+1)*num_parameter]
    else:
        num_parameter = int(num_all/(cpus-1))
        if task_index != cpus-1:
            parameter_array = parameter_array_all[task_index*num_parameter:(task_index+1)*num_parameter]
        else:
            parameter_array = parameter_array_all[task_index*num_parameter:num_all]
    return parameter_array

def find_close_values_in_one_array(array, precision=1e-2):
    new_array = []
    i0 = 0
    for a1 in array:
        j0 = 0
        for a2 in array:
            if j0>i0 and abs(a1-a2)<precision: 
                new_array.append([a1, a2])
            j0 +=1
        i0 += 1
    return new_array

def find_degenerate_points(k_array, eigenvalue_array, precision=1e-2):
    degenerate_k_array = []
    degenerate_eigenvalue_array = []
    i0 = 0
    for k in k_array:
        degenerate_points = find_close_values_in_one_array(eigenvalue_array[i0], precision=precision)
        if len(degenerate_points) != 0:
            degenerate_k_array.append(k)
            degenerate_eigenvalue_array.append(degenerate_points)
        i0 += 1
    return degenerate_k_array, degenerate_eigenvalue_array

def batch_reading_and_plotting(directory, xlabel='x', ylabel='y'):
    import re
    import os
    for root, dirs, files in os.walk(directory):
        for file in files:
            if re.search('^txt.', file[::-1]):
                filename = file[:-4]
                x_array, y_array = guan.read_one_dimensional_data(filename=filename)
                guan.plot(x_array, y_array, xlabel=xlabel, ylabel=ylabel, title=filename, show=0, save=1, filename=filename)

def write_file_list_in_markdown(directory, filename='a', reverse_positive_or_negative=1, starting_from_h1=None, banned_type=[], hide_file_type=None, divided_line=None, show_second_number=None, show_third_number=None): 
    import os
    f = open(filename+'.md', 'w', encoding="utf-8")
    filenames1 = os.listdir(directory)
    u0 = 0
    for filename1 in filenames1[::reverse_positive_or_negative]:
        filename1_with_path = os.path.join(directory,filename1) 
        if os.path.isfile(filename1_with_path):
            if os.path.splitext(filename1)[1] not in banned_type:
                if hide_file_type == None:
                    f.write('+ '+str(filename1)+'\n\n')
                else:
                    f.write('+ '+str(os.path.splitext(filename1)[0])+'\n\n')
        else:
            u0 += 1
            if divided_line != None and u0 != 1:
                f.write('--------\n\n')
            if starting_from_h1 == None:
                f.write('#')
            f.write('# '+str(filename1)+'\n\n')

            filenames2 = os.listdir(filename1_with_path) 
            i0 = 0     
            for filename2 in filenames2[::reverse_positive_or_negative]:
                filename2_with_path = os.path.join(directory, filename1, filename2) 
                if os.path.isfile(filename2_with_path):
                    if os.path.splitext(filename2)[1] not in banned_type:
                        if hide_file_type == None:
                            f.write('+ '+str(filename2)+'\n\n')
                        else:
                            f.write('+ '+str(os.path.splitext(filename2)[0])+'\n\n')
                else: 
                    i0 += 1
                    if starting_from_h1 == None:
                        f.write('#')
                    if show_second_number != None:
                        f.write('## '+str(i0)+'. '+str(filename2)+'\n\n')
                    else:
                        f.write('## '+str(filename2)+'\n\n')
                    
                    j0 = 0
                    filenames3 = os.listdir(filename2_with_path)
                    for filename3 in filenames3[::reverse_positive_or_negative]:
                        filename3_with_path = os.path.join(directory, filename1, filename2, filename3) 
                        if os.path.isfile(filename3_with_path): 
                            if os.path.splitext(filename3)[1] not in banned_type:
                                if hide_file_type == None:
                                    f.write('+ '+str(filename3)+'\n\n')
                                else:
                                    f.write('+ '+str(os.path.splitext(filename3)[0])+'\n\n')
                        else:
                            j0 += 1
                            if starting_from_h1 == None:
                                f.write('#')
                            if show_third_number != None:
                                f.write('### ('+str(j0)+') '+str(filename3)+'\n\n')
                            else:
                                f.write('### '+str(filename3)+'\n\n')

                            filenames4 = os.listdir(filename3_with_path)
                            for filename4 in filenames4[::reverse_positive_or_negative]:
                                filename4_with_path = os.path.join(directory, filename1, filename2, filename3, filename4) 
                                if os.path.isfile(filename4_with_path):
                                    if os.path.splitext(filename4)[1] not in banned_type:
                                        if hide_file_type == None:
                                            f.write('+ '+str(filename4)+'\n\n')
                                        else:
                                            f.write('+ '+str(os.path.splitext(filename4)[0])+'\n\n')
                                else: 
                                    if starting_from_h1 == None:
                                        f.write('#')
                                    f.write('#### '+str(filename4)+'\n\n')

                                    filenames5 = os.listdir(filename4_with_path)
                                    for filename5 in filenames5[::reverse_positive_or_negative]:
                                        filename5_with_path = os.path.join(directory, filename1, filename2, filename3, filename4, filename5) 
                                        if os.path.isfile(filename5_with_path): 
                                            if os.path.splitext(filename5)[1] not in banned_type:
                                                if hide_file_type == None:
                                                    f.write('+ '+str(filename5)+'\n\n')
                                                else:
                                                    f.write('+ '+str(os.path.splitext(filename5)[0])+'\n\n')
                                        else:
                                            if starting_from_h1 == None:
                                                f.write('#')
                                            f.write('##### '+str(filename5)+'\n\n')

                                            filenames6 = os.listdir(filename5_with_path)
                                            for filename6 in filenames6[::reverse_positive_or_negative]:
                                                filename6_with_path = os.path.join(directory, filename1, filename2, filename3, filename4, filename5, filename6) 
                                                if os.path.isfile(filename6_with_path): 
                                                    if os.path.splitext(filename6)[1] not in banned_type:
                                                        if hide_file_type == None:
                                                            f.write('+ '+str(filename6)+'\n\n')
                                                        else:
                                                            f.write('+ '+str(os.path.splitext(filename6)[0])+'\n\n')
                                                else:
                                                    if starting_from_h1 == None:
                                                        f.write('#')
                                                    f.write('###### '+str(filename6)+'\n\n')
    f.close()

def creat_necessary_file(directory, filename='readme', file_format='.md', content=''):
    import os
    directory_with_file = []
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            if root not in directory_with_file:
                directory_with_file.append(root)
    for root in directory_with_file:
        os.chdir(root)
        f = open(filename+file_format, 'w', encoding="utf-8")
        f.write(content)
        f.close()

def delete_file_with_specific_name(directory, filename='readme', file_format='.md'):
      import os
      for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            if files[i0] == filename+file_format:
                os.remove(root+'/'+files[i0])

def move_all_files_to_root_directory(directory):
    import os
    import shutil
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            shutil.move(root+'/'+files[i0], directory+'/'+files[i0])
    for i0 in range(100):
        for root, dirs, files in os.walk(directory):
            try:
                os.rmdir(root) 
            except:
                pass

def change_directory_by_replacement(current_key_word='code', new_key_word='data'):
    import os
    code_path = os.getcwd()
    data_path = code_path.replace('\\', '/') 
    data_path = data_path.replace(current_key_word, new_key_word) 
    if os.path.exists(data_path) == False:
        os.makedirs(data_path)
    os.chdir(data_path)

def rgb_to_hex(rgb, pound=1):
    if pound==0:
        return '%02x%02x%02x' % rgb
    else:
        return '#%02x%02x%02x' % rgb

def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    length = len(hex)
    return tuple(int(hex[i:i+length//3], 16) for i in range(0, length, length//3))






# Module 13: others

## download

def download_with_scihub(address=None, num=1):
    from bs4 import BeautifulSoup
    import re
    import requests
    import os
    if num==1 and address!=None:
        address_array = [address]
    else:
        address_array = []
        for i in range(num):
            address = input('\nInput：')
            address_array.append(address)
    for address in address_array:
        r = requests.post('https://sci-hub.st/', data={'request': address})
        print('\nResponse：', r)
        print('Address：', r.url)
        soup = BeautifulSoup(r.text, features='lxml')
        pdf_URL = soup.embed['src']
        # pdf_URL = soup.iframe['src'] # This is a code line of history version which fails to get pdf URL.
        if re.search(re.compile('^https:'), pdf_URL):
            pass
        else:
            pdf_URL = 'https:'+pdf_URL
        print('PDF address：', pdf_URL)
        name = re.search(re.compile('fdp.*?/'),pdf_URL[::-1]).group()[::-1][1::]
        print('PDF name：', name)
        print('Directory：', os.getcwd())
        print('\nDownloading...')
        r = requests.get(pdf_URL, stream=True)
        with open(name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=32):
                f.write(chunk)
        print('Completed!\n')
    if num != 1:
        print('All completed!\n')

## PDF

def get_links_from_pdf(pdf_path, link_starting_form=''):
    # Example: link_starting_form='https://doi.org'
    import PyPDF2
    import re
    pdfReader = PyPDF2.PdfFileReader(pdf_path)
    pages = pdfReader.getNumPages()
    i0 = 0
    links = []
    for page in range(pages):
        pageSliced = pdfReader.getPage(page)
        pageObject = pageSliced.getObject()
        if '/Annots' in pageObject.keys():
            ann = pageObject['/Annots']
            old = ''
            for a in ann:
                u = a.getObject()
                if '/A' in u.keys():
                    if re.search(re.compile('^'+link_starting_form), u['/A']['/URI']):
                        if u['/A']['/URI'] != old:
                            links.append(u['/A']['/URI']) 
                            i0 += 1
                            old = u['/A']['/URI']        
    return links

def pdf_to_text(pdf_path):
    from pdfminer.pdfparser import PDFParser, PDFDocument
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTTextBox
    from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
    import logging 
    logging.Logger.propagate = False 
    logging.getLogger().setLevel(logging.ERROR) 
    praser = PDFParser(open(pdf_path, 'rb'))
    doc = PDFDocument()
    praser.set_document(doc)
    doc.set_parser(praser)
    doc.initialize()
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        content = ''
        for page in doc.get_pages():
            interpreter.process_page(page)                        
            layout = device.get_result()                     
            for x in layout:
                if isinstance(x, LTTextBox):
                    content  = content + x.get_text().strip()
    return content

## audio

def str_to_audio(str='hello world', filename='str', rate=125, voice=1, read=1, save=0, compress=0, bitrate='16k', print_text=0):
    import pyttsx3
    if print_text==1:
        print(str)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')  
    engine.setProperty('voice', voices[voice].id)
    engine.setProperty("rate", rate)
    if save==1:
        engine.save_to_file(str, filename+'.wav')
        engine.runAndWait()
        print('Wav file saved!')
        if compress==1:
            import os
            os.rename(filename+'.wav', 'temp.wav')
            guan.compress_wav_to_mp3('temp.wav', output_filename=filename+'.mp3', bitrate=bitrate)
            os.remove('temp.wav')
    if read==1:
        engine.say(str)
        engine.runAndWait()

def txt_to_audio(txt_path, rate=125, voice=1, read=1, save=0, compress=0, bitrate='16k', print_text=0):
    import pyttsx3
    f = open(txt_path, 'r', encoding ='utf-8')
    text = f.read()
    if print_text==1:
        print(text)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')  
    engine.setProperty('voice', voices[voice].id)
    engine.setProperty("rate", rate)
    if save==1:
        import re
        filename = re.split('[/,\\\]', txt_path)[-1][:-4]
        engine.save_to_file(text, filename+'.wav')
        engine.runAndWait()
        print('Wav file saved!')
        if compress==1:
            import os
            os.rename(filename+'.wav', 'temp.wav')
            guan.compress_wav_to_mp3('temp.wav', output_filename=filename+'.mp3', bitrate=bitrate)
            os.remove('temp.wav')
    if read==1:
        engine.say(text)
        engine.runAndWait()

def pdf_to_audio(pdf_path, rate=125, voice=1, read=1, save=0, compress=0, bitrate='16k', print_text=0):
    import pyttsx3
    text = guan.pdf_to_text(pdf_path)
    text = text.replace('\n', ' ')
    if print_text==1:
        print(text)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')  
    engine.setProperty('voice', voices[voice].id)
    engine.setProperty("rate", rate)
    if save==1:
        import re
        filename = re.split('[/,\\\]', pdf_path)[-1][:-4]
        engine.save_to_file(text, filename+'.wav')
        engine.runAndWait()
        print('Wav file saved!')
        if compress==1:
            import os
            os.rename(filename+'.wav', 'temp.wav')
            guan.compress_wav_to_mp3('temp.wav', output_filename=filename+'.mp3', bitrate=bitrate)
            os.remove('temp.wav')
    if read==1:
        engine.say(text)
        engine.runAndWait()

def compress_wav_to_mp3(wav_path, output_filename='a.mp3', bitrate='16k'):
    # Note: Beside the installation of pydub, you may also need download FFmpeg on http://www.ffmpeg.org/download.html and add the bin path to the environment variable.
    from pydub import AudioSegment
    sound = AudioSegment.from_mp3(wav_path)
    sound.export(output_filename,format="mp3",bitrate=bitrate)

## words

def play_academic_words(reverse=0, random_on=0, bre_or_ame='ame', show_translation=1, show_link=1, translation_time=2, rest_time=1):
    from bs4 import BeautifulSoup
    import re
    import urllib.request
    import requests
    import os
    import pygame
    import time
    import ssl
    import random
    ssl._create_default_https_context = ssl._create_unverified_context
    html = urllib.request.urlopen("https://www.guanjihuan.com/archives/4418").read().decode('utf-8')
    if bre_or_ame == 'ame':
        directory = 'words_mp3_ameProns/'
    elif bre_or_ame == 'bre':
        directory = 'words_mp3_breProns/'
    exist_directory = os.path.exists(directory)
    html_file = urllib.request.urlopen("https://file.guanjihuan.com/words/"+directory).read().decode('utf-8')
    if exist_directory == 0:
        os.makedirs(directory)
    soup = BeautifulSoup(html, features='lxml')
    contents = re.findall('<h2>.*?</a></p>', html, re.S)
    if random_on==1:
        random.shuffle(contents)
    if reverse==1:
        contents.reverse()
    for content in contents:
        soup2 = BeautifulSoup(content, features='lxml')
        all_h2 = soup2.find_all('h2')
        for h2 in all_h2:
            if re.search('\d*. ', h2.get_text()):
                word = re.findall('[a-zA-Z].*', h2.get_text(), re.S)[0]
                exist = os.path.exists(directory+word+'.mp3')
                if not exist:
                    try:
                        if re.search(word, html_file):
                            r = requests.get("https://file.guanjihuan.com/words/"+directory+word+".mp3", stream=True)
                            with open(directory+word+'.mp3', 'wb') as f:
                                for chunk in r.iter_content(chunk_size=32):
                                    f.write(chunk)
                    except:
                        pass
                print(h2.get_text())
                try:
                    pygame.mixer.init()
                    track = pygame.mixer.music.load(directory+word+'.mp3')
                    pygame.mixer.music.play()
                    if show_link==1:
                        print('https://www.ldoceonline.com/dictionary/'+word)
                except:
                    pass
                translation = re.findall('<p>.*?</p>', content, re.S)[0][3:-4]
                if show_translation==1:
                    time.sleep(translation_time)
                    print(translation)
                time.sleep(rest_time)
                pygame.mixer.music.stop()
                print()

def play_selected_academic_words(reverse=0, random_on=0, bre_or_ame='ame', show_link=1, rest_time=3):
    from bs4 import BeautifulSoup
    import re
    import urllib.request
    import requests
    import os
    import pygame
    import time
    import ssl
    import random
    ssl._create_default_https_context = ssl._create_unverified_context
    html = urllib.request.urlopen("https://www.guanjihuan.com/archives/24732").read().decode('utf-8')
    if bre_or_ame == 'ame':
        directory = 'words_mp3_ameProns/'
    elif bre_or_ame == 'bre':
        directory = 'words_mp3_breProns/'
    exist_directory = os.path.exists(directory)
    html_file = urllib.request.urlopen("https://file.guanjihuan.com/words/"+directory).read().decode('utf-8')
    if exist_directory == 0:
        os.makedirs(directory)
    soup = BeautifulSoup(html, features='lxml')
    contents = re.findall('<li>\d.*?</li>', html, re.S)
    if random_on==1:
        random.shuffle(contents)
    if reverse==1:
        contents.reverse()
    for content in contents:
        soup2 = BeautifulSoup(content, features='lxml')
        all_li = soup2.find_all('li')
        for li in all_li:
            if re.search('\d*. ', li.get_text()):
                word = re.findall('\s[a-zA-Z].*?\s', li.get_text(), re.S)[0][1:-1]
                exist = os.path.exists(directory+word+'.mp3')
                if not exist:
                    try:
                        if re.search(word, html_file):
                            r = requests.get("https://file.guanjihuan.com/words/"+directory+word+".mp3", stream=True)
                            with open(directory+word+'.mp3', 'wb') as f:
                                for chunk in r.iter_content(chunk_size=32):
                                    f.write(chunk)
                    except:
                        pass
                print(li.get_text())
                try:
                    pygame.mixer.init()
                    track = pygame.mixer.music.load(directory+word+'.mp3')
                    pygame.mixer.music.play()
                    if show_link==1:
                        print('https://www.ldoceonline.com/dictionary/'+word)
                except:
                    pass
                time.sleep(rest_time)
                pygame.mixer.music.stop()
                print()

def play_element_words(random_on=0, show_translation=1, show_link=1, translation_time=2, rest_time=1):
    from bs4 import BeautifulSoup
    import re
    import urllib.request
    import requests
    import os
    import pygame
    import time
    import ssl
    import random
    ssl._create_default_https_context = ssl._create_unverified_context
    html = urllib.request.urlopen("https://www.guanjihuan.com/archives/10897").read().decode('utf-8')
    directory = 'prons/'
    exist_directory = os.path.exists(directory)
    html_file = urllib.request.urlopen("https://file.guanjihuan.com/words/periodic_table_of_elements/"+directory).read().decode('utf-8')
    if exist_directory == 0:
        os.makedirs(directory)
    soup = BeautifulSoup(html, features='lxml')
    contents = re.findall('<h2>.*?</a></p>', html, re.S)
    if random_on==1:
        random.shuffle(contents)
    for content in contents:
        soup2 = BeautifulSoup(content, features='lxml')
        all_h2 = soup2.find_all('h2')
        for h2 in all_h2:
            if re.search('\d*. ', h2.get_text()):
                word = re.findall('[a-zA-Z].* \(', h2.get_text(), re.S)[0][:-2]
                exist = os.path.exists(directory+word+'.mp3')
                if not exist:
                    try:
                        if re.search(word, html_file):
                            r = requests.get("https://file.guanjihuan.com/words/periodic_table_of_elements/prons/"+word+".mp3", stream=True)
                            with open(directory+word+'.mp3', 'wb') as f:
                                for chunk in r.iter_content(chunk_size=32):
                                    f.write(chunk)
                    except:
                        pass
                print(h2.get_text())
                try:
                    pygame.mixer.init()
                    track = pygame.mixer.music.load(directory+word+'.mp3')
                    pygame.mixer.music.play()
                    if show_link==1:
                        print('https://www.merriam-webster.com/dictionary/'+word)
                except:
                    pass
                translation = re.findall('<p>.*?</p>', content, re.S)[0][3:-4]
                if show_translation==1:
                    time.sleep(translation_time)
                    print(translation)
                time.sleep(rest_time)
                pygame.mixer.music.stop()
                print()