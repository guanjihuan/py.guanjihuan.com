# Module: Fourier_transform
import guan

# 通过元胞和跃迁项得到一维的哈密顿量（需要输入k值）
@guan.statistics_decorator
def one_dimensional_fourier_transform(k, unit_cell, hopping):
    import numpy as np
    import cmath
    unit_cell = np.array(unit_cell)
    hopping = np.array(hopping)
    hamiltonian = unit_cell+hopping*cmath.exp(1j*k)+hopping.transpose().conj()*cmath.exp(-1j*k)
    return hamiltonian

# 通过元胞和跃迁项得到二维方格子的哈密顿量（需要输入k值）
@guan.statistics_decorator
def two_dimensional_fourier_transform_for_square_lattice(k1, k2, unit_cell, hopping_1, hopping_2):
    import numpy as np
    import cmath
    unit_cell = np.array(unit_cell)
    hopping_1 = np.array(hopping_1)
    hopping_2 = np.array(hopping_2)
    hamiltonian = unit_cell+hopping_1*cmath.exp(1j*k1)+hopping_1.transpose().conj()*cmath.exp(-1j*k1)+hopping_2*cmath.exp(1j*k2)+hopping_2.transpose().conj()*cmath.exp(-1j*k2)
    return hamiltonian

# 通过元胞和跃迁项得到三维立方格子的哈密顿量（需要输入k值）
@guan.statistics_decorator
def three_dimensional_fourier_transform_for_cubic_lattice(k1, k2, k3, unit_cell, hopping_1, hopping_2, hopping_3):
    import numpy as np
    import cmath
    unit_cell = np.array(unit_cell)
    hopping_1 = np.array(hopping_1)
    hopping_2 = np.array(hopping_2)
    hopping_3 = np.array(hopping_3)
    hamiltonian = unit_cell+hopping_1*cmath.exp(1j*k1)+hopping_1.transpose().conj()*cmath.exp(-1j*k1)+hopping_2*cmath.exp(1j*k2)+hopping_2.transpose().conj()*cmath.exp(-1j*k2)+hopping_3*cmath.exp(1j*k3)+hopping_3.transpose().conj()*cmath.exp(-1j*k3)
    return hamiltonian

# 通过元胞和跃迁项得到一维的哈密顿量（返回的哈密顿量为携带k的函数）
@guan.statistics_decorator
def one_dimensional_fourier_transform_with_k(unit_cell, hopping):
    import functools
    import guan
    hamiltonian_function = functools.partial(guan.one_dimensional_fourier_transform, unit_cell=unit_cell, hopping=hopping)
    return hamiltonian_function

# 通过元胞和跃迁项得到二维方格子的哈密顿量（返回的哈密顿量为携带k的函数）
@guan.statistics_decorator
def two_dimensional_fourier_transform_for_square_lattice_with_k1_k2(unit_cell, hopping_1, hopping_2):
    import functools
    import guan
    hamiltonian_function = functools.partial(guan.two_dimensional_fourier_transform_for_square_lattice, unit_cell=unit_cell, hopping_1=hopping_1, hopping_2=hopping_2)
    return hamiltonian_function

# 通过元胞和跃迁项得到三维立方格子的哈密顿量（返回的哈密顿量为携带k的函数）
@guan.statistics_decorator
def three_dimensional_fourier_transform_for_cubic_lattice_with_k1_k2_k3(unit_cell, hopping_1, hopping_2, hopping_3):
    import functools
    import guan
    hamiltonian_function = functools.partial(guan.three_dimensional_fourier_transform_for_cubic_lattice, unit_cell=unit_cell, hopping_1=hopping_1, hopping_2=hopping_2, hopping_3=hopping_3)
    return hamiltonian_function

# 由实空间格矢得到倒空间格矢（一维）
@guan.statistics_decorator
def calculate_one_dimensional_reciprocal_lattice_vector(a1):
    import numpy as np
    b1 = 2*np.pi/a1
    return b1

# 由实空间格矢得到倒空间格矢（二维）
@guan.statistics_decorator
def calculate_two_dimensional_reciprocal_lattice_vectors(a1, a2):
    import numpy as np
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

# 由实空间格矢得到倒空间格矢（三维）
@guan.statistics_decorator
def calculate_three_dimensional_reciprocal_lattice_vectors(a1, a2, a3):
    import numpy as np
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)
    b1 = 2*np.pi*np.cross(a2, a3)/np.dot(a1, np.cross(a2, a3))
    b2 = 2*np.pi*np.cross(a3, a1)/np.dot(a1, np.cross(a2, a3))
    b3 = 2*np.pi*np.cross(a1, a2)/np.dot(a1, np.cross(a2, a3))
    return b1, b2, b3

# 由实空间格矢得到倒空间格矢（一维），这里为符号运算
@guan.statistics_decorator
def calculate_one_dimensional_reciprocal_lattice_vector_with_sympy(a1):
    import sympy
    b1 = 2*sympy.pi/a1
    return b1

# 由实空间格矢得到倒空间格矢（二维），这里为符号运算
@guan.statistics_decorator
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

# 由实空间格矢得到倒空间格矢（三维），这里为符号运算
@guan.statistics_decorator
def calculate_three_dimensional_reciprocal_lattice_vectors_with_sympy(a1, a2, a3):
    import sympy
    cross_a2_a3 = a2.cross(a3)
    cross_a3_a1 = a3.cross(a1)
    cross_a1_a2 = a1.cross(a2)
    b1 = 2*sympy.pi*cross_a2_a3/a1.dot(cross_a2_a3)
    b2 = 2*sympy.pi*cross_a3_a1/a1.dot(cross_a2_a3)
    b3 = 2*sympy.pi*cross_a1_a2/a1.dot(cross_a2_a3)
    return b1, b2, b3
