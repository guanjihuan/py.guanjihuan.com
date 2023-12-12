# Module: Hamiltonian_of_examples
import guan

# 构建一维的有限尺寸体系哈密顿量（可设置是否为周期边界条件）
@guan.statistics_decorator
def hamiltonian_of_finite_size_system_along_one_direction(N, on_site=0, hopping=1, period=0):
    import numpy as np
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

# 构建二维的方格子有限尺寸体系哈密顿量（可设置是否为周期边界条件）
@guan.statistics_decorator
def hamiltonian_of_finite_size_system_along_two_directions_for_square_lattice(N1, N2, on_site=0, hopping_1=1, hopping_2=1, period_1=0, period_2=0):
    import numpy as np
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

# 构建三维的立方格子有限尺寸体系哈密顿量（可设置是否为周期边界条件）
@guan.statistics_decorator
def hamiltonian_of_finite_size_system_along_three_directions_for_cubic_lattice(N1, N2, N3, on_site=0, hopping_1=1, hopping_2=1, hopping_3=1, period_1=0, period_2=0, period_3=0):
    import numpy as np
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

# 构建有限尺寸的SSH模型哈密顿量
@guan.statistics_decorator
def hamiltonian_of_finite_size_ssh_model(N, v=0.6, w=1, onsite_1=0, onsite_2=0, period=1):
    import numpy as np
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

# 获取Zigzag边的石墨烯条带的元胞间跃迁
@guan.statistics_decorator
def get_hopping_term_of_graphene_ribbon_along_zigzag_direction(N, eta=0):
    import numpy as np
    hopping = np.zeros((4*N, 4*N), dtype=complex)
    for i0 in range(N):
        hopping[4*i0+0, 4*i0+0] = eta
        hopping[4*i0+1, 4*i0+1] = eta
        hopping[4*i0+2, 4*i0+2] = eta
        hopping[4*i0+3, 4*i0+3] = eta
        hopping[4*i0+1, 4*i0+0] = 1
        hopping[4*i0+2, 4*i0+3] = 1
    return hopping

# 构建有限尺寸的石墨烯哈密顿量（可设置是否为周期边界条件）
@guan.statistics_decorator
def hamiltonian_of_finite_size_system_along_two_directions_for_graphene(N1, N2, period_1=0, period_2=0):
    import numpy as np
    import guan
    on_site = guan.hamiltonian_of_finite_size_system_along_one_direction(4)
    hopping_1 = guan.get_hopping_term_of_graphene_ribbon_along_zigzag_direction(1)
    hopping_2 = np.zeros((4, 4), dtype=complex)
    hopping_2[3, 0] = 1
    hamiltonian = guan.hamiltonian_of_finite_size_system_along_two_directions_for_square_lattice(N1, N2, on_site, hopping_1, hopping_2, period_1, period_2)
    return hamiltonian

# 获取石墨烯有效模型沿着x方向的在位能和跃迁项（其中，动量qy为参数）
@guan.statistics_decorator
def get_onsite_and_hopping_terms_of_2d_effective_graphene_along_one_direction(qy, t=1, staggered_potential=0, eta=0, valley_index=0):
    import numpy as np
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

# 获取BHZ模型的在位能和跃迁项
@guan.statistics_decorator
def get_onsite_and_hopping_terms_of_bhz_model(A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01, a=1):
    import numpy as np
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

# 获取半个BHZ模型的在位能和跃迁项（自旋向上）
@guan.statistics_decorator
def get_onsite_and_hopping_terms_of_half_bhz_model_for_spin_up(A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01, a=1):
    import numpy as np
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

# 获取半个BHZ模型的在位能和跃迁项（自旋向下）
@guan.statistics_decorator
def get_onsite_and_hopping_terms_of_half_bhz_model_for_spin_down(A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01, a=1):
    import numpy as np
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

# 一维链的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_simple_chain(k):
    import guan
    hamiltonian = guan.one_dimensional_fourier_transform(k, unit_cell=0, hopping=1)
    return hamiltonian

# 二维方格子的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_square_lattice(k1, k2):
    import guan
    hamiltonian = guan.two_dimensional_fourier_transform_for_square_lattice(k1, k2, unit_cell=0, hopping_1=1, hopping_2=1)
    return hamiltonian

# 准一维方格子条带的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_square_lattice_in_quasi_one_dimension(k, N=10, period=0):
    import numpy as np
    import guan
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

# 三维立方格子的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_cubic_lattice(k1, k2, k3):
    import guan
    hamiltonian = guan.three_dimensional_fourier_transform_for_cubic_lattice(k1, k2, k3, unit_cell=0, hopping_1=1, hopping_2=1, hopping_3=1)
    return hamiltonian

# SSH模型的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_ssh_model(k, v=0.6, w=1):
    import numpy as np
    import cmath
    hamiltonian = np.zeros((2, 2), dtype=complex)
    hamiltonian[0,1] = v+w*cmath.exp(-1j*k)
    hamiltonian[1,0] = v+w*cmath.exp(1j*k)
    return hamiltonian

# 石墨烯的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_graphene(k1, k2, staggered_potential=0, t=1, a='default'):
    import numpy as np
    import cmath
    import math
    if a == 'default':
        a = 1/math.sqrt(3)
    h0 = np.zeros((2, 2), dtype=complex)  # mass term
    h1 = np.zeros((2, 2), dtype=complex)  # nearest hopping
    h0[0, 0] = staggered_potential     
    h0[1, 1] = -staggered_potential
    h1[1, 0] = t*(cmath.exp(1j*k2*a)+cmath.exp(1j*math.sqrt(3)/2*k1*a-1j/2*k2*a)+cmath.exp(-1j*math.sqrt(3)/2*k1*a-1j/2*k2*a))   
    h1[0, 1] = h1[1, 0].conj()
    hamiltonian = h0 + h1
    return hamiltonian

# 石墨烯有效模型的哈密顿量（倒空间）
@guan.statistics_decorator
def effective_hamiltonian_of_graphene(qx, qy, t=1, staggered_potential=0, valley_index=0):
    import numpy as np
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

# 石墨烯有效模型离散化后的哈密顿量（倒空间）
@guan.statistics_decorator
def effective_hamiltonian_of_graphene_after_discretization(qx, qy, t=1, staggered_potential=0, valley_index=0):
    import numpy as np
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

# 准一维Zigzag边石墨烯条带的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_graphene_with_zigzag_in_quasi_one_dimension(k, N=10, M=0, t=1, period=0):
    import numpy as np
    import guan
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

# Haldane模型的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_haldane_model(k1, k2, M=2/3, t1=1, t2=1/3, phi='default', a='default'):
    import numpy as np
    import cmath
    import math
    if phi == 'default':
        phi=math.pi/4
    if a == 'default':
        a=1/math.sqrt(3)
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

# 准一维Haldane模型条带的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_haldane_model_in_quasi_one_dimension(k, N=10, M=2/3, t1=1, t2=1/3, phi='default', period=0):
    import numpy as np
    import cmath
    import math
    if phi == 'default':
        phi=math.pi/4
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

# 一个量子反常霍尔效应的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_one_QAH_model(k1, k2, t1=1, t2=1, t3=0.5, m=-1):
    import numpy as np
    import math
    hamiltonian = np.zeros((2, 2), dtype=complex)
    hamiltonian[0, 1] = 2*t1*math.cos(k1)-1j*2*t1*math.cos(k2)
    hamiltonian[1, 0] = 2*t1*math.cos(k1)+1j*2*t1*math.cos(k2)
    hamiltonian[0, 0] = m+2*t3*math.sin(k1)+2*t3*math.sin(k2)+2*t2*math.cos(k1+k2)
    hamiltonian[1, 1] = -(m+2*t3*math.sin(k1)+2*t3*math.sin(k2)+2*t2*math.cos(k1+k2))
    return hamiltonian

# BHZ模型的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_bhz_model(kx, ky, A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01):
    import numpy as np
    import math
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

# 半BHZ模型的哈密顿量（自旋向上）（倒空间）
@guan.statistics_decorator
def hamiltonian_of_half_bhz_model_for_spin_up(kx, ky, A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01):
    import numpy as np
    import math
    hamiltonian = np.zeros((2, 2), dtype=complex)
    varepsilon = C-2*D*(2-math.cos(kx)-math.cos(ky))
    d3 = -2*B*(2-(M/2/B)-math.cos(kx)-math.cos(ky))
    d1_d2 = A*(math.sin(kx)+1j*math.sin(ky))
    hamiltonian[0, 0] = varepsilon+d3
    hamiltonian[1, 1] = varepsilon-d3
    hamiltonian[0, 1] = np.conj(d1_d2)
    hamiltonian[1, 0] = d1_d2
    return hamiltonian

# 半BHZ模型的哈密顿量（自旋向下）（倒空间）
@guan.statistics_decorator
def hamiltonian_of_half_bhz_model_for_spin_down(kx, ky, A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01):
    import numpy as np
    import math
    hamiltonian = np.zeros((2, 2), dtype=complex)
    varepsilon = C-2*D*(2-math.cos(kx)-math.cos(ky))
    d3 = -2*B*(2-(M/2/B)-math.cos(kx)-math.cos(ky))
    d1_d2 = A*(math.sin(kx)+1j*math.sin(ky))
    hamiltonian[0, 0] = varepsilon+d3
    hamiltonian[1, 1] = varepsilon-d3
    hamiltonian[0, 1] = -d1_d2 
    hamiltonian[1, 0] = -np.conj(d1_d2)
    return hamiltonian

# BBH模型的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_bbh_model(kx, ky, gamma_x=0.5, gamma_y=0.5, lambda_x=1, lambda_y=1):
    import numpy as np
    import cmath
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

# Kagome模型的哈密顿量（倒空间）
@guan.statistics_decorator
def hamiltonian_of_kagome_lattice(kx, ky, t=1):
    import numpy as np
    import math
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