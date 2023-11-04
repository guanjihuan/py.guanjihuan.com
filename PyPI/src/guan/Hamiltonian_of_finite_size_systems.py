# Module: Hamiltonian_of_finite_size_systems

# 构建一维的有限尺寸体系哈密顿量（可设置是否为周期边界条件）
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
    import guan
    guan.statistics_of_guan_package()
    return hamiltonian

# 构建二维的方格子有限尺寸体系哈密顿量（可设置是否为周期边界条件）
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
    import guan
    guan.statistics_of_guan_package()
    return hamiltonian

# 构建三维的立方格子有限尺寸体系哈密顿量（可设置是否为周期边界条件）
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
    import guan
    guan.statistics_of_guan_package()
    return hamiltonian

# 构建有限尺寸的SSH模型哈密顿量
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
    import guan
    guan.statistics_of_guan_package()
    return hamiltonian

# 获取Zigzag边的石墨烯条带的元胞间跃迁
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
    import guan
    guan.statistics_of_guan_package()
    return hopping

# 构建有限尺寸的石墨烯哈密顿量（可设置是否为周期边界条件）
def hamiltonian_of_finite_size_system_along_two_directions_for_graphene(N1, N2, period_1=0, period_2=0):
    import numpy as np
    import guan
    on_site = guan.hamiltonian_of_finite_size_system_along_one_direction(4)
    hopping_1 = guan.get_hopping_term_of_graphene_ribbon_along_zigzag_direction(1)
    hopping_2 = np.zeros((4, 4), dtype=complex)
    hopping_2[3, 0] = 1
    hamiltonian = guan.hamiltonian_of_finite_size_system_along_two_directions_for_square_lattice(N1, N2, on_site, hopping_1, hopping_2, period_1, period_2)
    guan.statistics_of_guan_package()
    return hamiltonian

# 获取石墨烯有效模型沿着x方向的在位能和跃迁项（其中，动量qy为参数）
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
    import guan
    guan.statistics_of_guan_package()
    return h00, h01

# 获取BHZ模型的在位能和跃迁项
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
    import guan
    guan.statistics_of_guan_package()
    return H0, H1, H2

# 获取半个BHZ模型的在位能和跃迁项（自旋向上）
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
    import guan
    guan.statistics_of_guan_package()
    return H0, H1, H2

# 获取半个BHZ模型的在位能和跃迁项（自旋向下）
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
    import guan
    guan.statistics_of_guan_package()
    return H0, H1, H2