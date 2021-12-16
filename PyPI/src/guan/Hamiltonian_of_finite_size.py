# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com/about. The primary location of this package is on website https://py.guanjihuan.com.

# Hamiltonian of finite size

import numpy as np

def finite_size_along_one_direction(N, on_site=0, hopping=1, period=0):
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

def finite_size_along_two_directions_for_square_lattice(N1, N2, on_site=0, hopping_1=1, hopping_2=1, period_1=0, period_2=0):
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

def finite_size_along_three_directions_for_cubic_lattice(N1, N2, N3, on_site=0, hopping_1=1, hopping_2=1, hopping_3=1, period_1=0, period_2=0, period_3=0):
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

def hopping_along_zigzag_direction_for_graphene(N):
    hopping = np.zeros((4*N, 4*N), dtype=complex)
    for i0 in range(N):
        hopping[4*i0+1, 4*i0+0] = 1
        hopping[4*i0+2, 4*i0+3] = 1
    return hopping

def finite_size_along_two_directions_for_graphene(N1, N2, period_1=0, period_2=0):
    on_site = finite_size_along_one_direction(4)
    hopping_1 = hopping_along_zigzag_direction_for_graphene(1)
    hopping_2 = np.zeros((4, 4), dtype=complex)
    hopping_2[3, 0] = 1
    hamiltonian = finite_size_along_two_directions_for_square_lattice(N1, N2, on_site, hopping_1, hopping_2, period_1, period_2)
    return hamiltonian