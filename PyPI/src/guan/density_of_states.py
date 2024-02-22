# Module: density_of_states

# 计算体系的总态密度
def total_density_of_states(fermi_energy, hamiltonian, broadening=0.01):
    import numpy as np
    import math
    import guan
    green = guan.green_function(fermi_energy, hamiltonian, broadening)
    total_dos = -np.trace(np.imag(green))/math.pi
    return total_dos

# 对于不同费米能，计算体系的总态密度
def total_density_of_states_with_fermi_energy_array(fermi_energy_array, hamiltonian, broadening=0.01, print_show=0):
    import numpy as np
    import guan
    dim = np.array(fermi_energy_array).shape[0]
    total_dos_array = np.zeros(dim)
    i0 = 0
    for fermi_energy in fermi_energy_array:
        if print_show == 1:
            print(fermi_energy)
        total_dos_array[i0] = guan.total_density_of_states(fermi_energy, hamiltonian, broadening)
        i0 += 1
    return total_dos_array

# 计算方格子的局域态密度（其中，哈密顿量的维度为：dim_hamiltonian = N1*N2*internal_degree）
def local_density_of_states_for_square_lattice(fermi_energy, hamiltonian, N1, N2, internal_degree=1, broadening=0.01):
    import numpy as np
    import math
    import guan
    green = guan.green_function(fermi_energy, hamiltonian, broadening)
    local_dos = np.zeros((N2, N1))
    for i1 in range(N1):
        for i2 in range(N2):
            for i in range(internal_degree): 
                local_dos[i2, i1] = local_dos[i2, i1]-np.imag(green[i1*N2*internal_degree+i2*internal_degree+i, i1*N2*internal_degree+i2*internal_degree+i])/math.pi
    return local_dos

# 计算立方格子的局域态密度（其中，哈密顿量的维度为：dim_hamiltonian = N1*N2*N3*internal_degree）
def local_density_of_states_for_cubic_lattice(fermi_energy, hamiltonian, N1, N2, N3, internal_degree=1, broadening=0.01):
    import numpy as np
    import math
    import guan
    green = guan.green_function(fermi_energy, hamiltonian, broadening)
    local_dos = np.zeros((N3, N2, N1))
    for i1 in range(N1):
        for i2 in range(N2):
            for i3 in range(N3):
                for i in range(internal_degree): 
                    local_dos[i3, i2, i1] = local_dos[i3, i2, i1]-np.imag(green[i1*N2*N3*internal_degree+i2*N3*internal_degree+i3*internal_degree+i, i1*N2*N3*internal_degree+i2*N3*internal_degree+i3*internal_degree+i])/math.pi
    return local_dos

# 使用Dyson方程，计算方格子的局域态密度（其中，h00的维度为：dim_h00 = N2*internal_degree）
def local_density_of_states_for_square_lattice_using_dyson_equation(fermi_energy, h00, h01, N2, N1, internal_degree=1, broadening=0.01):
    import numpy as np
    import math
    import guan
    local_dos = np.zeros((N2, N1))
    green_11_1 = guan.green_function(fermi_energy, h00, broadening)
    for i1 in range(N1):
        green_nn_n_minus = green_11_1
        green_in_n_minus = green_11_1
        green_ni_n_minus = green_11_1
        green_ii_n_minus = green_11_1
        for _ in range(i1):
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
        if i1!=0:
            green_in_n_minus = green_nn_n
            green_ni_n_minus = green_nn_n
            green_ii_n_minus = green_nn_n
        for _ in range(N1-1-i1):
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

# 使用Dyson方程，计算方格子的局域态密度，方法二（其中，h00的维度为：dim_h00 = N2*internal_degree）
def local_density_of_states_for_square_lattice_using_dyson_equation_with_second_method(fermi_energy, h00, h01, N2, N1, internal_degree=1, broadening=0.01):
    import numpy as np
    import math
    import guan
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]   
    local_dos = np.zeros((N2, N1))
    green_11_1 = guan.green_function(fermi_energy, h00, broadening)
    for i1 in range(N1):
        green_nn_n_right_minus = green_11_1
        green_nn_n_left_minus = green_11_1
        if i1!=0:
            for _ in range(i1-1):
                green_nn_n_right = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_right_minus, broadening)
                green_nn_n_right_minus = green_nn_n_right
        if i1!=N1-1:
            for _ in range(N1-i1-2):
                G_nn_n_left = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_left_minus, broadening)
                green_nn_n_left_minus = G_nn_n_left
        if i1==0:
            green_ii_n = np.linalg.inv((fermi_energy+broadening*1j)*np.identity(dim)-h00-np.dot(np.dot(h01, green_nn_n_left_minus), h01.transpose().conj()))
        elif i1!=0 and i1!=N1-1:
            green_ii_n = np.linalg.inv((fermi_energy+broadening*1j)*np.identity(dim)-h00-np.dot(np.dot(h01.transpose().conj(), green_nn_n_right_minus), h01)-np.dot(np.dot(h01, green_nn_n_left_minus), h01.transpose().conj()))
        elif i1==N1-1: 
            green_ii_n = np.linalg.inv((fermi_energy+broadening*1j)*np.identity(dim)-h00-np.dot(np.dot(h01.transpose().conj(), green_nn_n_right_minus), h01))
        for i2 in range(N2):
            for i in range(internal_degree):
                local_dos[i2, i1] = local_dos[i2, i1] - np.imag(green_ii_n[i2*internal_degree+i, i2*internal_degree+i])/math.pi
    return local_dos

# 使用Dyson方程，计算立方格子的局域态密度（其中，h00的维度为：dim_h00 = N2*N3*internal_degree）
def local_density_of_states_for_cubic_lattice_using_dyson_equation(fermi_energy, h00, h01, N3, N2, N1, internal_degree=1, broadening=0.01):
    import numpy as np
    import math
    import guan
    local_dos = np.zeros((N3, N2, N1))
    green_11_1 = guan.green_function(fermi_energy, h00, broadening)
    for i1 in range(N1):
        green_nn_n_minus = green_11_1
        green_in_n_minus = green_11_1
        green_ni_n_minus = green_11_1
        green_ii_n_minus = green_11_1
        for _ in range(i1):
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening)
            green_nn_n_minus = green_nn_n
        if i1!=0:
            green_in_n_minus = green_nn_n
            green_ni_n_minus = green_nn_n
            green_ii_n_minus = green_nn_n
        for _ in range(N1-1-i1):
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

# 使用Dyson方程，计算方格子条带（考虑了电极自能）的局域态密度（其中，h00的维度为：dim_h00 = N2*internal_degree）
def local_density_of_states_for_square_lattice_with_self_energy_using_dyson_equation(fermi_energy, h00, h01, N2, N1, right_self_energy, left_self_energy, internal_degree=1, broadening=0.01):
    import numpy as np
    import math
    import guan
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
