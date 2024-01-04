# Module: quantum_transport
import guan

# 计算电导
@guan.statistics_decorator
def calculate_conductance(fermi_energy, h00, h01, length=100):
    import numpy as np
    import copy
    import guan
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

# 计算不同费米能下的电导
@guan.statistics_decorator
def calculate_conductance_with_fermi_energy_array(fermi_energy_array, h00, h01, length=100, print_show=0):
    import numpy as np
    import guan
    dim = np.array(fermi_energy_array).shape[0]
    conductance_array = np.zeros(dim)
    i0 = 0
    for fermi_energy in fermi_energy_array:
        conductance_array[i0] = np.real(guan.calculate_conductance(fermi_energy, h00, h01, length))
        if print_show == 1:
            print(fermi_energy, conductance_array[i0])
        i0 += 1
    return conductance_array

# 计算在势垒散射下的电导
@guan.statistics_decorator
def calculate_conductance_with_barrier(fermi_energy, h00, h01, length=100, barrier_length=20, barrier_potential=1):
    import numpy as np
    import copy
    import guan
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

# 计算在无序散射下的电导
@guan.statistics_decorator
def calculate_conductance_with_disorder(fermi_energy, h00, h01, disorder_intensity=2.0, disorder_concentration=1.0, length=100, calculation_times=1):
    import numpy as np
    import copy
    import guan
    right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead(fermi_energy, h00, h01)
    dim = np.array(h00).shape[0]
    conductance_averaged = 0
    for times in range(calculation_times):
        for ix in range(length+2):
            disorder = np.zeros((dim, dim))
            for dim0 in range(dim):
                if np.random.uniform(0, 1)<=disorder_concentration:
                    disorder[dim0, dim0] = np.random.uniform(-disorder_intensity, disorder_intensity)
            if ix == 0:
                green_nn_n = guan.green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
                green_0n_n = copy.deepcopy(green_nn_n)
            elif ix != length+1:
                green_nn_n = guan.green_function_nn_n(fermi_energy, h00+disorder, h01, green_nn_n, broadening=0)
                green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
            else:
                green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
                green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
        conductance = np.trace(np.dot(np.dot(np.dot(gamma_left, green_0n_n), gamma_right), green_0n_n.transpose().conj()))
        conductance_averaged += conductance
    conductance_averaged = conductance_averaged/calculation_times
    return conductance_averaged

# 计算在无序散射下的电导（需要输入无序数组）
@guan.statistics_decorator
def calculate_conductance_with_disorder_array(fermi_energy, h00, h01, disorder_array, length=100):
    import numpy as np
    import copy
    import guan
    right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead(fermi_energy, h00, h01)
    dim = np.array(h00).shape[0]
    for ix in range(length+2):
        if ix == 0:
            green_nn_n = guan.green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
            green_0n_n = copy.deepcopy(green_nn_n)
        elif ix != length+1:
            i0 = 0
            disorder = np.diag(disorder_array[i0*dim:(i0+1)*dim])
            i0 += 1
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00+disorder, h01, green_nn_n, broadening=0)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
        else:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
    conductance = np.trace(np.dot(np.dot(np.dot(gamma_left, green_0n_n), gamma_right), green_0n_n.transpose().conj()))
    return conductance

# 计算在无序垂直切片的散射下的电导
@guan.statistics_decorator
def calculate_conductance_with_slice_disorder(fermi_energy, h00, h01, disorder_intensity=2.0, disorder_concentration=1.0, length=100):
    import numpy as np
    import copy
    import guan
    right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead(fermi_energy, h00, h01)
    dim = np.array(h00).shape[0]
    for ix in range(length+2):
        disorder = np.zeros((dim, dim))
        if np.random.uniform(0, 1)<=disorder_concentration:
            disorder = np.random.uniform(-disorder_intensity, disorder_intensity)*np.eye(dim)
        if ix == 0:
            green_nn_n = guan.green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
            green_0n_n = copy.deepcopy(green_nn_n)
        elif ix != length+1:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00+disorder, h01, green_nn_n, broadening=0)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
        else:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
    conductance = np.trace(np.dot(np.dot(np.dot(gamma_left, green_0n_n), gamma_right), green_0n_n.transpose().conj()))
    return conductance

# 计算在无序水平切片的散射下的电导
@guan.statistics_decorator
def calculate_conductance_with_disorder_inside_unit_cell_which_keeps_translational_symmetry(fermi_energy, h00, h01, disorder_intensity=2.0, disorder_concentration=1.0, length=100):
    import numpy as np
    import copy
    import guan
    right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead(fermi_energy, h00, h01)
    dim = np.array(h00).shape[0]
    disorder = np.zeros((dim, dim))
    for dim0 in range(dim):
        if np.random.uniform(0, 1)<=disorder_concentration:
            disorder[dim0, dim0] = np.random.uniform(-disorder_intensity, disorder_intensity)
    for ix in range(length+2):
        if ix == 0:
            green_nn_n = guan.green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
            green_0n_n = copy.deepcopy(green_nn_n)
        elif ix != length+1:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00+disorder, h01, green_nn_n, broadening=0)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
        else:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
    conductance = np.trace(np.dot(np.dot(np.dot(gamma_left, green_0n_n), gamma_right), green_0n_n.transpose().conj()))
    return conductance

# 计算在随机空位的散射下的电导
@guan.statistics_decorator
def calculate_conductance_with_random_vacancy(fermi_energy, h00, h01, vacancy_concentration=0.5, vacancy_potential=1e9, length=100):
    import numpy as np
    import copy
    import guan
    right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead(fermi_energy, h00, h01)
    dim = np.array(h00).shape[0]
    for ix in range(length+2):
        random_vacancy = np.zeros((dim, dim))
        for dim0 in range(dim):
            if np.random.uniform(0, 1)<=vacancy_concentration:
                random_vacancy[dim0, dim0] = vacancy_potential
        if ix == 0:
            green_nn_n = guan.green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
            green_0n_n = copy.deepcopy(green_nn_n)
        elif ix != length+1:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00+random_vacancy, h01, green_nn_n, broadening=0)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
        else:
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
            green_0n_n = guan.green_function_in_n(green_0n_n, h01, green_nn_n)
    conductance = np.trace(np.dot(np.dot(np.dot(gamma_left, green_0n_n), gamma_right), green_0n_n.transpose().conj()))
    return conductance

# 计算在不同无序散射强度下的电导
@guan.statistics_decorator
def calculate_conductance_with_disorder_intensity_array(fermi_energy, h00, h01, disorder_intensity_array, disorder_concentration=1.0, length=100, calculation_times=1, print_show=0):
    import numpy as np
    import guan
    dim = np.array(disorder_intensity_array).shape[0]
    conductance_array = np.zeros(dim)
    i0 = 0
    for disorder_intensity in disorder_intensity_array:
        for times in range(calculation_times):
            conductance_array[i0] = conductance_array[i0]+np.real(guan.calculate_conductance_with_disorder(fermi_energy, h00, h01, disorder_intensity=disorder_intensity, disorder_concentration=disorder_concentration, length=length))
        if print_show == 1:
            print(disorder_intensity, conductance_array[i0]/calculation_times)
        i0 += 1
    conductance_array = conductance_array/calculation_times
    return conductance_array

# 计算在不同无序浓度下的电导
@guan.statistics_decorator
def calculate_conductance_with_disorder_concentration_array(fermi_energy, h00, h01, disorder_concentration_array, disorder_intensity=2.0, length=100, calculation_times=1, print_show=0):
    import numpy as np
    import guan
    dim = np.array(disorder_concentration_array).shape[0]
    conductance_array = np.zeros(dim)
    i0 = 0
    for disorder_concentration in disorder_concentration_array:
        for times in range(calculation_times):
            conductance_array[i0] = conductance_array[i0]+np.real(guan.calculate_conductance_with_disorder(fermi_energy, h00, h01, disorder_intensity=disorder_intensity, disorder_concentration=disorder_concentration, length=length))
        if print_show == 1:
            print(disorder_concentration, conductance_array[i0]/calculation_times)
        i0 += 1
    conductance_array = conductance_array/calculation_times
    return conductance_array

# 计算在不同无序散射长度下的电导
@guan.statistics_decorator
def calculate_conductance_with_scattering_length_array(fermi_energy, h00, h01, length_array, disorder_intensity=2.0, disorder_concentration=1.0, calculation_times=1, print_show=0):
    import numpy as np
    import guan
    dim = np.array(length_array).shape[0]
    conductance_array = np.zeros(dim)
    i0 = 0
    for length in length_array:
        for times in range(calculation_times):
            conductance_array[i0] = conductance_array[i0]+np.real(guan.calculate_conductance_with_disorder(fermi_energy, h00, h01, disorder_intensity=disorder_intensity, disorder_concentration=disorder_concentration, length=length))
        if print_show == 1:
            print(length, conductance_array[i0]/calculation_times)
        i0 += 1
    conductance_array = conductance_array/calculation_times
    return conductance_array

# 计算得到Gamma矩阵和格林函数，用于计算六端口的量子输运
@guan.statistics_decorator
def get_gamma_array_and_green_for_six_terminal_transmission(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10):
    import numpy as np
    import guan
    #   ---------------- Geometry ----------------
    #               lead2         lead3
    #   lead1(L)                          lead4(R)  
    #               lead6         lead5 
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
    self_energy1, gamma1 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_1, h01_for_lead_1, h_lead1_to_center)
    self_energy2, gamma2 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_2, h01_for_lead_1, h_lead2_to_center)
    self_energy3, gamma3 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_3, h01_for_lead_1, h_lead3_to_center)
    self_energy4, gamma4 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_4, h01_for_lead_1, h_lead4_to_center)
    self_energy5, gamma5 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_5, h01_for_lead_1, h_lead5_to_center)
    self_energy6, gamma6 = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00_for_lead_6, h01_for_lead_1, h_lead6_to_center)
    gamma_array = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6]
    green = np.linalg.inv(fermi_energy*np.eye(internal_degree*width*length)-center_hamiltonian-self_energy1-self_energy2-self_energy3-self_energy4-self_energy5-self_energy6)
    return gamma_array, green

# 计算六端口的透射矩阵
@guan.statistics_decorator
def calculate_six_terminal_transmission_matrix(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10):
    import numpy as np
    import guan
    gamma_array, green = guan.get_gamma_array_and_green_for_six_terminal_transmission(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width, length, internal_degree, moving_step_of_leads)
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

# 计算从电极1出发的透射系数
@guan.statistics_decorator
def calculate_six_terminal_transmissions_from_lead_1(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10):
    import numpy as np
    import guan
    gamma_array, green = guan.get_gamma_array_and_green_for_six_terminal_transmission(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width, length, internal_degree, moving_step_of_leads)
    transmission_12 = np.real(np.trace(np.dot(np.dot(np.dot(gamma_array[0], green), gamma_array[1]), green.transpose().conj())))
    transmission_13 = np.real(np.trace(np.dot(np.dot(np.dot(gamma_array[0], green), gamma_array[2]), green.transpose().conj())))
    transmission_14 = np.real(np.trace(np.dot(np.dot(np.dot(gamma_array[0], green), gamma_array[3]), green.transpose().conj())))
    transmission_15 = np.real(np.trace(np.dot(np.dot(np.dot(gamma_array[0], green), gamma_array[4]), green.transpose().conj())))
    transmission_16 = np.real(np.trace(np.dot(np.dot(np.dot(gamma_array[0], green), gamma_array[5]), green.transpose().conj())))
    return transmission_12, transmission_13, transmission_14, transmission_15, transmission_16

# 通过动量k的虚部，判断通道为传播通道还是衰减通道
@guan.statistics_decorator
def if_active_channel(k_of_channel):
    import numpy as np
    if np.abs(np.imag(k_of_channel))<1e-6:
        if_active = 1
    else:
        if_active = 0
    return if_active

# 获取通道的动量和速度，用于计算散射矩阵
@guan.statistics_decorator
def get_k_and_velocity_of_channel(fermi_energy, h00, h01):
    import numpy as np
    import math
    import copy
    import guan
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

# 获取分类后的动量和速度，以及U和F，用于计算散射矩阵
@guan.statistics_decorator
def get_classified_k_velocity_u_and_f(fermi_energy, h00, h01):
    import numpy as np
    import guan
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

# 计算散射矩阵
@guan.statistics_decorator
def calculate_scattering_matrix(fermi_energy, h00, h01, length=100):
    import numpy as np
    import math
    import copy
    import guan
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

# 从散射矩阵中，获取散射矩阵的信息
@guan.statistics_decorator
def information_of_scattering_matrix(transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active):
    import numpy as np
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

# 已知h00和h01，计算散射矩阵并获得散射矩阵的信息
@guan.statistics_decorator
def calculate_scattering_matrix_and_get_information(fermi_energy, h00, h01, length=100):
    import guan
    transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active = guan.calculate_scattering_matrix(fermi_energy, h00, h01, length=length)

    number_of_active_channels, number_of_evanescent_channels, k_of_right_moving_active_channels, k_of_left_moving_active_channels, velocity_of_right_moving_active_channels, velocity_of_left_moving_active_channels, transmission_matrix_for_active_channels, reflection_matrix_for_active_channels, total_transmission_of_channels, total_conductance, total_reflection_of_channels, sum_of_transmission_and_reflection_of_channels = guan.information_of_scattering_matrix(transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active)

    return number_of_active_channels, number_of_evanescent_channels, k_of_right_moving_active_channels, k_of_left_moving_active_channels, velocity_of_right_moving_active_channels, velocity_of_left_moving_active_channels, transmission_matrix_for_active_channels, reflection_matrix_for_active_channels, total_transmission_of_channels, total_conductance, total_reflection_of_channels, sum_of_transmission_and_reflection_of_channels

# 从散射矩阵中打印出散射矩阵的信息
@guan.statistics_decorator
def print_or_write_scattering_matrix_with_information_of_scattering_matrix(number_of_active_channels, number_of_evanescent_channels, k_of_right_moving_active_channels, k_of_left_moving_active_channels, velocity_of_right_moving_active_channels, velocity_of_left_moving_active_channels, transmission_matrix_for_active_channels, reflection_matrix_for_active_channels, total_transmission_of_channels, total_conductance, total_reflection_of_channels, sum_of_transmission_and_reflection_of_channels, print_show=1, write_file=0, filename='a', file_format='.txt'):
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

# 已知h00和h01，计算散射矩阵并打印出散射矩阵的信息
@guan.statistics_decorator
def print_or_write_scattering_matrix(fermi_energy, h00, h01, length=100, print_show=1, write_file=0, filename='a', file_format='.txt'):
    import guan
    transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active = guan.calculate_scattering_matrix(fermi_energy, h00, h01, length=length)

    number_of_active_channels, number_of_evanescent_channels, k_of_right_moving_active_channels, k_of_left_moving_active_channels, velocity_of_right_moving_active_channels, velocity_of_left_moving_active_channels, transmission_matrix_for_active_channels, reflection_matrix_for_active_channels, total_transmission_of_channels, total_conductance, total_reflection_of_channels, sum_of_transmission_and_reflection_of_channels = guan.information_of_scattering_matrix(transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active)

    guan.print_or_write_scattering_matrix_with_information_of_scattering_matrix(number_of_active_channels, number_of_evanescent_channels, k_of_right_moving_active_channels, k_of_left_moving_active_channels, velocity_of_right_moving_active_channels, velocity_of_left_moving_active_channels, transmission_matrix_for_active_channels, reflection_matrix_for_active_channels, total_transmission_of_channels, total_conductance, total_reflection_of_channels, sum_of_transmission_and_reflection_of_channels, print_show=print_show, write_file=write_file, filename=filename, file_format=file_format)

# 在无序下，计算散射矩阵
@guan.statistics_decorator
def calculate_scattering_matrix_with_disorder(fermi_energy, h00, h01, length=100, disorder_intensity=2.0, disorder_concentration=1.0):
    import numpy as np
    import math
    import copy
    import guan
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    k_right, k_left, velocity_right, velocity_left, f_right, f_left, u_right, u_left, ind_right_active = guan.get_classified_k_velocity_u_and_f(fermi_energy, h00, h01)
    right_self_energy = np.dot(h01, f_right)
    left_self_energy = np.dot(h01.transpose().conj(), np.linalg.inv(f_left))
    for i0 in range(length):
        disorder = np.zeros((dim, dim))
        for dim0 in range(dim):
            if np.random.uniform(0, 1)<=disorder_concentration:
                disorder[dim0, dim0] = np.random.uniform(-disorder_intensity, disorder_intensity)
        if i0 == 0:
            green_nn_n = guan.green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
            green_00_n = copy.deepcopy(green_nn_n)
            green_0n_n = copy.deepcopy(green_nn_n)
            green_n0_n = copy.deepcopy(green_nn_n)
        elif i0 != length-1: 
            green_nn_n = guan.green_function_nn_n(fermi_energy, h00+disorder, h01, green_nn_n, broadening=0) 
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

# 在无序下，计算散射矩阵，并获取散射矩阵多次计算的平均信息
@guan.statistics_decorator
def calculate_scattering_matrix_with_disorder_and_get_averaged_information(fermi_energy, h00, h01, length=100, disorder_intensity=2.0, disorder_concentration=1.0, calculation_times=1):
    import guan
    transmission_matrix_for_active_channels_averaged = 0
    reflection_matrix_for_active_channels_averaged = 0
    for i0 in range(calculation_times):
        transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active = guan.calculate_scattering_matrix_with_disorder(fermi_energy, h00, h01, length, disorder_intensity, disorder_concentration)

        number_of_active_channels, number_of_evanescent_channels, k_of_right_moving_active_channels, k_of_left_moving_active_channels, velocity_of_right_moving_active_channels, velocity_of_left_moving_active_channels, transmission_matrix_for_active_channels, reflection_matrix_for_active_channels, total_transmission_of_channels, total_conductance, total_reflection_of_channels, sum_of_transmission_and_reflection_of_channels = guan.information_of_scattering_matrix(transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active)

        transmission_matrix_for_active_channels_averaged += transmission_matrix_for_active_channels
        reflection_matrix_for_active_channels_averaged += reflection_matrix_for_active_channels
    transmission_matrix_for_active_channels_averaged = transmission_matrix_for_active_channels_averaged/calculation_times
    reflection_matrix_for_active_channels_averaged = reflection_matrix_for_active_channels_averaged/calculation_times
    return transmission_matrix_for_active_channels_averaged, reflection_matrix_for_active_channels_averaged