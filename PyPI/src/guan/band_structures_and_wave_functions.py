# Module: band_structures_and_wave_functions
import guan

# 计算哈密顿量的本征值
@guan.statistics_decorator
def calculate_eigenvalue(hamiltonian):
    import numpy as np
    if np.array(hamiltonian).shape==():
        eigenvalue = np.real(hamiltonian)
    else:
        eigenvalue, eigenvector = np.linalg.eigh(hamiltonian)
    return eigenvalue

# 输入哈密顿量函数（带一组参数），计算一组参数下的本征值，返回本征值向量组
@guan.statistics_decorator
def calculate_eigenvalue_with_one_parameter(x_array, hamiltonian_function, print_show=0):
    import numpy as np
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

# 输入哈密顿量函数（带两组参数），计算两组参数下的本征值，返回本征值向量组
@guan.statistics_decorator
def calculate_eigenvalue_with_two_parameters(x_array, y_array, hamiltonian_function, print_show=0, print_show_more=0):  
    import numpy as np
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

# 计算哈密顿量的本征矢
@guan.statistics_decorator
def calculate_eigenvector(hamiltonian):
    import numpy as np
    eigenvalue, eigenvector = np.linalg.eigh(hamiltonian)
    return eigenvector

# 通过二分查找的方法获取和相邻波函数一样规范的波函数
@guan.statistics_decorator
def find_vector_with_the_same_gauge_with_binary_search(vector_target, vector_ref, show_error=1, show_times=0, show_phase=0, n_test=1000, precision=1e-6):
    import numpy as np
    import cmath
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

# 通过乘一个相反的相位角，实现波函数的一个非零分量为实数，从而得到固定规范的波函数
@guan.statistics_decorator
def find_vector_with_fixed_gauge_by_making_one_component_real(vector, index=None):
    import numpy as np
    import cmath
    vector = np.array(vector)
    if index == None:
        index = np.argmax(np.abs(vector))
    angle = cmath.phase(vector[index])
    vector = vector*cmath.exp(-1j*angle)
    return vector

# 通过乘一个相反的相位角，实现波函数的一个非零分量为实数，从而得到固定规范的波函数（在一组波函数中选取最大的那个分量）
@guan.statistics_decorator
def find_vector_array_with_fixed_gauge_by_making_one_component_real(vector_array):
    import numpy as np
    import guan
    vector_sum = 0
    Num_k = np.array(vector_array).shape[0]
    for i0 in range(Num_k):
        vector_sum += np.abs(vector_array[i0])
    index = np.argmax(np.abs(vector_sum))
    for i0 in range(Num_k):
        vector_array[i0] = guan.find_vector_with_fixed_gauge_by_making_one_component_real(vector_array[i0], index=index)
    return vector_array

# 循环查找规范使得波函数的一个非零分量为实数，得到固定规范的波函数
@guan.statistics_decorator
def loop_find_vector_with_fixed_gauge_by_making_one_component_real(vector, precision=0.005, index=None):
    import numpy as np
    import cmath
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

# 循环查找规范使得波函数的一个非零分量为实数，得到固定规范的波函数（在一组波函数中选取最大的那个分量）
@guan.statistics_decorator
def loop_find_vector_array_with_fixed_gauge_by_making_one_component_real(vector_array, precision=0.005):
    import numpy as np
    import guan
    vector_sum = 0
    Num_k = np.array(vector_array).shape[0]
    for i0 in range(Num_k):
        vector_sum += np.abs(vector_array[i0])
    index = np.argmax(np.abs(vector_sum))
    for i0 in range(Num_k):
        vector_array[i0] = guan.loop_find_vector_with_fixed_gauge_by_making_one_component_real(vector_array[i0], precision=precision, index=index)
    return vector_array

# 旋转两个简并的波函数（说明：参数比较多，算法效率不高）
@guan.statistics_decorator
def rotation_of_degenerate_vectors(vector1, vector2, index1=None, index2=None, precision=0.01, criterion=0.01, show_theta=0):
    import numpy as np
    import math
    import cmath
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

# 旋转两个简并的波函数向量组（说明：参数比较多，算法效率不高）
@guan.statistics_decorator
def rotation_of_degenerate_vectors_array(vector1_array, vector2_array, precision=0.01, criterion=0.01, show_theta=0):
    import numpy as np
    import guan
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

# 在一组数据中找到数值相近的数
@guan.statistics_decorator
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

# 寻找能带的简并点
@guan.statistics_decorator
def find_degenerate_points(k_array, eigenvalue_array, precision=1e-2):
    import guan
    degenerate_k_array = []
    degenerate_eigenvalue_array = []
    i0 = 0
    for k in k_array:
        degenerate_points = guan.find_close_values_in_one_array(eigenvalue_array[i0], precision=precision)
        if len(degenerate_points) != 0:
            degenerate_k_array.append(k)
            degenerate_eigenvalue_array.append(degenerate_points)
        i0 += 1
    return degenerate_k_array, degenerate_eigenvalue_array