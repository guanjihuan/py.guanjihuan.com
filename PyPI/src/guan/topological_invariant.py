# Module: topological_invariant

# 通过高效法计算方格子的陈数
def calculate_chern_number_for_square_lattice_with_efficient_method(hamiltonian_function, precision=100, print_show=0):
    import numpy as np
    import math
    import cmath
    import guan
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

# 通过高效法计算方格子的陈数（可计算简并的情况）
def calculate_chern_number_for_square_lattice_with_efficient_method_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], precision=100, print_show=0): 
    import numpy as np
    import math
    import cmath
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
            # first dot product
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[i0, j0] = np.dot(np.conj(vector[:, dim1]), vector_delta_kx[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # second dot product
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[i0, j0] = np.dot(np.conj(vector_delta_kx[:, dim1]), vector_delta_kx_ky[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # third dot product
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[i0, j0] = np.dot(np.conj(vector_delta_kx_ky[:, dim1]), vector_delta_ky[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # four dot product
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[i0, j0] = np.dot(np.conj(vector_delta_ky[:, dim1]), vector[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value= det_value*dot_matrix
            chern_number += cmath.log(det_value)
    chern_number = chern_number/(2*math.pi*1j)
    return chern_number

# 通过Wilson loop方法计算方格子的陈数
def calculate_chern_number_for_square_lattice_with_wilson_loop(hamiltonian_function, precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0):
    import numpy as np
    import math
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

# 通过Wilson loop方法计算方格子的陈数（可计算简并的情况）
def calculate_chern_number_for_square_lattice_with_wilson_loop_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0):
    import numpy as np
    import math
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

# 通过高效法计算贝利曲率
def calculate_berry_curvature_with_efficient_method(hamiltonian_function, k_min='default', k_max='default', precision=100, print_show=0):
    import numpy as np
    import cmath
    import guan
    import math
    if k_min == 'default':
        k_min = -math.pi
    if k_max == 'default':
        k_max=math.pi
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

# 通过高效法计算贝利曲率（可计算简并的情况）
def calculate_berry_curvature_with_efficient_method_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], k_min='default', k_max='default', precision=100, print_show=0):
    import numpy as np
    import cmath
    import math
    if k_min == 'default':
        k_min = -math.pi
    if k_max == 'default':
        k_max=math.pi
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
            # first dot product
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[i0, j0] = np.dot(np.conj(vector[:, dim1]), vector_delta_kx[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # second dot product
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[i0, j0] = np.dot(np.conj(vector_delta_kx[:, dim1]), vector_delta_kx_ky[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # third dot product
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[i0, j0] = np.dot(np.conj(vector_delta_kx_ky[:, dim1]), vector_delta_ky[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value = det_value*dot_matrix
            # four dot product
            dot_matrix = np.zeros((dim , dim), dtype=complex)
            i0 = 0
            for dim1 in index_of_bands:
                j0 = 0
                for dim2 in index_of_bands:
                    dot_matrix[i0, j0] = np.dot(np.conj(vector_delta_ky[:, dim1]), vector[:, dim2])
                    j0 += 1
                i0 += 1
            dot_matrix = np.linalg.det(dot_matrix)/abs(np.linalg.det(dot_matrix))
            det_value= det_value*dot_matrix
            berry_curvature = cmath.log(det_value)/delta/delta*1j
            berry_curvature_array[j00, i00] = berry_curvature
            j00 += 1
        i00 += 1
    return k_array, berry_curvature_array

# 通过Wilson loop方法计算贝里曲率
def calculate_berry_curvature_with_wilson_loop(hamiltonian_function, k_min='default', k_max='default', precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0):
    import numpy as np
    import math
    if k_min == 'default':
        k_min = -math.pi
    if k_max == 'default':
        k_max=math.pi
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

# 通过Wilson loop方法计算贝里曲率（可计算简并的情况）
def calculate_berry_curvature_with_wilson_loop_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], k_min='default', k_max='default', precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0):
    import numpy as np
    import math
    if k_min == 'default':
        k_min = -math.pi
    if k_max == 'default':
        k_max=math.pi
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

# 计算蜂窝格子的陈数（高效法）
def calculate_chern_number_for_honeycomb_lattice(hamiltonian_function, a=1, precision=300, print_show=0):
    import numpy as np
    import math
    import cmath
    import guan
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

# 计算Wilson loop
def calculate_wilson_loop(hamiltonian_function, k_min='default', k_max='default', precision=100, print_show=0):
    import numpy as np
    import guan
    import math
    if k_min == 'default':
        k_min = -math.pi
    if k_max == 'default':
        k_max=math.pi
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