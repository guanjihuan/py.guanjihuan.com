# API Reference shows all functions in Guan package. The current version is guan-0.0.185, updated on December 05, 2023.

import guan




























# Module 1: basic functions

guan.test()

sigma_0 = guan.sigma_0()

sigma_x = guan.sigma_x()

sigma_y = guan.sigma_y()

sigma_z = guan.sigma_z()

sigma_00 = guan.sigma_00()

sigma_0x = guan.sigma_0x()

sigma_0y = guan.sigma_0y()

sigma_0z = guan.sigma_0z()

sigma_x0 = guan.sigma_x0()

sigma_xx = guan.sigma_xx()

sigma_xy = guan.sigma_xy()

sigma_xz = guan.sigma_xz()

sigma_y0 = guan.sigma_y0()

sigma_yx = guan.sigma_yx()

sigma_yy = guan.sigma_yy()

sigma_yz = guan.sigma_yz()

sigma_z0 = guan.sigma_z0()

sigma_zx = guan.sigma_zx()

sigma_zy = guan.sigma_zy()

sigma_zz = guan.sigma_zz()































# Module 2: Fourier transform

# 通过元胞和跃迁项得到一维的哈密顿量（需要输入k值）
hamiltonian = guan.one_dimensional_fourier_transform(k, unit_cell, hopping)

# 通过元胞和跃迁项得到二维方格子的哈密顿量（需要输入k值）
hamiltonian = guan.two_dimensional_fourier_transform_for_square_lattice(k1, k2, unit_cell, hopping_1, hopping_2)

# 通过元胞和跃迁项得到三维立方格子的哈密顿量（需要输入k值）
hamiltonian = guan.three_dimensional_fourier_transform_for_cubic_lattice(k1, k2, k3, unit_cell, hopping_1, hopping_2, hopping_3)

# 通过元胞和跃迁项得到一维的哈密顿量（返回的哈密顿量为携带k的函数）
hamiltonian_function = guan.one_dimensional_fourier_transform_with_k(unit_cell, hopping)

# 通过元胞和跃迁项得到二维方格子的哈密顿量（返回的哈密顿量为携带k的函数）
hamiltonian_function = guan.two_dimensional_fourier_transform_for_square_lattice_with_k1_k2(unit_cell, hopping_1, hopping_2)

# 通过元胞和跃迁项得到三维立方格子的哈密顿量（返回的哈密顿量为携带k的函数）
hamiltonian_function = guan.three_dimensional_fourier_transform_for_cubic_lattice_with_k1_k2_k3(unit_cell, hopping_1, hopping_2, hopping_3)

# 由实空间格矢得到倒空间格矢（一维）
b1 = guan.calculate_one_dimensional_reciprocal_lattice_vector(a1)

# 由实空间格矢得到倒空间格矢（二维）
b1, b2 = guan.calculate_two_dimensional_reciprocal_lattice_vectors(a1, a2)

# 由实空间格矢得到倒空间格矢（三维）
b1, b2, b3 = guan.calculate_three_dimensional_reciprocal_lattice_vectors(a1, a2, a3)

# 由实空间格矢得到倒空间格矢（一维），这里为符号运算
b1 = guan.calculate_one_dimensional_reciprocal_lattice_vector_with_sympy(a1)

# 由实空间格矢得到倒空间格矢（二维），这里为符号运算
b1, b2 = guan.calculate_two_dimensional_reciprocal_lattice_vectors_with_sympy(a1, a2)

# 由实空间格矢得到倒空间格矢（三维），这里为符号运算
b1, b2, b3 = guan.calculate_three_dimensional_reciprocal_lattice_vectors_with_sympy(a1, a2, a3)






























# Module 3: Hamiltonian of finite size systems

# 构建一维的有限尺寸体系哈密顿量（可设置是否为周期边界条件）
hamiltonian = guan.hamiltonian_of_finite_size_system_along_one_direction(N, on_site=0, hopping=1, period=0)

# 构建二维的方格子有限尺寸体系哈密顿量（可设置是否为周期边界条件）
hamiltonian = guan.hamiltonian_of_finite_size_system_along_two_directions_for_square_lattice(N1, N2, on_site=0, hopping_1=1, hopping_2=1, period_1=0, period_2=0)

# 构建三维的立方格子有限尺寸体系哈密顿量（可设置是否为周期边界条件）
hamiltonian = guan.hamiltonian_of_finite_size_system_along_three_directions_for_cubic_lattice(N1, N2, N3, on_site=0, hopping_1=1, hopping_2=1, hopping_3=1, period_1=0, period_2=0, period_3=0)

# 构建有限尺寸的SSH模型哈密顿量
hamiltonian = guan.hamiltonian_of_finite_size_ssh_model(N, v=0.6, w=1, onsite_1=0, onsite_2=0, period=1)

# 获取Zigzag边的石墨烯条带的元胞间跃迁
hopping = guan.get_hopping_term_of_graphene_ribbon_along_zigzag_direction(N, eta=0)

# 构建有限尺寸的石墨烯哈密顿量（可设置是否为周期边界条件）
hamiltonian = guan.hamiltonian_of_finite_size_system_along_two_directions_for_graphene(N1, N2, period_1=0, period_2=0)

# 获取石墨烯有效模型沿着x方向的在位能和跃迁项（其中，动量qy为参数）
h00, h01 = guan.get_onsite_and_hopping_terms_of_2d_effective_graphene_along_one_direction(qy, t=1, staggered_potential=0, eta=0, valley_index=0)

# 获取BHZ模型的在位能和跃迁项
H0, H1, H2 = guan.get_onsite_and_hopping_terms_of_bhz_model(A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01, a=1)

# 获取半个BHZ模型的在位能和跃迁项（自旋向上）
H0, H1, H2 = guan.get_onsite_and_hopping_terms_of_half_bhz_model_for_spin_up(A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01, a=1)

# 获取半个BHZ模型的在位能和跃迁项（自旋向下）
H0, H1, H2 = guan.get_onsite_and_hopping_terms_of_half_bhz_model_for_spin_down(A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01, a=1)























# Module 4: Hamiltonian of models in the reciprocal space

# 一维链的哈密顿量
hamiltonian = guan.hamiltonian_of_simple_chain(k)

# 二维方格子的哈密顿量
hamiltonian = guan.hamiltonian_of_square_lattice(k1, k2)

# 准一维方格子条带的哈密顿量
hamiltonian = guan.hamiltonian_of_square_lattice_in_quasi_one_dimension(k, N=10, period=0)

# 三维立方格子的哈密顿量
hamiltonian = guan.hamiltonian_of_cubic_lattice(k1, k2, k3)

# SSH模型的哈密顿量
hamiltonian = guan.hamiltonian_of_ssh_model(k, v=0.6, w=1)

# 石墨烯的哈密顿量
hamiltonian = guan.hamiltonian_of_graphene(k1, k2, staggered_potential=0, t=1, a='default')

# 石墨烯有效模型的哈密顿量
hamiltonian = guan.effective_hamiltonian_of_graphene(qx, qy, t=1, staggered_potential=0, valley_index=0)

# 石墨烯有效模型离散化后的哈密顿量
hamiltonian = guan.effective_hamiltonian_of_graphene_after_discretization(qx, qy, t=1, staggered_potential=0, valley_index=0)

# 准一维Zigzag边石墨烯条带的哈密顿量
hamiltonian = guan.hamiltonian_of_graphene_with_zigzag_in_quasi_one_dimension(k, N=10, M=0, t=1, period=0)

# Haldane模型的哈密顿量
hamiltonian = guan.hamiltonian_of_haldane_model(k1, k2, M=2/3, t1=1, t2=1/3, phi='default', a='default')

# 准一维Haldane模型条带的哈密顿量
hamiltonian = guan.hamiltonian_of_haldane_model_in_quasi_one_dimension(k, N=10, M=2/3, t1=1, t2=1/3, phi='default', period=0)

# 一个量子反常霍尔效应的哈密顿量
hamiltonian = guan.hamiltonian_of_one_QAH_model(k1, k2, t1=1, t2=1, t3=0.5, m=-1)

# BHZ模型的哈密顿量
hamiltonian = guan.hamiltonian_of_bhz_model(kx, ky, A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01)

# 半BHZ模型的哈密顿量（自旋向上）
hamiltonian = guan.hamiltonian_of_half_bhz_model_for_spin_up(kx, ky, A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01)

# 半BHZ模型的哈密顿量（自旋向下）
hamiltonian = guan.hamiltonian_of_half_bhz_model_for_spin_down(kx, ky, A=0.3645/5, B=-0.686/25, C=0, D=-0.512/25, M=-0.01)

# BBH模型的哈密顿量
hamiltonian = guan.hamiltonian_of_bbh_model(kx, ky, gamma_x=0.5, gamma_y=0.5, lambda_x=1, lambda_y=1)

# Kagome模型的哈密顿量
hamiltonian = guan.hamiltonian_of_kagome_lattice(kx, ky, t=1)

























# Module 5: band structures and wave functions

# 计算哈密顿量的本征值
eigenvalue = guan.calculate_eigenvalue(hamiltonian)

# 输入哈密顿量函数（带一组参数），计算一组参数下的本征值，返回本征值向量组
eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(x_array, hamiltonian_function, print_show=0)

# 输入哈密顿量函数（带两组参数），计算两组参数下的本征值，返回本征值向量组
eigenvalue_array = guan.calculate_eigenvalue_with_two_parameters(x_array, y_array, hamiltonian_function, print_show=0, print_show_more=0)

# 计算哈密顿量的本征矢
eigenvector = guan.calculate_eigenvector(hamiltonian)

# 通过二分查找的方法获取和相邻波函数一样规范的波函数
vector_target = guan.find_vector_with_the_same_gauge_with_binary_search(vector_target, vector_ref, show_error=1, show_times=0, show_phase=0, n_test=1000, precision=1e-6)

# 通过使得波函数的一个非零分量为实数，得到固定规范的波函数
vector = guan.find_vector_with_fixed_gauge_by_making_one_component_real(vector, precision=0.005, index=None)

# 通过使得波函数的一个非零分量为实数，得到固定规范的波函数（在一组波函数中选取最大的那个分量）
vector_array = guan.find_vector_array_with_fixed_gauge_by_making_one_component_real(vector_array, precision=0.005)

# 旋转两个简并的波函数（说明：参数比较多，效率不高）
vector1, vector2 = guan.rotation_of_degenerate_vectors(vector1, vector2, index1=None, index2=None, precision=0.01, criterion=0.01, show_theta=0)

# 旋转两个简并的波函数向量组（说明：参数比较多，效率不高）
vector1_array, vector2_array = guan.rotation_of_degenerate_vectors_array(vector1_array, vector2_array, precision=0.01, criterion=0.01, show_theta=0)





























# Module 6: Green functions

# 输入哈密顿量，得到格林函数
green = guan.green_function(fermi_energy, hamiltonian, broadening, self_energy=0)

# 在Dyson方程中的一个中间格林函数G_{nn}^{n}
green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening, self_energy=0)

# 在Dyson方程中的一个中间格林函数G_{in}^{n}
green_in_n = guan.green_function_in_n(green_in_n_minus, h01, green_nn_n)

# 在Dyson方程中的一个中间格林函数G_{ni}^{n}
green_ni_n = guan.green_function_ni_n(green_nn_n, h01, green_ni_n_minus)

# 在Dyson方程中的一个中间格林函数G_{ii}^{n}
green_ii_n = guan.green_function_ii_n(green_ii_n_minus, green_in_n_minus, h01, green_nn_n, green_ni_n_minus)

# 计算转移矩阵（该矩阵可以用来计算表面格林函数）
transfer = guan.transfer_matrix(fermi_energy, h00, h01)

# 计算电极的表面格林函数
right_lead_surface, left_lead_surface = guan.surface_green_function_of_lead(fermi_energy, h00, h01)

# 计算电极的自能（基于Dyson方程的小矩阵形式）
right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead(fermi_energy, h00, h01)

# 计算电极的自能（基于中心区整体的大矩阵形式）
right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead_with_h_LC_and_h_CR(fermi_energy, h00, h01, h_LC, h_CR)

# 计算电极的自能（基于中心区整体的大矩阵形式，可适用于多端电导的计算）
self_energy, gamma = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00, h01, h_lead_to_center)

# 计算考虑电极自能后的中心区的格林函数
green, gamma_right, gamma_left = guan.green_function_with_leads(fermi_energy, h00, h01, h_LC, h_CR, center_hamiltonian)

# 计算用于计算局域电流的格林函数G_n
G_n = guan.electron_correlation_function_green_n_for_local_current(fermi_energy, h00, h01, h_LC, h_CR, center_hamiltonian)

































# Module 7: density of states

# 计算体系的总态密度
total_dos = guan.total_density_of_states(fermi_energy, hamiltonian, broadening=0.01)

# 对于不同费米能，计算体系的总态密度
total_dos_array = guan.total_density_of_states_with_fermi_energy_array(fermi_energy_array, hamiltonian, broadening=0.01, print_show=0)

# 计算方格子的局域态密度（其中，哈密顿量的维度为：dim_hamiltonian = N1*N2*internal_degree）
local_dos = guan.local_density_of_states_for_square_lattice(fermi_energy, hamiltonian, N1, N2, internal_degree=1, broadening=0.01)

# 计算立方格子的局域态密度（其中，哈密顿量的维度为：dim_hamiltonian = N1*N2*N3*internal_degree）
local_dos = guan.local_density_of_states_for_cubic_lattice(fermi_energy, hamiltonian, N1, N2, N3, internal_degree=1, broadening=0.01)

# 利用Dyson方程，计算方格子的局域态密度（其中，h00的维度为：dim_h00 = N2*internal_degree）
local_dos = guan.local_density_of_states_for_square_lattice_using_dyson_equation(fermi_energy, h00, h01, N2, N1, internal_degree=1, broadening=0.01)

# 利用Dyson方程，计算立方格子的局域态密度（其中，h00的维度为：dim_h00 = N2*N3*internal_degree）
local_dos = guan.local_density_of_states_for_cubic_lattice_using_dyson_equation(fermi_energy, h00, h01, N3, N2, N1, internal_degree=1, broadening=0.01)

# 利用Dyson方程，计算方格子条带（考虑了电极自能）的局域态密度（其中，h00的维度为：dim_h00 = N2*internal_degree）
local_dos = guan.local_density_of_states_for_square_lattice_with_self_energy_using_dyson_equation(fermi_energy, h00, h01, N2, N1, right_self_energy, left_self_energy, internal_degree=1, broadening=0.01)





























# Module 8: quantum transport

# 计算电导
conductance = guan.calculate_conductance(fermi_energy, h00, h01, length=100)

# 计算不同费米能下的电导
conductance_array = guan.calculate_conductance_with_fermi_energy_array(fermi_energy_array, h00, h01, length=100, print_show=0)

# 计算在势垒散射下的电导
conductance = guan.calculate_conductance_with_barrier(fermi_energy, h00, h01, length=100, barrier_length=20, barrier_potential=1)

# 计算在无序散射下的电导
conductance = guan.calculate_conductance_with_disorder(fermi_energy, h00, h01, disorder_intensity=2.0, disorder_concentration=1.0, length=100, calculation_times=1)

# 计算在无序垂直切片的散射下的电导
conductance = guan.calculate_conductance_with_slice_disorder(fermi_energy, h00, h01, disorder_intensity=2.0, disorder_concentration=1.0, length=100)

# 计算在无序水平切片的散射下的电导
conductance = guan.calculate_conductance_with_disorder_inside_unit_cell_which_keeps_translational_symmetry(fermi_energy, h00, h01, disorder_intensity=2.0, disorder_concentration=1.0, length=100)

# 计算在随机空位的散射下的电导
conductance = guan.calculate_conductance_with_random_vacancy(fermi_energy, h00, h01, vacancy_concentration=0.5, vacancy_potential=1e9, length=100)

# 计算在不同无序散射强度下的电导
conductance_array = guan.calculate_conductance_with_disorder_intensity_array(fermi_energy, h00, h01, disorder_intensity_array, disorder_concentration=1.0, length=100, calculation_times=1, print_show=0)

# 计算在不同无序浓度下的电导
conductance_array = guan.calculate_conductance_with_disorder_concentration_array(fermi_energy, h00, h01, disorder_concentration_array, disorder_intensity=2.0, length=100, calculation_times=1, print_show=0)

# 计算在不同无序散射长度下的电导
conductance_array = guan.calculate_conductance_with_scattering_length_array(fermi_energy, h00, h01, length_array, disorder_intensity=2.0, disorder_concentration=1.0, calculation_times=1, print_show=0)

# 计算得到Gamma矩阵和格林函数，用于计算六端口的量子输运
gamma_array, green = guan.get_gamma_array_and_green_for_six_terminal_transmission(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10)

# 计算六端口的透射矩阵
transmission_matrix = guan.calculate_six_terminal_transmission_matrix(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10)

# 计算从电极1出发的透射系数
transmission_12, transmission_13, transmission_14, transmission_15, transmission_16 = guan.calculate_six_terminal_transmissions_from_lead_1(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10)

# 通过动量k的虚部，判断通道为传播通道还是衰减通道
if_active = guan.if_active_channel(k_of_channel)

# 获取通道的动量和速度，用于计算散射矩阵
k_of_channel, velocity_of_channel, eigenvalue, eigenvector = guan.get_k_and_velocity_of_channel(fermi_energy, h00, h01)

# 获取分类后的动量和速度，以及U和F，用于计算散射矩阵
k_right, k_left, velocity_right, velocity_left, f_right, f_left, u_right, u_left, ind_right_active = guan.get_classified_k_velocity_u_and_f(fermi_energy, h00, h01)

# 计算散射矩阵
transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active = guan.calculate_scattering_matrix(fermi_energy, h00, h01, length=100)

# 从散射矩阵中，获取散射矩阵的信息
number_of_active_channels, number_of_evanescent_channels, k_of_right_moving_active_channels, k_of_left_moving_active_channels, velocity_of_right_moving_active_channels, velocity_of_left_moving_active_channels, transmission_matrix_for_active_channels, reflection_matrix_for_active_channels, total_transmission_of_channels, total_conductance, total_reflection_of_channels, sum_of_transmission_and_reflection_of_channels = guan.information_of_scattering_matrix(transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active)

# 已知h00和h01，计算散射矩阵并获得散射矩阵的信息
number_of_active_channels, number_of_evanescent_channels, k_of_right_moving_active_channels, k_of_left_moving_active_channels, velocity_of_right_moving_active_channels, velocity_of_left_moving_active_channels, transmission_matrix_for_active_channels, reflection_matrix_for_active_channels, total_transmission_of_channels, total_conductance, total_reflection_of_channels, sum_of_transmission_and_reflection_of_channels = guan.calculate_scattering_matrix_and_get_information(fermi_energy, h00, h01, length=100)

# 从散射矩阵中，打印出散射矩阵的信息
guan.print_or_write_scattering_matrix_with_information_of_scattering_matrix(number_of_active_channels, number_of_evanescent_channels, k_of_right_moving_active_channels, k_of_left_moving_active_channels, velocity_of_right_moving_active_channels, velocity_of_left_moving_active_channels, transmission_matrix_for_active_channels, reflection_matrix_for_active_channels, total_transmission_of_channels, total_conductance, total_reflection_of_channels, sum_of_transmission_and_reflection_of_channels, print_show=1, write_file=0, filename='a', file_format='.txt')

# 已知h00和h01，计算散射矩阵并打印出散射矩阵的信息
guan.print_or_write_scattering_matrix(fermi_energy, h00, h01, length=100, print_show=1, write_file=0, filename='a', file_format='.txt')

# 在无序下，计算散射矩阵
transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active = guan.calculate_scattering_matrix_with_disorder(fermi_energy, h00, h01, length=100, disorder_intensity=2.0, disorder_concentration=1.0)

# 在无序下，计算散射矩阵，并获取散射矩阵多次计算的平均信息
transmission_matrix_for_active_channels_averaged, reflection_matrix_for_active_channels_averaged = guan.calculate_scattering_matrix_with_disorder_and_get_averaged_information(fermi_energy, h00, h01, length=100, disorder_intensity=2.0, disorder_concentration=1.0, calculation_times=1)





























# Module 9: topological invariant

# 通过高效法计算方格子的陈数
chern_number = guan.calculate_chern_number_for_square_lattice_with_efficient_method(hamiltonian_function, precision=100, print_show=0)

# 通过高效法计算方格子的陈数（可计算简并的情况）
chern_number = guan.calculate_chern_number_for_square_lattice_with_efficient_method_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], precision=100, print_show=0)

# 通过Wilson loop方法计算方格子的陈数
chern_number = guan.calculate_chern_number_for_square_lattice_with_wilson_loop(hamiltonian_function, precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0)

# 通过Wilson loop方法计算方格子的陈数（可计算简并的情况）
chern_number = guan.calculate_chern_number_for_square_lattice_with_wilson_loop_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0)

# 通过高效法计算贝利曲率
k_array, berry_curvature_array = guan.calculate_berry_curvature_with_efficient_method(hamiltonian_function, k_min='default', k_max='default', precision=100, print_show=0)

# 通过高效法计算贝利曲率（可计算简并的情况）
k_array, berry_curvature_array = guan.calculate_berry_curvature_with_efficient_method_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], k_min='default', k_max='default', precision=100, print_show=0)

# 通过Wilson loop方法计算贝里曲率
k_array, berry_curvature_array = guan.calculate_berry_curvature_with_wilson_loop(hamiltonian_function, k_min='default', k_max='default', precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0)

# 通过Wilson loop方法计算贝里曲率（可计算简并的情况）
k_array, berry_curvature_array = guan.calculate_berry_curvature_with_wilson_loop_for_degenerate_case(hamiltonian_function, index_of_bands=[0, 1], k_min='default', k_max='default', precision_of_plaquettes=20, precision_of_wilson_loop=5, print_show=0)

# 计算蜂窝格子的陈数（高效法）
chern_number = guan.calculate_chern_number_for_honeycomb_lattice(hamiltonian_function, a=1, precision=300, print_show=0)

# 计算Wilson loop
wilson_loop_array = guan.calculate_wilson_loop(hamiltonian_function, k_min='default', k_max='default', precision=100, print_show=0)































# Module 10: plot figures

# 导入plt, fig, ax
plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=0.2, adjust_left=0.2, labelsize=20)

# 基于plt, fig, ax开始画图
guan.plot_without_starting_fig(plt, fig, ax, x_array, y_array, xlabel='x', ylabel='y', title='', fontsize=20, style='', y_min=None, y_max=None, linewidth=None, markersize=None, color=None)

# 画图
guan.plot(x_array, y_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style='', y_min=None, y_max=None, linewidth=None, markersize=None, adjust_bottom=0.2, adjust_left=0.2)

# 一组横坐标数据，两组纵坐标数据画图
guan.plot_two_array(x_array, y1_array, y2_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, markersize_1=None, markersize_2=None, adjust_bottom=0.2, adjust_left=0.2)

# 两组横坐标数据，两组纵坐标数据画图
guan.plot_two_array_with_two_horizontal_array(x1_array, x2_array, y1_array, y2_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, markersize_1=None, markersize_2=None, adjust_bottom=0.2, adjust_left=0.2)

# 一组横坐标数据，三组纵坐标数据画图
guan.plot_three_array(x_array, y1_array, y2_array, y3_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', style_3='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, linewidth_3=None,markersize_1=None, markersize_2=None, markersize_3=None, adjust_bottom=0.2, adjust_left=0.2)

# 三组横坐标数据，三组纵坐标数据画图
guan.plot_three_array_with_three_horizontal_array(x1_array, x2_array, x3_array, y1_array, y2_array, y3_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', style_3='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, linewidth_3=None, markersize_1=None, markersize_2=None, markersize_3=None, adjust_bottom=0.2, adjust_left=0.2)

# 画三维图
guan.plot_3d_surface(x_array, y_array, matrix, xlabel='x', ylabel='y', zlabel='z', title='', fontsize=20, labelsize=15, show=1, save=0, filename='a', file_format='.jpg', dpi=300, z_min=None, z_max=None, rcount=100, ccount=100)

# 画Contour图
guan.plot_contour(x_array, y_array, matrix, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=15, cmap='jet', levels=None, show=1, save=0, filename='a', file_format='.jpg', dpi=300)

# 画棋盘图/伪彩色图
guan.plot_pcolor(x_array, y_array, matrix, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=15, cmap='jet', levels=None, show=1, save=0, filename='a', file_format='.jpg', dpi=300)

# 通过坐标画点和线
guan.draw_dots_and_lines(coordinate_array, draw_dots=1, draw_lines=1, max_distance=1.1, line_style='-k', linewidth=1, dot_style='ro', markersize=3, show=1, save=0, filename='a', file_format='.eps', dpi=300)

# 合并两个图片
guan.combine_two_images(image_path_array, figsize=(16,8), show=0, save=1, filename='a', file_format='.jpg', dpi=300)

# 合并三个图片
guan.combine_three_images(image_path_array, figsize=(16,5), show=0, save=1, filename='a', file_format='.jpg', dpi=300)

# 合并四个图片
guan.combine_four_images(image_path_array, figsize=(16,16), show=0, save=1, filename='a', file_format='.jpg', dpi=300)

# 对于某个目录中的txt文件，批量读取和画图
guan.batch_reading_and_plotting(directory, xlabel='x', ylabel='y')

# 制作GIF动画
guan.make_gif(image_path_array, filename='a', duration=0.1)

# 选取颜色
color_array = guan.color_matplotlib()






























# Module 11: read and write

# 将数据存到文件
guan.dump_data(data, filename, file_format='.txt')

# 从文件中恢复数据到变量
data = guan.load_data(filename, file_format='.txt')

# 读取文件中的一维数据（每一行一组x和y）
x_array, y_array = guan.read_one_dimensional_data(filename='a', file_format='.txt')

# 读取文件中的一维数据（每一行一组x和y）（支持复数形式）
x_array, y_array = guan.read_one_dimensional_complex_data(filename='a', file_format='txt')

# 读取文件中的二维数据（第一行和列分别为横纵坐标）
x_array, y_array, matrix = guan.read_two_dimensional_data(filename='a', file_format='.txt')

# 读取文件中的二维数据（第一行和列分别为横纵坐标）（支持复数形式）
x_array, y_array, matrix = guan.read_two_dimensional_complex_data(filename='a', file_format='.txt')

# 读取文件中的二维数据（不包括x和y）
matrix = guan.read_two_dimensional_data_without_xy_array(filename='a', file_format='.txt')

# 打开文件用于新增内容
f = guan.open_file(filename='a', file_format='.txt')

# 在文件中写入一维数据（每一行一组x和y）
guan.write_one_dimensional_data(x_array, y_array, filename='a', file_format='.txt')

# 在文件中写入一维数据（每一行一组x和y）（需要输入文件）
guan.write_one_dimensional_data_without_opening_file(x_array, y_array, f)

# 在文件中写入二维数据（第一行和列分别为横纵坐标）
guan.write_two_dimensional_data(x_array, y_array, matrix, filename='a', file_format='.txt')

# 在文件中写入二维数据（第一行和列分别为横纵坐标）（需要输入文件）
guan.write_two_dimensional_data_without_opening_file(x_array, y_array, matrix, f)

# 在文件中写入二维数据（不包括x和y）
guan.write_two_dimensional_data_without_xy_array(matrix, filename='a', file_format='.txt')

# 在文件中写入二维数据（不包括x和y）（需要输入文件）
guan.write_two_dimensional_data_without_xy_array_and_without_opening_file(matrix, f)

# 以显示编号的样式，打印数组
guan.print_array_with_index(array, show_index=1, index_type=0)




































# Module 12: data processing

# 并行计算前的预处理，把参数分成多份
parameter_array = guan.preprocess_for_parallel_calculations(parameter_array_all, cpus=1, task_index=0)

# 在一组数据中找到数值相近的数
new_array = guan.find_close_values_in_one_array(array, precision=1e-2)

# 寻找能带的简并点
degenerate_k_array, degenerate_eigenvalue_array = guan.find_degenerate_points(k_array, eigenvalue_array, precision=1e-2)

# 选取一个种子生成固定的随机整数
rand_num = guan.generate_random_int_number_for_a_specific_seed(seed=0, x_min=0, x_max=10)

# 统计运行的日期和时间，写进文件
guan.statistics_with_day_and_time(content='', filename='a', file_format='.txt')

# 统计Python文件中import的数量并排序
import_statement_counter = guan.count_number_of_import_statements(filename, file_format='.py', num=1000)

# 将RGB转成HEX
hex = guan.rgb_to_hex(rgb, pound=1)

# 将HEX转成RGB
rgb = guan.hex_to_rgb(hex)

# 使用MD5进行散列加密
hashed_password = guan.encryption_MD5(password, salt='')

# 使用SHA-256进行散列加密
hashed_password = guan.encryption_SHA_256(password, salt='')

# 获取当前日期字符串
datetime_date = guan.get_date(bar=True)

# 获取当前时间字符串
datetime_time = guan.get_time()

# 获取本月的所有日期
day_array = guan.get_days_of_the_current_month(str_or_datetime='str')

# 获取上个月份
year_of_last_month, last_month = guan.get_last_month()

# 获取上上个月份
year_of_the_month_before_last, the_month_before_last = guan.get_the_month_before_last()

# 获取上个月的所有日期
day_array = guan.get_days_of_the_last_month(str_or_datetime='str')

# 获取上上个月的所有日期
day_array = guan.get_days_of_the_month_before_last(str_or_datetime='str')

# 获取所有股票
title, stock_data = guan.all_stocks()

# 获取所有股票的代码
stock_symbols = guan.all_stock_symbols()

# 从股票代码获取股票名称
stock_name = guan.find_stock_name_from_symbol(symbol='000002')

# 获取单个股票的历史数据
title, stock_data = guan.history_data_of_one_stock(symbol='000002', period='daily', start_date="19000101", end_date='21000101')

# 播放学术单词
guan.play_academic_words(reverse=0, random_on=0, bre_or_ame='ame', show_translation=1, show_link=1, translation_time=2, rest_time=1)

# 播放挑选过后的学术单词
guan.play_selected_academic_words(reverse=0, random_on=0, bre_or_ame='ame', show_link=1, rest_time=3)

# 播放元素周期表上的单词
guan.play_element_words(random_on=0, show_translation=1, show_link=1, translation_time=2, rest_time=1)


























# Module 13: file processing

# 如果不存在文件夹，则新建文件夹
guan.make_directory(directory='./test')

# 复制一份文件
guan.copy_file(file1='./a.txt', file2='./b.txt')

# 拼接两个PDF文件
guan.combine_two_pdf_files(input_file_1='a.pdf', input_file_2='b.pdf', output_file='combined_file.pdf')

# 将PDF文件转成文本
content = guan.pdf_to_text(pdf_path)

# 获取PDF文献中的链接。例如: link_starting_form='https://doi.org'
links = guan.get_links_from_pdf(pdf_path, link_starting_form='')

# 通过Sci-Hub网站下载文献
guan.download_with_scihub(address=None, num=1)

# 将文件目录结构写入Markdown文件
guan.write_file_list_in_markdown(directory='./', filename='a', reverse_positive_or_negative=1, starting_from_h1=None, banned_file_format=[], hide_file_format=None, divided_line=None, show_second_number=None, show_third_number=None)

# 查找文件名相同的文件
repeated_file = guan.find_repeated_file_with_same_filename(directory='./', ignored_directory_with_words=[], ignored_file_with_words=[], num=1000)

# 统计各个子文件夹中的文件数量
guan.count_file_in_sub_directory(directory='./', smaller_than_num=None)

# 产生必要的文件，例如readme.md
guan.creat_necessary_file(directory, filename='readme', file_format='.md', content='', overwrite=None, ignored_directory_with_words=[])

# 删除特定文件名的文件
guan.delete_file_with_specific_name(directory, filename='readme', file_format='.md')

# 所有文件移到根目录（慎用）
guan.move_all_files_to_root_directory(directory)

# 改变当前的目录位置
guan.change_directory_by_replacement(current_key_word='code', new_key_word='data')

# 生成二维码
guan.creat_qrcode(data="https://www.guanjihuan.com", filename='a', file_format='.png')

# 将文本转成音频
guan.str_to_audio(str='hello world', filename='str', rate=125, voice=1, read=1, save=0, compress=0, bitrate='16k', print_text=0)

# 将txt文件转成音频
guan.txt_to_audio(txt_path, rate=125, voice=1, read=1, save=0, compress=0, bitrate='16k', print_text=0)

# 将PDF文件转成音频
guan.pdf_to_audio(pdf_path, rate=125, voice=1, read=1, save=0, compress=0, bitrate='16k', print_text=0)

# 将wav音频文件压缩成MP3音频文件
guan.compress_wav_to_mp3(wav_path, output_filename='a.mp3', bitrate='16k')
