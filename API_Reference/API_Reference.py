import guan
import math

# basic functions

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



# Fourier transform

hamiltonian = guan.one_dimensional_fourier_transform(k, unit_cell, hopping)

hamiltonian = guan.two_dimensional_fourier_transform_for_square_lattice(k1, k2, unit_cell, hopping_1, hopping_2)

hamiltonian = guan.three_dimensional_fourier_transform_for_cubic_lattice(k1, k2, k3, unit_cell, hopping_1, hopping_2, hopping_3)

hamiltonian_function = guan.one_dimensional_fourier_transform_with_k(unit_cell, hopping)

hamiltonian_function = guan.two_dimensional_fourier_transform_for_square_lattice_with_k1_k2(unit_cell, hopping_1, hopping_2)

hamiltonian_function = guan.three_dimensional_fourier_transform_for_cubic_lattice_with_k1_k2_k3(unit_cell, hopping_1, hopping_2, hopping_3)

b1 = guan.calculate_one_dimensional_reciprocal_lattice_vector(a1)

b1, b2 = guan.calculate_two_dimensional_reciprocal_lattice_vectors(a1, a2)

b1, b2, b3 = guan.calculate_three_dimensional_reciprocal_lattice_vectors(a1, a2, a3)

b1 = guan.calculate_one_dimensional_reciprocal_lattice_vector_with_sympy(a1)

b1, b2 = guan.calculate_two_dimensional_reciprocal_lattice_vectors_with_sympy(a1, a2)

b1, b2, b3 = guan.calculate_three_dimensional_reciprocal_lattice_vectors_with_sympy(a1, a2, a3)



# Hamiltonian of finite size systems

hamiltonian = guan.hamiltonian_of_finite_size_system_along_one_direction(N, on_site=0, hopping=1, period=0)

hamiltonian = guan.hamiltonian_of_finite_size_system_along_two_directions_for_square_lattice(N1, N2, on_site=0, hopping_1=1, hopping_2=1, period_1=0, period_2=0)

hamiltonian = guan.hamiltonian_of_finite_size_system_along_three_directions_for_cubic_lattice(N1, N2, N3, on_site=0, hopping_1=1, hopping_2=1, hopping_3=1, period_1=0, period_2=0, period_3=0)

hopping = guan.hopping_matrix_along_zigzag_direction_for_graphene_ribbon(N)

hamiltonian = guan.hamiltonian_of_finite_size_system_along_two_directions_for_graphene(N1, N2, period_1=0, period_2=0)



# Hamiltonian of models in the reciprocal space

hamiltonian = guan.hamiltonian_of_simple_chain(k)

hamiltonian = guan.hamiltonian_of_square_lattice(k1, k2)

hamiltonian = guan.hamiltonian_of_square_lattice_in_quasi_one_dimension(k, N=10)

hamiltonian = guan.hamiltonian_of_cubic_lattice(k1, k2, k3)

hamiltonian = guan.hamiltonian_of_ssh_model(k, v=0.6, w=1)

hamiltonian = guan.hamiltonian_of_graphene(k1, k2, M=0, t=1, a=1/math.sqrt(3))

hamiltonian = guan.hamiltonian_of_graphene_with_zigzag_in_quasi_one_dimension(k, N=10, M=0, t=1)

hamiltonian = guan.hamiltonian_of_haldane_model(k1, k2, M=2/3, t1=1, t2=1/3, phi=math.pi/4, a=1/math.sqrt(3))

hamiltonian = guan.hamiltonian_of_haldane_model_in_quasi_one_dimension(k, N=10, M=2/3, t1=1, t2=1/3, phi=math.pi/4)

hamiltonian = guan.hamiltonian_of_one_QAH_model(k1, k2, t1=1, t2=1, t3=0.5, m=-1)

hamiltonian = guan.hamiltonian_of_BBH_model(kx, ky, gamma_x=0.5, gamma_y=0.5, lambda_x=1, lambda_y=1)



# band structures and wave functions

eigenvalue = guan.calculate_eigenvalue(hamiltonian)

eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(x_array, hamiltonian_function, print_show=0)

eigenvalue_array = guan.calculate_eigenvalue_with_two_parameters(x_array, y_array, hamiltonian_function, print_show=0, print_show_more=0)

eigenvector = guan.calculate_eigenvector(hamiltonian)

vector_target = guan.find_vector_with_the_same_gauge_with_binary_search(vector_target, vector_ref, show_error=1, show_times=0, show_phase=0, n_test=1000, precision=1e-6)

vector = guan.find_vector_with_fixed_gauge_by_making_one_component_real(vector, precision=0.005, index=None)

vector_array = guan.find_vector_array_with_fixed_gauge_by_making_one_component_real(vector_array, precision=0.005)

vector1, vector2 = guan.rotation_of_degenerate_vectors(vector1, vector2, index1, index2, precision=0.01, criterion=0.01, show_theta=0)

vector1_array, vector2_array = guan.rotation_of_degenerate_vectors_array(vector1_array, vector2_array, precision=0.01, criterion=0.01, show_theta=0)



# Green functions

green = guan.green_function(fermi_energy, hamiltonian, broadening, self_energy=0)

green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening, self_energy=0)

green_in_n = guan.green_function_in_n(green_in_n_minus, h01, green_nn_n)

green_ni_n = guan.green_function_ni_n(green_nn_n, h01, green_ni_n_minus)

green_ii_n = guan.green_function_ii_n(green_ii_n_minus, green_in_n_minus, h01, green_nn_n, green_ni_n_minus)

transfer = guan.transfer_matrix(fermi_energy, h00, h01)

right_lead_surface, left_lead_surface = guan.surface_green_function_of_lead(fermi_energy, h00, h01)

right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead(fermi_energy, h00, h01)

right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead_with_h_LC_and_h_CR(fermi_energy, h00, h01, h_LC, h_CR)

self_energy, gamma = guan.self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00, h01, h_lead_to_center)

green, gamma_right, gamma_left = guan.green_function_with_leads(fermi_energy, h00, h01, h_LC, h_CR, center_hamiltonian)

G_n = guan.electron_correlation_function_green_n_for_local_current(fermi_energy, h00, h01, h_LC, h_CR, center_hamiltonian)



# density of states

total_dos = guan.total_density_of_states(fermi_energy, hamiltonian, broadening=0.01)

total_dos_array = guan.total_density_of_states_with_fermi_energy_array(fermi_energy_array, hamiltonian, broadening=0.01, print_show=0)

local_dos = guan.local_density_of_states_for_square_lattice(fermi_energy, hamiltonian, N1, N2, internal_degree=1, broadening=0.01)

local_dos = guan.local_density_of_states_for_cubic_lattice(fermi_energy, hamiltonian, N1, N2, N3, internal_degree=1, broadening=0.01)

local_dos = guan.local_density_of_states_for_square_lattice_using_dyson_equation(fermi_energy, h00, h01, N2, N1, internal_degree=1, broadening=0.01)

local_dos = guan.local_density_of_states_for_cubic_lattice_using_dyson_equation(fermi_energy, h00, h01, N3, N2, N1, internal_degree=1, broadening=0.01)

local_dos = guan.local_density_of_states_for_square_lattice_with_self_energy_using_dyson_equation(fermi_energy, h00, h01, N2, N1, right_self_energy, left_self_energy, internal_degree=1, broadening=0.01)



# quantum transport

conductance = guan.calculate_conductance(fermi_energy, h00, h01, length=100)

conductance_array = guan.calculate_conductance_with_fermi_energy_array(fermi_energy_array, h00, h01, length=100, print_show=0)

conductance = guan.calculate_conductance_with_disorder(fermi_energy, h00, h01, disorder_intensity=2.0, disorder_concentration=1.0, length=100)

conductance_array = guan.calculate_conductance_with_disorder_intensity_array(fermi_energy, h00, h01, disorder_intensity_array, disorder_concentration=1.0, length=100, calculation_times=1, print_show=0)

conductance_array = guan.calculate_conductance_with_disorder_concentration_array(fermi_energy, h00, h01, disorder_concentration_array, disorder_intensity=2.0, length=100, calculation_times=1, print_show=0)

conductance_array = guan.calculate_conductance_with_scattering_length_array(fermi_energy, h00, h01, length_array, disorder_intensity=2.0, disorder_concentration=1.0, calculation_times=1, print_show=0)

gamma_array, green = guan.get_gamma_array_and_green_for_six_terminal_transmission(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10)

transmission_matrix = guan.calculate_six_terminal_transmission_matrix(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10)

transmission_12, transmission_13, transmission_14, transmission_15, transmission_16 = guan.calculate_six_terminal_transmissions_from_lead_1(fermi_energy, h00_for_lead_4, h01_for_lead_4, h00_for_lead_2, h01_for_lead_2, center_hamiltonian, width=10, length=50, internal_degree=1, moving_step_of_leads=10)

if_active = guan.if_active_channel(k_of_channel)

k_of_channel, velocity_of_channel, eigenvalue, eigenvector = guan.get_k_and_velocity_of_channel(fermi_energy, h00, h01)

k_right, k_left, velocity_right, velocity_left, f_right, f_left, u_right, u_left, ind_right_active = guan.get_classified_k_velocity_u_and_f(fermi_energy, h00, h01)

transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active = guan.calculate_scattering_matrix(fermi_energy, h00, h01, length=100)

guan.print_or_write_scattering_matrix(fermi_energy, h00, h01, length=100, print_show=1, write_file=0, filename='a', format='txt')



# topological invariant

chern_number = guan.calculate_chern_number_for_square_lattice(hamiltonian_function, precision=100, print_show=0)

chern_number = guan.calculate_chern_number_for_square_lattice_with_Wilson_loop(hamiltonian_function, precision_of_plaquettes=10, precision_of_Wilson_loop=100, print_show=0)

chern_number = guan.calculate_chern_number_for_honeycomb_lattice(hamiltonian_function, a=1, precision=300, print_show=0)

wilson_loop_array = guan.calculate_wilson_loop(hamiltonian_function, k_min=-math.pi, k_max=math.pi, precision=100, print_show=0)



# read and write

x_array, y_array = guan.read_one_dimensional_data(filename='a', format='txt')

x_array, y_array, matrix = guan.read_two_dimensional_data(filename='a', format='txt')

guan.write_one_dimensional_data(x_array, y_array, filename='a', format='txt')

guan.write_two_dimensional_data(x_array, y_array, matrix, filename='a', format='txt')




# plot figures

guan.plot(x_array, y_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', format='jpg', dpi=300, type='', y_min=None, y_max=None, linewidth=None, markersize=None, adjust_bottom=0.2, adjust_left=0.2)

guan.plot_3d_surface(x_array, y_array, matrix, xlabel='x', ylabel='y', zlabel='z', title='', fontsize=20, labelsize=15, show=1, save=0, filename='a', format='jpg', dpi=300, z_min=None, z_max=None, rcount=100, ccount=100)

guan.plot_contour(x_array, y_array, matrix, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=15, show=1, save=0, filename='a', format='jpg', dpi=300)



# data processing

parameter_array = guan.preprocess_for_parallel_calculations(parameter_array_all, cpus=1, task_index=0)

guan.change_directory_by_replacement(current_key_word='codes', new_key_word='data')

guan.batch_reading_and_plotting(directory, xlabel='x', ylabel='y')



# others

guan.download_with_scihub(address=None, num=1)

guan.str_to_audio(str='hello world', rate=125, voice=1, read=1, save=0, print_text=0)

guan.txt_to_audio(txt_path, rate=125, voice=1, read=1, save=0, print_text=0)

content = guan.pdf_to_text(pdf_path)

guan.pdf_to_audio(pdf_path, rate=125, voice=1, read=1, save=0, print_text=0)

guan.play_academic_words(bre_or_ame='ame', random_on=0, show_translation=1, show_link=1, translation_time=2, rest_time=1)

guan.play_element_words(random_on=0, show_translation=1, show_link=1, translation_time=2, rest_time=1)