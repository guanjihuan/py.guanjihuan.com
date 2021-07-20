import guan

# test
guan.test()

# basic functions
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

# calculate reciprocal lattice vectors
b1 = guan.calculate_one_dimensional_reciprocal_lattice_vector(a1)
b1, b2 = guan.calculate_two_dimensional_reciprocal_lattice_vectors(a1, a2)
b1, b2, b3 = guan.calculate_three_dimensional_reciprocal_lattice_vectors(a1, a2, a3)
b1 = guan.calculate_one_dimensional_reciprocal_lattice_vector_with_sympy(a1)
b1, b2 = guan.calculate_two_dimensional_reciprocal_lattice_vectors_with_sympy(a1, a2)
b1, b2, b3 = guan.calculate_three_dimensional_reciprocal_lattice_vectors_with_sympy(a1, a2, a3)

# Fourier transform
hamiltonian = guan.one_dimensional_fourier_transform(k, unit_cell, hopping)
hamiltonian = guan.two_dimensional_fourier_transform_for_square_lattice(k1, k2, unit_cell, hopping_1, hopping_2)
hamiltonian = guan.three_dimensional_fourier_transform_for_cubic_lattice(k1, k2, k3, unit_cell, hopping_1, hopping_2, hopping_3)

# Hamiltonian of finite size
hamiltonian = guan.finite_size_along_one_direction(N, on_site=0, hopping=1, period=0)
hamiltonian = guan.finite_size_along_two_directions_for_square_lattice(N1, N2, on_site=0, hopping_1=1, hopping_2=1, period_1=0, period_2=0)
hamiltonian = guan.finite_size_along_three_directions_for_cubic_lattice(N1, N2, N3, on_site=0, hopping_1=1, hopping_2=1, hopping_3=1, period_1=0, period_2=0, period_3=0)
hopping = guan.hopping_along_zigzag_direction_for_graphene(N)
hamiltonian = guan.finite_size_along_two_directions_for_graphene(N1, N2, period_1=0, period_2=0)

# Hamiltonian of models in the reciprocal space
hamiltonian = guan.hamiltonian_of_simple_chain(k)
hamiltonian = guan.hamiltonian_of_square_lattice(k1, k2)
hamiltonian = guan.hamiltonian_of_square_lattice_in_quasi_one_dimension(k, N=10)
hamiltonian = guan.hamiltonian_of_cubic_lattice(k1, k2, k3)
hamiltonian = guan.hamiltonian_of_ssh_model(k, v=0.6, w=1)
hamiltonian = guan.hamiltonian_of_graphene(k1, k2, M=0, t=1, a=1/sqrt(3))
hamiltonian = guan.hamiltonian_of_graphene_with_zigzag_in_quasi_one_dimension(k, N=10, M=0, t=1)
hamiltonian = guan.hamiltonian_of_haldane_model(k1, k2, M=2/3, t1=1, t2=1/3, phi=pi/4, a=1/sqrt(3))
hamiltonian = guan.hamiltonian_of_haldane_model_in_quasi_one_dimension(k, N=10, M=2/3, t1=1, t2=1/3, phi=pi/4)

# calculate band structures
eigenvalue = guan.calculate_eigenvalue(hamiltonian)
eigenvalue_array = guan.calculate_eigenvalue_with_one_parameter(x, hamiltonian_function):
eigenvalue_array = guan.calculate_eigenvalue_with_two_parameters(x, y, hamiltonian_function)

# calculate wave functions
eigenvector = guan.calculate_eigenvector(hamiltonian)

# find vector with the same gauge
vector_target = guan.find_vector_with_the_same_gauge_with_binary_search(vector_target, vector_ref, show_error=1, show_times=0, show_phase=0, n_test=10001, precision=1e-6)
vector = guan.find_vector_with_fixed_gauge_by_making_one_component_real(vector, precision=0.005)

# calculate Green functions
green = guan.green_function(fermi_energy, hamiltonian, broadening, self_energy=0)
green_nn_n = guan.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening, self_energy=0)
green_in_n = guan.green_function_in_n(green_in_n_minus, h01, green_nn_n)
green_ni_n = guan.green_function_ni_n(green_nn_n, h01, green_ni_n_minus)
green_ii_n = guan.green_function_ii_n(green_ii_n_minus, green_in_n_minus, h01, green_nn_n, green_ni_n_minus)

# calculate density of states
total_dos = guan.total_density_of_states(fermi_energy, hamiltonian, broadening=0.01)
total_dos_array = guan.total_density_of_states_with_fermi_energy_array(fermi_energy_array, hamiltonian, broadening=0.01)
local_dos = guan.local_density_of_states_for_square_lattice(fermi_energy, hamiltonian, N1, N2, internal_degree=1, broadening=0.01)
local_dos = guan.local_density_of_states_for_cubic_lattice(fermi_energy, hamiltonian, N1, N2, N3, internal_degree=1, broadening=0.01)
local_dos = guan.local_density_of_states_for_square_lattice_using_dyson_equation(fermi_energy, h00, h01, N2, N1, internal_degree=1, broadening=0.01)
local_dos = guan.local_density_of_states_for_cubic_lattice_using_dyson_equation(fermi_energy, h00, h01, N3, N2, N1, internal_degree=1, broadening=0.01)

# calculate conductance
transfer = guan.transfer_matrix(fermi_energy, h00, h01)
right_lead_surface, left_lead_surface = guan.surface_green_function_of_lead(fermi_energy, h00, h01)
right_self_energy, left_self_energy = guan.self_energy_of_lead(fermi_energy, h00, h01)
conductance = guan.calculate_conductance(fermi_energy, h00, h01, length=100)
conductance_array = guan.calculate_conductance_with_fermi_energy_array(fermi_energy_array, h00, h01, length=100)

# scattering matrix
if_active = guan.if_active_channel(k_of_channel)
k_of_channel, velocity_of_channel, eigenvalue, eigenvector = guan.get_k_and_velocity_of_channel(fermi_energy, h00, h01)
k_right, k_left, velocity_right, velocity_left, f_right, f_left, u_right, u_left, ind_right_active = guan.get_classified_k_velocity_u_and_f(fermi_energy, h00, h01)
transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active = guan.calculate_scattering_matrix(fermi_energy, h00, h01, length=100)
guan.print_or_write_scattering_matrix(fermi_energy, h00, h01, length=100, on_print=1, on_write=0)

# calculate Chern number
chern_number = guan.calculate_chern_number_for_square_lattice(hamiltonian_function, precision=100)

# calculate Wilson loop
wilson_loop_array = guan.calculate_wilson_loop(hamiltonian_function, k_min=-pi, k_max=pi, precision=100)

# read and write
x, y = guan.read_one_dimensional_data(filename='a')
x, y, matrix = guan.read_two_dimensional_data(filename='a')
guan.write_one_dimensional_data(x, y, filename='a')
guan.write_two_dimensional_data(x, y, matrix, filename='a')

# plot figures
guan.plot(x, y, xlabel='x', ylabel='y', title='', filename='a', show=1, save=0, type='', y_min=None, y_max=None)
guan.plot_3d_surface(x, y, matrix, xlabel='x', ylabel='y', zlabel='z', title='', filename='a', show=1, save=0, z_min=None, z_max=None)
guan.plot_contour(x, y, matrix, xlabel='x', ylabel='y', title='', filename='a', show=1, save=0)

# download
guan.download_with_scihub(address=None, num=1)

# audio
guan.txt_to_audio(txt_path, rate=125, voice=1, read=1, save=0, print_text=0)
content = guan.pdf_to_text(pdf_path)
guan.pdf_to_audio(pdf_path, rate=125, voice=1, read=1, save=0, print_text=0)