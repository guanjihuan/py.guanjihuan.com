import gjh

# test
gjh.test()


# basic functions
sigma_0 = gjh.sigma_0()
sigma_x = gjh.sigma_x()
sigma_y = gjh.sigma_y()
sigma_z = gjh.sigma_z()

sigma_00 = gjh.sigma_00()
sigma_0x = gjh.sigma_0x()
sigma_0y = gjh.sigma_0y()
sigma_0z = gjh.sigma_0z()

sigma_x0 = gjh.sigma_x0()
sigma_xx = gjh.sigma_xx()
sigma_xy = gjh.sigma_xy()
sigma_xz = gjh.sigma_xz()

sigma_y0 = gjh.sigma_y0()
sigma_yx = gjh.sigma_yx()
sigma_yy = gjh.sigma_yy()
sigma_yz = gjh.sigma_yz()

sigma_z0 = gjh.sigma_z0()
sigma_zx = gjh.sigma_zx()
sigma_zy = gjh.sigma_zy()
sigma_zz = gjh.sigma_zz()


# Hermitian Hamiltonian of tight binding model 
hamiltonian = gjh.finite_size_along_one_direction(N, on_site=0, hopping=1, period=0)
hamiltonian = gjh.finite_size_along_two_directions_for_square_lattice(N1, N2, on_site=0, hopping_1=1, hopping_2=1, period_1=0, period_2=0)
hamiltonian = gjh.finite_size_along_three_directions_for_cubic_lattice(N1, N2, N3, on_site=0, hopping_1=1, hopping_2=1, hopping_3=1, period_1=0, period_2=0, period_3=0)
hamiltonian = gjh.one_dimensional_fourier_transform(k, unit_cell, hopping)
hamiltonian = gjh.two_dimensional_fourier_transform_for_square_lattice(k1, k2, unit_cell, hopping_1, hopping_2)
hamiltonian = gjh.three_dimensional_fourier_transform_for_cubic_lattice(k1, k2, k3, unit_cell, hopping_1, hopping_2, hopping_3)


# Hamiltonian of graphene lattice
hopping = gjh.hopping_along_zigzag_direction_for_graphene(N)
hamiltonian = gjh.finite_size_along_two_directions_for_graphene(N1, N2, period_1=0, period_2=0)


# calculate band structures
eigenvalue = gjh.calculate_eigenvalue(hamiltonian)
eigenvalue_array = gjh.calculate_eigenvalue_with_one_parameter(x, hamiltonian_function):
eigenvalue_array = gjh.calculate_eigenvalue_with_two_parameters(x, y, hamiltonian_function)


# calculate wave functions
eigenvector = gjh.calculate_eigenvector(hamiltonian)


# calculate Green functions
green = gjh.green_function(fermi_energy, hamiltonian, broadening, self_energy=0)
green_nn_n = gjh.green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening, self_energy=0)
green_in_n = gjh.green_function_in_n(green_in_n_minus, h01, green_nn_n)
green_ni_n = gjh.green_function_ni_n(green_nn_n, h01, green_ni_n_minus)
green_ii_n = gjh.green_function_ii_n(green_ii_n_minus, green_in_n_minus, h01, green_nn_n, green_ni_n_minus)


# calculate density of states
total_dos = gjh.total_density_of_states(fermi_energy, hamiltonian, broadening=0.01)
total_dos_array = gjh.total_density_of_states_with_fermi_energy_array(fermi_energy_array, hamiltonian, broadening=0.01)
local_dos = gjh.local_density_of_states_for_square_lattice(fermi_energy, hamiltonian, N1, N2, internal_degree=1, broadening=0.01)
local_dos = gjh.local_density_of_states_for_cubic_lattice(fermi_energy, hamiltonian, N1, N2, N3, internal_degree=1, broadening=0.01)
local_dos = gjh.local_density_of_states_for_square_lattice_using_dyson_equation(fermi_energy, h00, h01, N2, N1, internal_degree=1, broadening=0.01)
local_dos = gjh.local_density_of_states_for_cubic_lattice_using_dyson_equation(fermi_energy, h00, h01, N3, N2, N1, internal_degree=1, broadening=0.01)


# calculate conductance
transfer = gjh.transfer_matrix(fermi_energy, h00, h01)
right_lead_surface, left_lead_surface = gjh.surface_green_function_of_lead(fermi_energy, h00, h01
right_self_energy, left_self_energy = gjh.self_energy_of_lead(fermi_energy, h00, h01)
conductance = gjh.calculate_conductance(fermi_energy, h00, h01, length=100)
conductance_array = gjh.calculate_conductance_with_fermi_energy_array(fermi_energy_array, h00, h01, length=100)


# scattering matrix
if_active = gjh.if_active_channel(k_of_channel)
k_of_channel, velocity_of_channel, eigenvalue, eigenvector = gjh.get_k_and_velocity_of_channel(fermi_energy, h00, h01)
k_right, k_left, velocity_right, velocity_left, f_right, f_left, u_right, u_left, ind_right_active = gjh.get_classified_k_velocity_u_and_f(fermi_energy, h00, h01)
transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active = gjh.calculate_scattering_matrix(fermi_energy, h00, h01, length=100)
gjh.print_or_write_scattering_matrix(fermi_energy, h00, h01, length=100, on_print=1, on_write=0)


# calculate Chern number
chern_number = gjh.calculate_chern_number_for_square_lattice(hamiltonian_function, precision=100)


# calculate Wilson loop
wilson_loop_array = gjh.calculate_wilson_loop(hamiltonian_function, k_min=-pi, k_max=pi, precision=100)


# read and write
x, y = gjh.read_one_dimensional_data(filename='a')
x, y, matrix = gjh.read_two_dimensional_data(filename='a')
gjh.write_one_dimensional_data(x, y, filename='a')
gjh.write_two_dimensional_data(x, y, matrix, filename='a')


# plot figures
gjh.plot(x, y, xlabel='x', ylabel='y', title='', filename='a', show=1, save=0, type='', y_min=None, y_max=None)
gjh.plot_3d_surface(x, y, matrix, xlabel='x', ylabel='y', zlabel='z', title='', filename='a', show=1, save=0, z_min=None, z_max=None)
gjh.plot_contour(x, y, matrix, xlabel='x', ylabel='y', title='', filename='a', show=1, save=0)


# download
gjh.download_with_scihub(address=None, num=1)