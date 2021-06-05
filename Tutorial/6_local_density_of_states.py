import gjh
import numpy as np

fermi_energy = 0
N1 = 3
N2 = 4
hamiltonian = gjh.finite_size_along_two_directions_for_square_lattice(N1,N2)
LDOS = gjh.local_density_of_states_for_square_lattice(fermi_energy, hamiltonian, N1=N1, N2=N2)
print('square lattice:\n', LDOS, '\n')

h00 = gjh.finite_size_along_one_direction(N2)
h01 = np.identity(N2)
LDOS = gjh.local_density_of_states_for_square_lattice_using_dyson_equation(fermi_energy, h00=h00, h01=h01, N2=N2, N1=N1)
print(LDOS, '\n\n')
gjh.plot_contour(range(N1), range(N2), LDOS)


N1 = 3
N2 = 4
N3 = 5
hamiltonian = gjh.finite_size_along_three_directions_for_cubic_lattice(N1, N2, N3)
LDOS = gjh.local_density_of_states_for_cubic_lattice(fermi_energy, hamiltonian, N1=N1, N2=N2, N3=N3)
print('cubic lattice:\n', LDOS, '\n')

h00 = gjh.finite_size_along_two_directions_for_square_lattice(N2, N3)
h01 = np.identity(N2*N3)
LDOS = gjh.local_density_of_states_for_cubic_lattice_using_dyson_equation(fermi_energy, h00, h01, N3=N3, N2=N2, N1=N1)
print(LDOS)