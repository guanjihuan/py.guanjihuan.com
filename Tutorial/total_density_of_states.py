import gjh
import numpy as np

hamiltonian = gjh.finite_size_along_two_directions_for_square_lattice(2,2)
fermi_energy_array = np.linspace(-4, 4, 400)
total_dos_array = gjh.total_density_of_states_with_fermi_energy_array(fermi_energy_array, hamiltonian, broadening=0.1)
gjh.plot(fermi_energy_array, total_dos_array, xlabel='E', ylabel='Total DOS', type='-o')