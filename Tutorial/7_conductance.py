import gjh
import numpy as np

fermi_energy_array = np.linspace(-5, 5, 400)
h00 = gjh.finite_size_along_one_direction(4)
h01 = np.identity(4)
conductance_array = gjh.calculate_conductance_with_fermi_energy_array(fermi_energy_array, h00, h01)
gjh.plot(fermi_energy_array, conductance_array, xlabel='E', ylabel='Conductance', type='-o')