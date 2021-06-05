import gjh
import numpy as np

fermi_energy = 0
h00 = gjh.finite_size_along_one_direction(4)
h01 = np.identity(4)
gjh.print_or_write_scattering_matrix(fermi_energy, h00, h01)