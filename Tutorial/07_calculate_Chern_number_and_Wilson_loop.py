import guan
import numpy as np
from math import *

# calculate Chern number
chern_number = guan.calculate_chern_number_for_square_lattice(guan.hamiltonian_of_one_QAH_model, precision=100)
print('\nChern number=', chern_number)

# calculate Wilson loop
wilson_loop_array = guan.calculate_wilson_loop(guan.hamiltonian_of_ssh_model)
print('Wilson loop =', wilson_loop_array)
p = np.log(wilson_loop_array)/2/pi/1j
print('p =', p, '\n')