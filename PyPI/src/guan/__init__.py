# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com/about (Ji-Huan Guan, 关济寰). The primary location of this package is on website https://py.guanjihuan.com. The GitHub location of this package is on https://github.com/guanjihuan/py.guanjihuan.com.

from basic_functions import *
from Fourier_transform import *
from Hamiltonian_of_finite_size_systems import *
from Hamiltonian_of_models_in_reciprocal_space import *
from band_structures_and_wave_functions import *
from Green_functions import *
from density_of_states import *
from quantum_transport import *
from topological_invariant import *
from plot_figures import *
from read_and_write import *
from file_processing import *
from data_processing import *

import guan
rand_number = guan.get_random_number(start=1, end=10)
if rand_number == 5:
    guan.notification_of_upgrade()