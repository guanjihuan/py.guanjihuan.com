import guan
import numpy as np
import cmath
from math import *

def hamiltonian_function(k): # SSH model
    gamma = 0.5
    lambda0 = 1
    delta = 0
    hamiltonian = np.zeros((2, 2), dtype=complex)
    hamiltonian[0,0] = delta
    hamiltonian[1,1] = -delta
    hamiltonian[0,1] = gamma+lambda0*cmath.exp(-1j*k)
    hamiltonian[1,0] = gamma+lambda0*cmath.exp(1j*k)
    return hamiltonian

wilson_loop_array = guan.calculate_wilson_loop(hamiltonian_function)
print('wilson loop =', wilson_loop_array)
p = np.log(wilson_loop_array)/2/pi/1j
print('p =', p, '\n')