# GUAN is an open-source python package developed and maintained by https://www.guanjihuan.com. The primary location of this package is on website https://py.guanjihuan.com.

# find vector with the same gauge

import numpy as np
import cmath
from math import *

def find_vector_with_the_same_gauge_with_binary_search(vector_target, vector_ref, show_error=1, show_times=0, show_phase=0, n_test=10001, precision=1e-6):
    phase_1_pre = 0
    phase_2_pre = pi
    for i0 in range(n_test):
        test_1 = np.sum(np.abs(vector_target*cmath.exp(1j*phase_1_pre) - vector_ref))
        test_2 = np.sum(np.abs(vector_target*cmath.exp(1j*phase_2_pre) - vector_ref))
        if test_1 < precision:
            phase = phase_1_pre
            if show_times==1:
                print('Binary search times=', i0)
            break
        if i0 == n_test-1:
            phase = phase_1_pre
            if show_error==1:
                print('Gauge not found with binary search times=', i0)
        if test_1 < test_2:
            if i0 == 0:
                phase_1 = phase_1_pre-(phase_2_pre-phase_1_pre)/2
                phase_2 = phase_1_pre+(phase_2_pre-phase_1_pre)/2
            else:
                phase_1 = phase_1_pre
                phase_2 = phase_1_pre+(phase_2_pre-phase_1_pre)/2
        else:
            if i0 == 0:
                phase_1 = phase_2_pre-(phase_2_pre-phase_1_pre)/2
                phase_2 = phase_2_pre+(phase_2_pre-phase_1_pre)/2
            else:
                phase_1 = phase_2_pre-(phase_2_pre-phase_1_pre)/2
                phase_2 = phase_2_pre 
        phase_1_pre = phase_1
        phase_2_pre = phase_2
    vector_target = vector_target*cmath.exp(1j*phase)
    if show_phase==1:
        print('Phase=', phase)   
    return vector_target

def find_vector_with_fixed_gauge_by_making_one_component_real(vector, precision=0.005, index=None):
    if index == None:
        index = np.argmax(np.abs(vector))
    sign_pre = np.sign(np.imag(vector[index]))
    for phase in np.arange(0, 2*pi, precision):
        sign =  np.sign(np.imag(vector[index]*cmath.exp(1j*phase)))
        if np.abs(np.imag(vector[index]*cmath.exp(1j*phase))) < 1e-9 or sign == -sign_pre:
            break
        sign_pre = sign
    vector = vector*cmath.exp(1j*phase)
    if np.real(vector[index]) < 0:
        vector = -vector
    return vector 