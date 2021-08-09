# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com. The primary location of this package is on website https://py.guanjihuan.com.

# calculate scattering matrix

import numpy as np
import copy
from .calculate_Green_functions import *
from .calculate_conductance import *


def if_active_channel(k_of_channel):
    if np.abs(np.imag(k_of_channel))<1e-6:
        if_active = 1
    else:
        if_active = 0
    return if_active

def get_k_and_velocity_of_channel(fermi_energy, h00, h01):
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    transfer = transfer_matrix(fermi_energy, h00, h01)
    eigenvalue, eigenvector = np.linalg.eig(transfer)
    k_of_channel = np.log(eigenvalue)/1j
    ind = np.argsort(np.real(k_of_channel))
    k_of_channel = np.sort(k_of_channel)
    temp = np.zeros((2*dim, 2*dim), dtype=complex)
    temp2 = np.zeros((2*dim), dtype=complex)
    i0 = 0
    for ind0 in ind:
        temp[:, i0] = eigenvector[:, ind0]
        temp2[i0] = eigenvalue[ind0]
        i0 += 1
    eigenvalue = copy.deepcopy(temp2)
    temp = temp[0:dim, :]
    factor = np.zeros(2*dim, dtype=complex)
    for dim0 in range(dim):
        factor = factor+np.square(np.abs(temp[dim0, :]))
    for dim0 in range(2*dim):
        temp[:, dim0] = temp[:, dim0]/np.sqrt(factor[dim0])
    velocity_of_channel = np.zeros((2*dim), dtype=complex)
    for dim0 in range(2*dim):
        velocity_of_channel[dim0] = eigenvalue[dim0]*np.dot(np.dot(temp[0:dim, :].transpose().conj(), h01),temp[0:dim, :])[dim0, dim0]
    velocity_of_channel = -2*np.imag(velocity_of_channel)
    eigenvector = copy.deepcopy(temp) 
    return k_of_channel, velocity_of_channel, eigenvalue, eigenvector

def get_classified_k_velocity_u_and_f(fermi_energy, h00, h01):
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    k_of_channel, velocity_of_channel, eigenvalue, eigenvector = get_k_and_velocity_of_channel(fermi_energy, h00, h01)
    ind_right_active = 0; ind_right_evanescent = 0; ind_left_active = 0; ind_left_evanescent = 0
    k_right = np.zeros(dim, dtype=complex); k_left = np.zeros(dim, dtype=complex)
    velocity_right = np.zeros(dim, dtype=complex); velocity_left = np.zeros(dim, dtype=complex)
    lambda_right = np.zeros(dim, dtype=complex); lambda_left = np.zeros(dim, dtype=complex)
    u_right = np.zeros((dim, dim), dtype=complex); u_left = np.zeros((dim, dim), dtype=complex)
    for dim0 in range(2*dim):
        if_active = if_active_channel(k_of_channel[dim0])
        if if_active_channel(k_of_channel[dim0]) == 1:
            direction = np.sign(velocity_of_channel[dim0])
        else:
            direction = np.sign(np.imag(k_of_channel[dim0]))
        if direction == 1:
            if if_active == 1:  # right-moving active channel
                k_right[ind_right_active] = k_of_channel[dim0]
                velocity_right[ind_right_active] = velocity_of_channel[dim0]
                lambda_right[ind_right_active] = eigenvalue[dim0]
                u_right[:, ind_right_active] = eigenvector[:, dim0]
                ind_right_active += 1
            else:               # right-moving evanescent channel
                k_right[dim-1-ind_right_evanescent] = k_of_channel[dim0]
                velocity_right[dim-1-ind_right_evanescent] = velocity_of_channel[dim0]
                lambda_right[dim-1-ind_right_evanescent] = eigenvalue[dim0]
                u_right[:, dim-1-ind_right_evanescent] = eigenvector[:, dim0]
                ind_right_evanescent += 1
        else:
            if if_active == 1:  # left-moving active channel
                k_left[ind_left_active] = k_of_channel[dim0]
                velocity_left[ind_left_active] = velocity_of_channel[dim0]
                lambda_left[ind_left_active] = eigenvalue[dim0]
                u_left[:, ind_left_active] = eigenvector[:, dim0]
                ind_left_active += 1
            else:               # left-moving evanescent channel
                k_left[dim-1-ind_left_evanescent] = k_of_channel[dim0]
                velocity_left[dim-1-ind_left_evanescent] = velocity_of_channel[dim0]
                lambda_left[dim-1-ind_left_evanescent] = eigenvalue[dim0]
                u_left[:, dim-1-ind_left_evanescent] = eigenvector[:, dim0]
                ind_left_evanescent += 1
    lambda_matrix_right = np.diag(lambda_right)
    lambda_matrix_left = np.diag(lambda_left)
    f_right = np.dot(np.dot(u_right, lambda_matrix_right), np.linalg.inv(u_right))
    f_left = np.dot(np.dot(u_left, lambda_matrix_left), np.linalg.inv(u_left))
    return k_right, k_left, velocity_right, velocity_left, f_right, f_left, u_right, u_left, ind_right_active

def calculate_scattering_matrix(fermi_energy, h00, h01, length=100):
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    k_right, k_left, velocity_right, velocity_left, f_right, f_left, u_right, u_left, ind_right_active = get_classified_k_velocity_u_and_f(fermi_energy, h00, h01)
    right_self_energy = np.dot(h01, f_right)
    left_self_energy = np.dot(h01.transpose().conj(), np.linalg.inv(f_left))
    for i0 in range(length):
        if i0 == 0:
            green_nn_n = green_function(fermi_energy, h00, broadening=0, self_energy=left_self_energy)
            green_00_n = copy.deepcopy(green_nn_n)
            green_0n_n = copy.deepcopy(green_nn_n)
            green_n0_n = copy.deepcopy(green_nn_n)
        elif i0 != length-1: 
            green_nn_n = green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0) 
        else:
            green_nn_n = green_function_nn_n(fermi_energy, h00, h01, green_nn_n, broadening=0, self_energy=right_self_energy)
        green_00_n = green_function_ii_n(green_00_n, green_0n_n, h01, green_nn_n, green_n0_n)
        green_0n_n = green_function_in_n(green_0n_n, h01, green_nn_n)
        green_n0_n = green_function_ni_n(green_nn_n, h01, green_n0_n)
    temp = np.dot(h01.transpose().conj(), np.linalg.inv(f_right)-np.linalg.inv(f_left))
    transmission_matrix = np.dot(np.dot(np.linalg.inv(u_right), np.dot(green_n0_n, temp)), u_right) 
    reflection_matrix = np.dot(np.dot(np.linalg.inv(u_left), np.dot(green_00_n, temp)-np.identity(dim)), u_right)
    for dim0 in range(dim):
        for dim1 in range(dim):
            if_active = if_active_channel(k_right[dim0])*if_active_channel(k_right[dim1])
            if if_active == 1:
                transmission_matrix[dim0, dim1] = np.sqrt(np.abs(velocity_right[dim0]/velocity_right[dim1])) * transmission_matrix[dim0, dim1]
                reflection_matrix[dim0, dim1] = np.sqrt(np.abs(velocity_left[dim0]/velocity_right[dim1]))*reflection_matrix[dim0, dim1]
            else:
                transmission_matrix[dim0, dim1] = 0
                reflection_matrix[dim0, dim1] = 0
    sum_of_tran_refl_array = np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])), axis=0)+np.sum(np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active])), axis=0)
    for sum_of_tran_refl in sum_of_tran_refl_array:
        if sum_of_tran_refl > 1.001:
            print('Error Alert: scattering matrix is not normalized!')
    return transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active

def print_or_write_scattering_matrix(fermi_energy, h00, h01, length=100, on_print=1, on_write=0):
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    transmission_matrix, reflection_matrix, k_right, k_left, velocity_right, velocity_left, ind_right_active = calculate_scattering_matrix(fermi_energy, h00, h01, length)
    if on_print == 1:
        print('\nActive channel (left or right) = ', ind_right_active)
        print('Evanescent channel (left or right) = ', dim-ind_right_active, '\n')
        print('K of right-moving active channels:\n', np.real(k_right[0:ind_right_active]))
        print('K of left-moving active channels:\n', np.real(k_left[0:ind_right_active]), '\n')
        print('Velocity of right-moving active channels:\n', np.real(velocity_right[0:ind_right_active]))
        print('Velocity of left-moving active channels:\n', np.real(velocity_left[0:ind_right_active]), '\n')
        print('Transmission matrix:\n', np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])))
        print('Reflection matrix:\n', np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active])), '\n')
        print('Total transmission of channels:\n', np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])), axis=0))
        print('Total reflection of channels:\n',np.sum(np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active])), axis=0))
        print('Sum of transmission and reflection of channels:\n', np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])), axis=0) + np.sum(np.square(np.abs(reflection_matrix[0:ind_right_active, 0:ind_right_active])), axis=0))
        print('Total conductance = ', np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active]))), '\n')
    if on_write == 1:
        with open('a.txt', 'w') as f:
            f.write('Active channel (left or right) = ' + str(ind_right_active) + '\n')
            f.write('Evanescent channel (left or right) = ' + str(dim - ind_right_active) + '\n\n')
            f.write('Channel               K                                     Velocity\n')
            for ind0 in range(ind_right_active):
                f.write('   '+str(ind0 + 1) + '   |    '+str(np.real(k_right[ind0]))+'            ' + str(np.real(velocity_right[ind0]))+'\n')
            f.write('\n')
            for ind0 in range(ind_right_active):
                f.write('  -' + str(ind0 + 1) + '   |    ' + str(np.real(k_left[ind0])) + '            ' + str(np.real(velocity_left[ind0])) + '\n')
            f.write('\nScattering matrix:\n              ')
            for ind0 in range(ind_right_active):
                f.write(str(ind0+1)+'               ')
            f.write('\n')
            for ind1 in range(ind_right_active):
                f.write('  '+str(ind1+1)+'    ')
                for ind2 in range(ind_right_active):
                    f.write('%f' % np.square(np.abs(transmission_matrix[ind1, ind2]))+'    ')
                f.write('\n')
            f.write('\n')
            for ind1 in range(ind_right_active):
                f.write(' -'+str(ind1+1)+'    ')
                for ind2 in range(ind_right_active):
                    f.write('%f' % np.square(np.abs(reflection_matrix[ind1, ind2]))+'    ')
                f.write('\n')
            f.write('\n')
            f.write('Total transmission of channels:\n'+str(np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active])), axis=0))+'\n')
            f.write('Total conductance = '+str(np.sum(np.square(np.abs(transmission_matrix[0:ind_right_active, 0:ind_right_active]))))+'\n')
