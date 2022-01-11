# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com/about. The primary location of this package is on website https://py.guanjihuan.com.

# calculate Green functions

import numpy as np
import guan

def green_function(fermi_energy, hamiltonian, broadening, self_energy=0):
    if np.array(hamiltonian).shape==():
        dim = 1
    else:
        dim = np.array(hamiltonian).shape[0]
    green = np.linalg.inv((fermi_energy+broadening*1j)*np.eye(dim)-hamiltonian-self_energy)
    return green

def green_function_nn_n(fermi_energy, h00, h01, green_nn_n_minus, broadening, self_energy=0):
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]   
    green_nn_n = np.linalg.inv((fermi_energy+broadening*1j)*np.identity(dim)-h00-np.dot(np.dot(h01.transpose().conj(), green_nn_n_minus), h01)-self_energy)
    return green_nn_n

def green_function_in_n(green_in_n_minus, h01, green_nn_n):
    green_in_n = np.dot(np.dot(green_in_n_minus, h01), green_nn_n)
    return green_in_n

def green_function_ni_n(green_nn_n, h01, green_ni_n_minus):
    h01 = np.array(h01)
    green_ni_n = np.dot(np.dot(green_nn_n, h01.transpose().conj()), green_ni_n_minus)
    return green_ni_n

def green_function_ii_n(green_ii_n_minus, green_in_n_minus, h01, green_nn_n, green_ni_n_minus):
    green_ii_n = green_ii_n_minus+np.dot(np.dot(np.dot(np.dot(green_in_n_minus, h01), green_nn_n), h01.transpose().conj()),green_ni_n_minus)
    return green_ii_n

def transfer_matrix(fermi_energy, h00, h01):
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    transfer = np.zeros((2*dim, 2*dim), dtype=complex)
    transfer[0:dim, 0:dim] = np.dot(np.linalg.inv(h01), fermi_energy*np.identity(dim)-h00)
    transfer[0:dim, dim:2*dim] = np.dot(-1*np.linalg.inv(h01), h01.transpose().conj())
    transfer[dim:2*dim, 0:dim] = np.identity(dim)
    transfer[dim:2*dim, dim:2*dim] = 0
    return transfer

def surface_green_function_of_lead(fermi_energy, h00, h01):
    h01 = np.array(h01)
    if np.array(h00).shape==():
        dim = 1
    else:
        dim = np.array(h00).shape[0]
    fermi_energy = fermi_energy+1e-9*1j
    transfer = transfer_matrix(fermi_energy, h00, h01)
    eigenvalue, eigenvector = np.linalg.eig(transfer)
    ind = np.argsort(np.abs(eigenvalue))
    temp = np.zeros((2*dim, 2*dim), dtype=complex)
    i0 = 0
    for ind0 in ind:
        temp[:, i0] = eigenvector[:, ind0]
        i0 += 1
    s1 = temp[dim:2*dim, 0:dim]
    s2 = temp[0:dim, 0:dim]
    s3 = temp[dim:2*dim, dim:2*dim]
    s4 = temp[0:dim, dim:2*dim]
    right_lead_surface = np.linalg.inv(fermi_energy*np.identity(dim)-h00-np.dot(np.dot(h01, s2), np.linalg.inv(s1)))
    left_lead_surface = np.linalg.inv(fermi_energy*np.identity(dim)-h00-np.dot(np.dot(h01.transpose().conj(), s3), np.linalg.inv(s4)))
    return right_lead_surface, left_lead_surface

def self_energy_of_lead(fermi_energy, h00, h01):
    h01 = np.array(h01)
    right_lead_surface, left_lead_surface = surface_green_function_of_lead(fermi_energy, h00, h01)
    right_self_energy = np.dot(np.dot(h01, right_lead_surface), h01.transpose().conj())
    left_self_energy = np.dot(np.dot(h01.transpose().conj(), left_lead_surface), h01)
    gamma_right = (right_self_energy - right_self_energy.transpose().conj())*1j
    gamma_left = (left_self_energy - left_self_energy.transpose().conj())*1j
    return right_self_energy, left_self_energy, gamma_right, gamma_left

def self_energy_of_lead_with_h_LC_and_h_CR(fermi_energy, h00, h01, h_LC, h_CR):
    h_LC = np.array(h_LC)
    h_CR = np.array(h_CR)
    right_lead_surface, left_lead_surface = surface_green_function_of_lead(fermi_energy, h00, h01)
    right_self_energy = np.dot(np.dot(h_CR, right_lead_surface), h_CR.transpose().conj())
    left_self_energy = np.dot(np.dot(h_LC.transpose().conj(), left_lead_surface), h_LC)
    gamma_right = (right_self_energy - right_self_energy.transpose().conj())*1j
    gamma_left = (left_self_energy - left_self_energy.transpose().conj())*1j
    return right_self_energy, left_self_energy, gamma_right, gamma_left

def self_energy_of_lead_with_h_lead_to_center(fermi_energy, h00, h01, h_lead_to_center):
    h_lead_to_center = np.array(h_lead_to_center)
    right_lead_surface, left_lead_surface = surface_green_function_of_lead(fermi_energy, h00, h01)
    self_energy = np.dot(np.dot(h_lead_to_center.transpose().conj(), right_lead_surface), h_lead_to_center)
    gamma = (self_energy - self_energy.transpose().conj())*1j
    return self_energy, gamma

def green_function_with_leads(fermi_energy, h00, h01, h_LC, h_CR, center_hamiltonian):
    dim = np.array(center_hamiltonian).shape[0]
    right_self_energy, left_self_energy, gamma_right, gamma_left = self_energy_of_lead_with_h_LC_and_h_CR(fermi_energy, h00, h01, h_LC, h_CR)
    green = np.linalg.inv(fermi_energy*np.identity(dim)-center_hamiltonian-left_self_energy-right_self_energy)
    return green, gamma_right, gamma_left

def electron_correlation_function_green_n_for_local_current(fermi_energy, h00, h01, h_LC, h_CR, center_hamiltonian):
    right_self_energy, left_self_energy, gamma_right, gamma_left = guan.self_energy_of_lead_with_h_LC_and_h_CR(fermi_energy, h00, h01, h_LC, h_CR)
    green = guan.green_function(fermi_energy, center_hamiltonian, broadening=0, self_energy=left_self_energy+right_self_energy)
    G_n = np.imag(np.dot(np.dot(green, gamma_left), green.transpose().conj()))
    return G_n