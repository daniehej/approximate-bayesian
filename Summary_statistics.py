#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Our proposed summary statistics as well as summary statistics from [Bharti 2020]

@author: grp2
"""

import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Initial Settings
# =============================================================================
"""
    Parameters
    ----------
    Lamda_cluster : 
        Cluster arrival rate, i.e., paraneter assiciated with the cluster 
        intensity.
    lambda_ray : 
        Cluster arrival rate, i.e., paraneter assiciated with the within 
        cluster intensity.
    Gamma_cluster : 
        Cluster arrival decay time constant (cluster arrival decay rate).
    gamma_ray : 
        Within cluster arrival decay time constant (cluster arrival decay rate).
    Q : 
        Average power gain of the first component within the first cluster 
        (Average power gain of th first arrival).
    K : 
        Number of frequency points.
    B : 
        Bandwidth
"""
from Spencer_model import (
    linspace, M, 
    Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray,
    simulate_Spencer_ULA
    )


# =============================================================================
# Our proposed summaries
# =============================================================================
def D_diff(y_m, y_m1):
    """
    Absolute squared difference of magnitude.
    """
    return np.abs(np.abs(y_m)-np.abs(y_m1))**2


def D_diff_angle(y_m, y_m1):
    """
    Absolute squared difference of phase.
    """
    return np.abs(np.angle(y_m)-np.angle(y_m1))**2


def summary_pair_antenna(H,m1,m2):
    """
    Returns the sample mean and sample variance, with respect to time, of the
    absolute difference of the signals recieved by antenna m1 and m2,
    respectively.
    """
    h = np.fft.ifft(H, axis=1)
    mean_D = 0 
    var_D = 0
    diff = D_diff(h[m1,:], h[m2,:])
    mean_D = np.mean(diff)
    var_D = np.var(diff)
    return mean_D, var_D


def summary_pair_antenna_angle(H,m1,m2):
    """
    Returns the sample mean and sample variance, with respect to time, of the
    absolute difference of the angle of signals recieved by antenna m1 and m2,
    respectively.
    """
    h = np.fft.ifft(H, axis=1)
    mean_D_angle = 0 
    var_D_angle = 0
    diff_angle = D_diff_angle(h[m1,:], h[m2,:])
    mean_D_angle = np.mean(diff_angle)
    var_D_angle = np.var(diff_angle)
    return mean_D_angle, var_D_angle


def summary_all_pair_antenna_angle(H):
    """
    Returns the sample mean, with respect to antennas, for the magnitude of the
    sample mean and sample variance of all pair of consecutive antennas.
    """
    total_mean_D_angle = 0 
    total_var_D_angle = 0
    for i in range(M-1):
        diff_list_angle = summary_pair_antenna_angle(H, i, i+1)
        total_mean_D_angle += diff_list_angle[0]
        total_var_D_angle += diff_list_angle[1]
    total_mean_D_angle = total_mean_D_angle/(M-1)
    total_var_D_angle = total_var_D_angle/(M-1)
    return total_mean_D_angle, total_var_D_angle


def summary_all_pair_antenna(H):
    """
    Returns the sample mean, with respect to antennas, for the angle of the 
    sample mean and sample variance of all pair of consecutive antennas.
    """
    total_mean_D = 0 
    total_var_D = 0
    for i in range(M-1):
        diff_list = summary_pair_antenna(H, i, i+1)
        total_mean_D += diff_list[0]
        total_var_D += diff_list[1]
    total_mean_D = total_mean_D/(M-1)
    total_var_D = total_var_D/(M-1)
    return total_mean_D, total_var_D


# =============================================================================
# Plot our proposed summaries
# =============================================================================
def plot_informativ_antenna_pair(m1,m2):
    """
    Plot informativity about the angle for the antenna pair m1, m2.
    """
    prior = np.linspace(0.01, np.deg2rad(30), 100)
    mean_list = np.zeros(100)
    var_list = np.zeros(100)
    mean_list2=np.zeros(100)
    var_list2 = np.zeros(100)
    for j in range(100):
        H = simulate_Spencer_ULA(Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray, prior[j])
        diff_list = summary_pair_antenna(H,m1,m2)
        diff_list2 =summary_pair_antenna_angle(H,m1,m2)
        mean_list[j] = diff_list[0]
        var_list[j] = diff_list[1]
        mean_list2[j] = diff_list2[0]
        var_list2[j] = diff_list2[1]
    
    plt.figure(figsize=(4.5,3.5))
    plt.plot(np.rad2deg(prior), mean_list, 'bo', alpha=0.5)
    plt.xlabel(r"$\sigma$ $[\degree]$")
    plt.ylabel(r'$\mu_{D_{(0,1)}}$')
    plt.tight_layout()
    plt.savefig("figures/developed_summary_pair_mean_magnitude.pdf")
    plt.show()
    plt.figure(figsize=(4.5,3.5))
    plt.plot(np.rad2deg(prior), var_list, 'bo', alpha=0.5)
    plt.xlabel(r"$\sigma$ $[\degree]$")
    plt.ylabel(r'$\sigma^2_{D_{(%d,%d)}}$'%(m1,m2))
    plt.tight_layout()
    plt.savefig("figures/developed_summary_pair_var_magnitude.pdf")
    plt.show()
    plt.figure(figsize=(4.5,3.5))
    plt.plot(np.rad2deg(prior), mean_list2, 'bo', alpha=0.5)
    plt.xlabel(r"$\sigma$ $[\degree]$")
    plt.ylabel(r'$\mu_{A_{(0,1)}}$')
    plt.tight_layout()
    plt.savefig("figures/developed_summary_pair_mean_angle.pdf")
    plt.show()
    plt.figure(figsize=(4.5,3.5))
    plt.plot(np.rad2deg(prior), var_list2, 'bo', alpha=0.5)
    plt.xlabel(r"$\sigma$ $[\degree]$")
    plt.ylabel(r'$\sigma^2_{A_{(%d,%d)}}$'%(m1,m2))
    plt.tight_layout()
    plt.savefig("figures/developed_summary_pair_var_angle.pdf")
    plt.show()


def plot_informativ_all_antennas():
    """
    Plot informativity about the angle over all antennas in the array.
    """
    prior = np.linspace(0.01, np.deg2rad(30), 100)
    mean_list = np.zeros(100)
    var_list = np.zeros(100)
    mean_list2=np.zeros(100)
    var_list2 = np.zeros(100)
    for j in range(100):
        H = simulate_Spencer_ULA(Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray, prior[j])
        diff_list = summary_all_pair_antenna(H)
        diff_list2 =summary_all_pair_antenna_angle(H)
        mean_list[j] = diff_list[0]
        var_list[j] = diff_list[1]
        mean_list2[j] = diff_list2[0]
        var_list2[j] = diff_list2[1]

    plt.figure(figsize=(4.5,3.5))
    plt.plot(np.rad2deg(prior), mean_list, 'bo', alpha=0.5)
    plt.xlabel(r"$\sigma$ $[\degree]$")
    plt.ylabel(r'$\hat{\mu}_D$')
    plt.tight_layout()
    plt.savefig("figures/developed_summary_all_mean_magnitude.pdf")
    plt.show()
    plt.figure(figsize=(4.5,3.5))
    plt.plot(np.rad2deg(prior), var_list, 'bo', alpha=0.5)
    plt.xlabel(r"$\sigma$ $[\degree]$")
    plt.ylabel(r'$\hat{\sigma}^2_D$')
    plt.tight_layout()
    plt.savefig("figures/developed_summary_all_var_magnitude.pdf")
    plt.show()
    plt.figure(figsize=(4.5,3.5))
    plt.plot(np.rad2deg(prior), mean_list2, 'bo', alpha=0.5)
    plt.xlabel(r"$\sigma$ $[\degree]$")
    plt.ylabel(r'$\hat{\mu}_A$')
    plt.tight_layout()
    plt.savefig("figures/developed_summary_all_mean_angle.pdf")
    plt.show()
    plt.figure(figsize=(4.5,3.5))
    plt.plot(np.rad2deg(prior), var_list2, 'bo', alpha=0.5)
    plt.xlabel(r"$\sigma$ $[\degree]$")
    plt.ylabel(r'$\hat{\sigma}^2_A$')
    plt.tight_layout()
    plt.savefig("figures/developed_summary_all_var_angle.pdf")
    plt.show()


# =============================================================================
# Summaries from [Bharti 2020] as well as our proposed summaries
# =============================================================================
def temporal_moment(y, n, order):
    """
    Temporal moment of order order
    """
    return integrate.simps(linspace**order*np.abs(y)**n, linspace)


def summary_bharti(data):
    """
    Summaries from [Bharti 2020]
    """
    h = np.fft.ifft(data)
    tm2 = np.array([temporal_moment(h, 2, i) for i in range(3)])
    tm4 = np.array([temporal_moment(h, 4, i) for i in range(3)])
    mu2 = np.mean(tm2, axis=1)
    mu4 = np.mean(tm4, axis=1)
    cov2 = np.cov(tm2)
    cov4 = np.cov(tm4)
    y_abs_squared= (np.abs(h)**2) #PDP
    y_mean = np.mean(np.asarray(y_abs_squared), axis = 0) #APDP
    y_mean_db = 10*np.log10(y_mean) #APDP i dB
    y_mean_sorted = np.sort(y_mean_db)[::-1] #APDP descending order
    y_mean_diff = y_mean_db - y_mean_sorted
    rho_max = np.max(y_mean_diff)
    rho_var = np.var(y_mean_diff[y_mean_diff>0])
    summary_vec = [mu2[0], mu2[1], mu2[2], mu4[0], mu4[1], mu4[2], cov2[1,2], cov2[0,2], cov2[0,1], cov2[0,0], cov2[1,1], cov2[2,2],cov4[1,2], cov4[0,2], cov4[0,1], cov4[0,0], cov4[1,1], cov4[2,2], rho_max, rho_var]
    return np.array(summary_vec)


def summary_ours(data):
    """
    Calculates summaries from [Bharti 2020] as well as our proposed summaries.
    """
    h = np.fft.ifft(data)
    tm2 = np.array([temporal_moment(h, 2, i) for i in range(3)])
    tm4 = np.array([temporal_moment(h, 4, i) for i in range(3)])
    mu2 = np.mean(tm2, axis=1)
    mu4 = np.mean(tm4, axis=1)
    cov2 = np.cov(tm2)
    cov4 = np.cov(tm4)
    y_abs_squared= (np.abs(h)**2) #PDP
    y_mean = np.mean(np.asarray(y_abs_squared), axis = 0) #APDP
    y_mean_db = 10*np.log10(y_mean) #APDP i dB
    y_mean_sorted = np.sort(y_mean_db)[::-1] #APDP descending order
    y_mean_diff = y_mean_db - y_mean_sorted
    rho_max = np.max(y_mean_diff)
    rho_var = np.var(y_mean_diff[y_mean_diff>0])
    
    total_mean_D, total_var_D = summary_all_pair_antenna(data)
    total_mean_D_angle, total_var_D_angle = summary_all_pair_antenna_angle(data)
    
    summary_vec = [mu2[0], mu2[1], mu2[2], mu4[0], mu4[1], mu4[2], cov2[1,2], cov2[0,2], cov2[0,1], cov2[0,0], cov2[1,1], cov2[2,2],cov4[1,2], cov4[0,2], cov4[0,1], cov4[0,0], cov4[1,1], cov4[2,2], rho_max, rho_var, total_mean_D, total_var_D, total_mean_D_angle, total_var_D_angle]
    return np.array(summary_vec)


def summary_only_ours(data):
    """
    Calculates our proposed summaries.
    """
    total_mean_D, total_var_D = summary_all_pair_antenna(data)
    total_mean_D_angle, total_var_D_angle = summary_all_pair_antenna_angle(data)
    summary_vec = [total_mean_D, total_var_D, total_mean_D_angle, total_var_D_angle]
    return np.array(summary_vec)


# =============================================================================
# Plot for informativity of summaries from [Bharti 2020]
# =============================================================================
def informativ_Bharti(N):
    """
    Creates plots to determine whether the summaries from [Bharti 2020] are 
    informative about the parameter sigma in the Spencer model.
    """
    var_laplace_prior = np.linspace(0.01, np.deg2rad(30), N)
    summary_vec = np.zeros((N,20))
    for j in range(N):
        H = simulate_Spencer_ULA(Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray, var_laplace_prior[j])
        summary_vec[j,:] = summary_bharti(H)
        print("Summaries have been computed")
    summary_vec = np.array(summary_vec)
    
    #Plot
    fig, axs = plt.subplots(7, 3, sharex='col', figsize=(10,14))
    for row in range(7):
        for col in range(3):
            if row != 6 or col !=2:
                axs[row,col].plot(np.rad2deg(var_laplace_prior), summary_vec[:,row*3 + col],'bo', alpha=0.5)
            
    axs[6,0].set_xlabel(r"$\sigma$ $[\degree]$")
    axs[6,1].set_xlabel(r"$\sigma$ $[\degree]$")    
    axs[5,2].set_xlabel(r"$\sigma$ $[\degree]$")
    
    #First row
    axs[0,0].set_ylabel(r"mean$(m_0^{(2)})$")
    axs[0,1].set_ylabel(r"mean$(m_1^{(2)})$")
    axs[0,2].set_ylabel(r"mean$(m_2^{(2)})$")
    
    #second row
    axs[1,0].set_ylabel(r"mean$(m_0^{(4)})$")
    axs[1,1].set_ylabel(r"mean$(m_1^{(4)})$")
    axs[1,2].set_ylabel(r"mean$(m_2^{(4)})$")
    
    #third row
    axs[2,0].set_ylabel(r"cov$(m_1^{(2)},m_2^{(2)})$")
    axs[2,1].set_ylabel(r"cov$(m_0^{(2)},m_2^{(2)})$")
    axs[2,2].set_ylabel(r"cov$(m_0^{(2)},m_1^{(2)})$")
    
    #fourth row
    axs[3,0].set_ylabel(r"cov$(m_0^{(2)},m_0^{(2)})$")
    axs[3,1].set_ylabel(r"cov$(m_1^{(2)},m_1^{(2)})$")
    axs[3,2].set_ylabel(r"cov$(m_2^{(2)},m_2^{(2)})$")
    
    #fifth row
    axs[4,0].set_ylabel(r"cov$(m_1^{(4)},m_2^{(4)})$")
    axs[4,1].set_ylabel(r"cov$(m_0^{(4)},m_2^{(4)})$")
    axs[4,2].set_ylabel(r"cov$(m_0^{(4)},m_1^{(4)})$")
    
    #sixth row
    axs[5,0].set_ylabel(r"cov$(m_0^{(4)},m_0^{(4)})$")
    axs[5,1].set_ylabel(r"cov$(m_1^{(4)},m_1^{(4)})$")
    axs[5,2].set_ylabel(r"cov$(m_2^{(4)},m_2^{(4)})$")
    
    #seventh row
    axs[6,0].set_ylabel(r"$\varrho_{max}$")
    axs[6,1].set_ylabel(r"$\varrho_{var}$")
    axs[6,2].axis('off')
    plt.subplots_adjust(wspace=0.33)
    plt.subplots_adjust(hspace=0.25)
    #plt.tight_layout()
    plt.savefig("figures/spencer_summaries_bharti_informativity.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Compute with seed for reproducibility
    np.random.seed(1337)
    plot_informativ_all_antennas()
    np.random.seed(1337)
    plot_informativ_antenna_pair(0, 1)
    np.random.seed(1337)
    informativ_Bharti(100)
                