#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the Saleh-Valenzuela model

@author: grp2
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Initial Settings
# =============================================================================
"""
    Parameters
    ----------
    Lamda_cluster : TYPE
        Cluster arrival rate, i.e., paraneter assiciated with the cluster 
        intensity.
    lambda_ray : TYPE
        Cluster arrival rate, i.e., paraneter assiciated with the within 
        cluster intensity.
    Gamma_cluster : TYPE
        Cluster arrival decay time constant (cluster arrival decay rate).
    gamma_ray : TYPE
        Within cluster arrival decay time constant (cluster arrival decay rate).
    Q : TYPE
        Average power gain of the first component within the first cluster 
        (Average power gain of th first arrival).
    K : integer
        Number of frequency points.
    B : TYPE
        Bandwidth
"""
np.random.seed(1337)
Lambda_cluster = 1*1E7
Lambda_ray = 1E9
Gamma_cluster = 1E-8
Gamma_ray = 1E-9
Q = 5*1E-8

K = 801
B = 4*1E9
t_max = (K-1)/B
Delta_f = 1/t_max
linspace = np.linspace(0,t_max,K)


# =============================================================================
# Poisson Point Process (for delay)
# =============================================================================
def PPP_(intensity, start):
    """
    Sample a Poisson point process by sampling the number of points from a 
    Poisson point process and uniformly distributing them since it is homogenius.
    """
    t_interval = t_max-start

    number_of_points = np.random.poisson(intensity*t_interval)
    PPP = np.random.uniform(0, t_interval, (number_of_points,1))
    return (np.concatenate(np.vstack([0,PPP])))


# =============================================================================
# Complex gain
# =============================================================================
def gain(Q, T, tau):
    """
    Simulates the complex gain
    """
    var = Q*np.exp(-T/Gamma_cluster-tau/Gamma_ray)
    
    real = np.random.normal(0,np.sqrt(var)) 
    im = 1j* np.random.normal(0,np.sqrt(var))
    return real + im


# =============================================================================
# Simulate the SV model
# =============================================================================
def simulate_SV(Lambda_cluster, Lambda_ray, Q, K):
    """
    Channel model given the parameters
    """
    H = np.zeros(K, dtype = np.complex)
    T = PPP_(Lambda_cluster,0)
    tau_list = []
    for l in range(len(T)):
        tau = PPP_(Lambda_ray,T[l])
        tau_list.append(tau)
        for p in range(len(tau)):
            beta_l = gain(Q, T[l],tau[p])
            H += beta_l*np.exp(-1j*2*np.pi*Delta_f*np.arange(K)*(T[l]+tau[p]))
    return H, T, tau_list


# =============================================================================
# Plot
# =============================================================================
def plot_SV(T, linspace, scale = 'Normal'):
    """
    Simulate data from the S-V model and plot the data along with the cluster
    arrival points.
    """
    
    index = []
    for j in range(len(T)):
        index.append(np.argmin(np.abs(linspace - T[j])))
    
    y_abs_squared = np.abs(np.fft.ifft(H_k))**2
    
    plt.figure(figsize=(4.5,3.5))
    plt.title(r'The S-V model')
    plt.plot(linspace,y_abs_squared,label = r'$\vert y(t) \vert^2$')
    plt.plot(linspace[index],(y_abs_squared[index]),'rx',label = r'$T_l$')
    plt.xlabel(r'Delay $[s]$')
    plt.ylabel(r'Power')
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(4.5,3.5))
    plt.title( r'The S-V model')
    plt.plot(linspace[:],10*np.log10(y_abs_squared),label = r'$10\log_{10}\vert y(t) \vert^2$')
    plt.plot(linspace[index],10*np.log10(y_abs_squared[index]),'rx',label = r'$T_l$')
    plt.ylabel(r'Power $[dB]$')
    plt.xlabel(r'Delay $[s]$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/sv_pdp.pdf")


def delay_power_intensity(t_max, K, Gamma_cluster, Gamma_ray, Lambda_cluster, Lambda_ray, Q):
    """
    Calculate the delay power intensity of the S-V model.
    """
    k_1 = Lambda_cluster * ( 1 + Lambda_ray * ((Gamma_cluster * Gamma_ray)/(Gamma_cluster-Gamma_ray)))
    k_2 = Lambda_ray*(1-Lambda_cluster*(Gamma_cluster * Gamma_ray)/(Gamma_cluster-Gamma_ray))
    P_t= Q*(k_1 * np.exp(-linspace/Gamma_cluster)+ k_2 * np.exp(-linspace / Gamma_ray))
    P_t[0] += Q
    return P_t


def plot_SV_power_intensity(N_sim):
    """
    Plot the delay power intensity as well as the average power delay profile.
    """
    y_abs_squared = []
    for n in range(N_sim):
        y_abs_squared.append(np.abs(np.fft.ifft(simulate_SV(Lambda_cluster, Lambda_ray, Q, K)[0]))**2)
        
    y_mean = np.mean(np.asarray(y_abs_squared), axis = 0)
    
    P_t_dB = 10*np.log10(delay_power_intensity(t_max, K, Gamma_cluster, Gamma_ray, Lambda_cluster, Lambda_ray, Q))-10*np.log10(B)
    y_mean_dB = 10*np.log10(y_mean)

    plt.figure(figsize=(4.5,3.5))
    #plt.title(r'Average power delay profile')
    plt.plot(linspace[:], P_t_dB[:],label = r'Delay Power Intensity $[dB]$')
    plt.plot(linspace[:],y_mean_dB[:],label = r'APDP $[dB]$')
    plt.ylabel(r'Power $[dB]$')
    plt.xlabel(r'Delay $[s]$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/SV_delay_power_intensity.pdf")


if __name__ == "__main__":
    # Compute with seed for reproducibility
    np.random.seed(1337)
    H_k, T, tau_list = simulate_SV(Lambda_cluster, Lambda_ray, Q, K)
    plot_SV(T, linspace)
    np.random.seed(1337)
    plot_SV_power_intensity(N_sim = 1000)
