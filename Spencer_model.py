#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the Spencer model.

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
Lambda_cluster = 1E7
lambda_ray = 1E9
Gamma_cluster = 1E-8
gamma_ray = 1E-9
Q = 5*1E-8

K = 801

B = 4*1E9
t_max = (K-1)/B
var_laplace = np.deg2rad(20)

theta = [Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray, var_laplace]

Delta_f = 1/t_max
Delta_f_tilde = 1/(2*np.pi)

linspace = np.linspace(0,t_max,K) 

wave_length = 3E8/60E9
d = wave_length/2
M = 25

# =============================================================================
# Poisson Point Process (for delay)
# =============================================================================
def PPP_(intensity, start):
    """
    Sample a Poisson point process by sampling the number of points from a 
    Poisson distribution and uniformly distributing them since it is homogenius.
    """
    t_interval = -start+t_max
    #Simulate Poisson point process
    number_of_points = np.random.poisson(intensity*t_interval)# Poisson counting Process
    PPP = np.random.uniform(0, t_interval, number_of_points) #Distributing the number of points uniformly
    return np.concatenate(([0],PPP))

# =============================================================================
# Distributions (for angle)
# =============================================================================
def Phi_(T):
    """
    Cluster angle uniformly distributed on the interval [0, 2π). The first 
    cluster always arrives at π.
    """
    Phi = np.random.uniform(0, 2*np.pi, size=len(T)-1)
    return np.concatenate(([np.pi], Phi))


def omega(var_laplace, size):
    """
    Ray angle is Laplace distributed.

    """
    return np.random.laplace(0, var_laplace/np.sqrt(2), size)


# =============================================================================
# Complex gain (mark distribution)
# =============================================================================
def gain(Q, T, tau):
    """
    Simulates the complex gain.

    """
    var = Q*np.exp((-T/Gamma_cluster)-(tau/gamma_ray))
    real = np.random.normal(0, np.sqrt(var))
    im = 1j* np.random.normal(0, np.sqrt(var))
    return real + im


# =============================================================================
# Simulate the Spencer model 
# =============================================================================
def simulate_Spencer_ULA(Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray, var_laplace):
    """
    The channel model returns frequency domain simulations of data given the 
    parameters.
    
    """
    #Generate points from the point process
    T = PPP_(Lambda_cluster,0)
    tau_list = [PPP_(lambda_ray, t) for t in T]
    times_list = [tau_list[l] + T[l] for l in range(len(T))]
    
    Phi = Phi_(T)
    angle_ray = [omega(var_laplace, len(tau)) for tau in tau_list]
    angle_list = [angle_ray[l] + Phi[l] for l in range(len(T))]
    angle_sin = [np.sin(angles) for angles in angle_list]
    
    beta = [np.array([gain(Q, T[l], tau) for tau in tau_list[l]]) for l in range(len(T))]
    
    
    #Compute reused values for later use
    phase_const = -1j*2*(np.pi/wave_length)*d
    Ms = np.arange(M)
    Ks = np.arange(K)
    H_exp_const = -1j*2*np.pi*Delta_f
    
    #Calculate H
    H = np.zeros((M,K), dtype = np.complex)
    for l in range(len(T)):
        angle = np.exp(phase_const*np.outer(angle_sin[l], Ms))
        H_exp = np.exp(H_exp_const*np.outer(Ks, (times_list[l])).T)
        H_const_list = H_exp*beta[l][:, None]
        H += sum(np.outer(H_const_list[p], angle[p]).T for p in range(len(times_list[l])))
    return H


# =============================================================================
# Plot
# =============================================================================
def show_plot():
    """
    Simulate and plot data from the Spencer model.

    """
    H = simulate_Spencer_ULA(Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray, var_laplace)

    plt.figure(figsize=(4.5,3.5))
    for j in [0,24]:
        plt.plot(linspace, 10*np.log10(abs(np.fft.ifft(H[j]))**2),label = r'$10\log_{10}\vert y^{(%d)}(t)\vert^2$'%j, alpha=0.5)
    #plt.title('Spencer ULA model for 2 antennas')
    plt.ylabel('Power [dB]')
    plt.xlabel("Delay [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/spencer_ula_2_antennas.pdf")
    plt.show()
    
    
def plot_Spencer_power_intensity(N_sim):
    """
    Simulate N_sim realizations in order to calculate the average power delay
    profile. Subsequently, this is plotted.
    """
    y_abs_squared0 = []
    y_abs_squared1 = []

    for n in range(N_sim):
        H = simulate_Spencer_ULA(Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray, var_laplace)[:2]
        y_abs_squared0.append(np.abs(np.fft.ifft(H[0]))**2)
        y_abs_squared1.append(np.abs(np.fft.ifft(H[1]))**2)
        print(f"simulation # {n}")
        
    y_mean0 = np.mean(np.asarray(y_abs_squared0), axis = 0)
    y_mean1 = np.mean(np.asarray(y_abs_squared1), axis = 0)

    y_mean0_dB = 10*np.log10(y_mean0)
    y_mean1_dB = 10*np.log10(y_mean1)

    plt.figure(figsize=(4.5,3.5))
    plt.plot(linspace[:],y_mean0_dB[:],label = r'APDP Antenna 0 $[dB]$', alpha=0.5)
    plt.plot(linspace, y_mean1_dB, label= r"APDP Antenna 1 $[dB]$", alpha=0.5)
    plt.ylabel(r'Power $[dB]$')
    plt.xlabel(r'Delay $[s]$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/delay_power_intensity.pdf")
    
    plt.figure()
    plt.plot(linspace[:400], y_mean0_dB[:400], label = r'APDP Antenna 0 $[dB]$', alpha=0.5)
    plt.plot(linspace[:400], y_mean1_dB[:400], label= r"APDP Antenna 1 $[dB]$", alpha=0.5)
    plt.ylabel(r'Power $[dB]$')
    plt.xlabel(r'Delay $[s]$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/delay_power_intensity_400.pdf")
    
def plot_Spencer_PDP_Antenna_average():
    """
    Simulate a realization of the Spencer model and average the received signal
    over all M antennas.

    """
    H = simulate_Spencer_ULA(Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray, var_laplace)
    PDP = np.abs(np.fft.ifft(H, axis=1))**2
    PDP_average_dB = 10*np.log10(np.mean(PDP, axis=0))
    plt.figure(figsize=(4.5,3.5))
    plt.plot(linspace[:], PDP_average_dB, label = r'Mean of PDP of antennas $[dB]$')
    plt.ylabel(r'Power $[dB]$')
    plt.xlabel(r'Delay $[s]$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/delay_power_intensity_average_over_antennae.pdf")
    
    plt.figure()
    plt.plot(linspace[:400], PDP_average_dB[:400], label = r'Mean of PDP of antennas $[dB]$')
    plt.ylabel(r'Power $[dB]$')
    plt.xlabel(r'Delay $[s]$')
    plt.legend()
    #plt.tight_layout()
    plt.savefig("figures/delay_power_intensity_average_over_antennae_400.pdf")



if __name__ == "__main__":
    # Compute with seed for reproducibility
    np.random.seed(1337)
    show_plot()
    np.random.seed(1337)
    plot_Spencer_power_intensity(N_sim = 1000)
    np.random.seed(1337)
    plot_Spencer_PDP_Antenna_average()
