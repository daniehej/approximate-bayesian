#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply the ABC method on the Spencer model

@author: grp2
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from Spencer_model import (
    linspace, K, M,
    Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray, var_laplace,
    simulate_Spencer_ULA
    )

from Summary_statistics import summary_ours

# =============================================================================
# Setup
# =============================================================================
k_sim = 200
N_params = 6
epsilon = 1.9

# Summary statistic normalization constant determined in initial experiment.
norm_const = np.array([
    6.78520543e-17, 4.58003300e-25, 2.82629765e-32, 5.65219443e-24,
    7.10027033e-33, 8.67177484e-41, 6.98593201e-58, 2.11056876e-50,
    2.22541388e-43, 6.74366516e-35, 7.26755265e-51, 1.10019662e-64,
    1.33932122e-74, 6.27343303e-66, 6.42069771e-58, 9.98752113e-49,
    2.33662505e-66, 4.60618325e-82, 5.35248151e+00, 9.04229173e+00,
    1.68259790e-11, 4.88389380e-20, 2.42792049e+00, 2.03152692e+01,
    ])

laplace_var_max = np.deg2rad(30)

# =============================================================================
# Functions
# =============================================================================
def rejection_abc(data_obs, k_accept, epsilon, prior, model, summary, distance):
    """
    Performs the rejection ABC method on the observed data with the given
    functions and parameters.

    Parameters
    ----------
    data_obs : complex nd-array
        The observed data which the ABC method should be applied to.
    k_accept : integer
        Amount of samples to accept.
    epsilon : float
        Tolerance.
    prior : function()
        Draws parameter vector from prior range.
    model : function(*theta)
        Simulates data with same shape as data_obs.
    summary : function(data)
        Calculates summary statistics.
    distance : function(s, s_obs)
        Distance between summary vectors.

    Returns
    -------
    data_result : complex nd array with shape (k_accept x data_obs.shape)
        Array of k_accept realisations of data.
    theta_result : nd array with shape (k_accept x 6)
        Array of accepted parameter vectors.
    N_iters : integer
        Amount of inner iterations.

    """
    s_obs = summary(data_obs)
    s_obs_norm = s_obs/norm_const
    theta_result = np.zeros((k_accept, 6))
    data_result = np.zeros((k_accept, *data_obs.shape), dtype=np.complex)
    dist_result = np.zeros(k_accept)
    i = 0
    N_iters = 0

    while i < k_accept and N_iters < 1000000:
        theta_sim = prior()
        data_sim = model(*theta_sim)
        s_sim_norm = summary(data_sim)/norm_const
        dist_sim = distance(s_obs_norm, s_sim_norm)
        print(".", end="")
        if dist_sim < epsilon:
            theta_result[i] = theta_sim
            data_result[i] = data_sim
            dist_result[i] = dist_sim
            i += 1
            print("")
            print(f"accepted {i}")
            print(theta_sim)
        N_iters += 1

    return data_result, theta_result, N_iters


def prior_spencer():
    """
    Prior range for the Spencer model

    """
    return np.array([np.random.uniform(1e-9, 1e-7), # Q
                     np.random.uniform(5e6, 1e8),   # Lambda
                     np.random.uniform(5e6, 4e9),   # lambda
                     np.random.uniform(0, 1e-7),    # Gamma
                     np.random.uniform(0, 1e-7),    # gamma
                     np.random.uniform(0.01, laplace_var_max),
                     ])


def prior_linspace(N_prior):
    """
    Returns N_prior uniformly spaced values over the prior range.

    """
    return np.array([np.linspace(1e-9, 1e-7, N_prior), # Q
                     np.linspace(5e6, 1e8, N_prior),   # Lambda
                     np.linspace(5e6, 4e9, N_prior),   # lambda
                     np.linspace(0, 1e-7, N_prior),    # Gamma
                     np.linspace(0, 1e-7, N_prior),    # gamma
                     np.linspace(0.01, laplace_var_max, N_prior),  # var_laplace
                     ])


def distance_euclidean(s, s_sim):
    """
    Calculates Euclidean distance

    """
    return np.linalg.norm(s-s_sim)


def kde(theta):
    """
    Calculates KDE and MMSE estimate of the parameters.

    """
    kdes = [stats.gaussian_kde(theta.T[i]) for i in range(theta.shape[1])]
    mmses = [np.mean(theta[:,i]) for i in range(theta.shape[1])]
    return kdes, mmses

# =============================================================================
# Main script
# =============================================================================
if __name__ == "__main__":
    # Define true parameter
    theta_true = [Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray, var_laplace]
    np.random.seed(1337)  # Compute with seed for reproducibility

    data_obs = simulate_Spencer_ULA(Q, Lambda_cluster, lambda_ray, Gamma_cluster, gamma_ray, var_laplace)
    data, theta_abc, N_iters = rejection_abc(data_obs, k_sim, epsilon, prior_spencer, simulate_Spencer_ULA, summary_ours, distance_euclidean)
    kdes, mmses = kde(theta_abc)
    
    print(f"number of iterations: {N_iters}")
    print(f"accept rate: {k_sim/N_iters*100:.4}%")
    
    # =========================================================================
    # Plot
    # =========================================================================
    linspaces = prior_linspace(100)
    labels = [r"$Q$", r"$\Lambda$", r"$\lambda$", r"$\Gamma$", r"$\gamma$", r"$\sigma$"]

    fig, axs = plt.subplots(1, 6, figsize=(12.5,2.5))
    for i in range(5):
        axs[i].plot(linspaces[i], kdes[i](linspaces[i]))
        axs[i].axvline(theta_true[i], color='g', linestyle='--', label="True value")
        axs[i].axvline(mmses[i], color='k', label="MMSE estimate")
        axs[i].tick_params(labelleft=False)
        axs[i].set_xlabel(labels[i])
    for i in [5]:
        axs[i].plot(np.rad2deg(linspaces[i]), kdes[i](linspaces[i]))
        axs[i].axvline(np.rad2deg(theta_true[i]), color='g', linestyle='--', label="True value")
        axs[i].axvline(np.rad2deg(mmses[i]), color='k', label="MMSE estimate")
        axs[i].tick_params(labelleft=False)
        axs[i].legend(loc="lower center")
        axs[i].set_xlabel(labels[i])
    plt.subplots_adjust(wspace=0.08)
    plt.savefig(f"figures/abc_{k_sim}_{epsilon}.pdf", bbox_inches='tight')
    plt.show()
    sds = np.std(theta_abc, axis=0)
    
    # =========================================================================
    # Save values to file
    # =========================================================================
    with open(f"figures/abc_{k_sim}_{epsilon}.txt", 'w') as f:
        f.write("mmse:\n")
        for mmse in mmses: f.write(f"{mmse:.4E}, ")
        f.write("\nsd:\n")
        for sd in sds: f.write(f"{sd:.4E}, ")
    
    print(mmses)
    print(sds)
