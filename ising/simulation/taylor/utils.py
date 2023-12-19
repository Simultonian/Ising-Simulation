import numpy as np
import math


def calculate_mu(mu_samples, count, coeffs):
    """
    Formula is in the algorithm 1 of the paper
    """
    norm_1_sq = np.linalg.norm(coeffs, ord=1) ** 2
    mu_sum = sum(mu_samples)

    mu = (norm_1_sq * mu_sum) / count
    return mu


def get_small_k_probs(t_bar, r, cap_k):
    ks = np.arange(cap_k + 1)
    k_vec = np.zeros(cap_k + 1, dtype=np.complex128)

    def apply_k(k):
        # Function according to the formula
        t1 = ((1j * t_bar / r) ** k) / math.factorial(k)
        t2 = np.sqrt(1 + ((t_bar / (r * (k + 1))) ** 2))
        return t1 * t2

    vectorized_apply_k = np.vectorize(apply_k)

    k_vec = vectorized_apply_k(ks)

    # Odd positions are 0
    k_vec[1::2] = 0
    return k_vec


def get_cap_k(t_bar, obs_norm, eps) -> int:
    t_bar = abs(t_bar)
    numr = np.log(t_bar * obs_norm / eps)
    return int(np.ceil(numr / np.log(numr)))


def calculate_exp(time, pauli, k):
    eye = np.identity(pauli.shape[0])
    dr = np.sqrt(1 + ((time) / (k + 1)) ** 2)

    term2 = (1j * (time) * pauli) / (k + 1)
    rotate = (eye - term2) / dr
    return rotate


def get_alphas(t_bar, cap_k, r):
    return get_small_k_probs(t_bar=t_bar, r=r, cap_k=cap_k)
