import numpy as np

def calculate_mu(mu_samples, count, coeffs):
    """
    Formula is in the algorithm 1 of the paper
    """
    norm_1_sq = np.linalg.norm(coeffs, ord=1) ** 2
    mu_sum = sum(mu_samples)

    mu = (norm_1_sq * mu_sum) / count
    return mu

def calculate_lcu_constants(m, delta_t, t):
    """
    Coefficients are calculated by:
        c_j = (delta_t / sqrt(2pi)) * e^{-j^2 delta_t^2 / 2}

    The overall time value for given j is calculated by:
        t_j = j * delta_t * sqrt(2*t)
    """
    # from -M to M
    js = np.arange(-m, m + 1)
    coeffs = (delta_t / np.sqrt(2 * np.pi)) * np.exp(-1 * js**2 * (delta_t**2) / 2)
    times = js * delta_t * np.sqrt(2 * t)

    return coeffs, times
