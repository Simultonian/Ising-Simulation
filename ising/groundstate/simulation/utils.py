import numpy as np


def spectral_norm(eigenval):
    """
    ||O|| = max eigenvalue of (O @ O^T), can be rewritten as the following
    """
    max_eigval = max(np.abs(eigenval**2)) ** 0.5
    return max_eigval


def calculate_mu(mu_samples, count, coeffs):
    """
    Formula is in the algorithm 1 of the paper
    """
    norm_1_sq = np.linalg.norm(coeffs, ord=1) ** 2
    mu_sum = sum(mu_samples)

    mu = (norm_1_sq * mu_sum) / count
    return mu


def ground_state_constants(spectral_gap, eeta, eps, prob, obs_norm):
    """
    Calculating the parameters for the LCU based on the given constants.

    Values calculated:
        - t: Base time for evolution
        - count: Number of iterations (T in the paper)
        - gamma: Error gap between LCU and exact function
        - delta_t: Time factor (refer paper)
        - m: M from paper, the max value of j in the loop

    Calculations:
        - t = (1/delta^2) * log(||O|| / eeta * eps)

        - count = (||O||^2 ln(2/delta)) / (eps^2 * eeta^4)

        - gamma = eps * eeta^2 / 12 ||O||

        - delta_t = 1 / (sqrt(2t) + sqrt(2log(5/gamma)))
    """

    t = (1 / spectral_gap**2) * np.log2(1 / (eeta * eps))
    count = int(((obs_norm**2) * np.log(2 / prob)) / (eps**2 * eeta**4))

    gamma = (eps * (eeta**2)) / (12 * obs_norm)
    m = int(
        np.ceil(
            np.sqrt(2)
            * (np.sqrt(t) + np.sqrt(np.log2(5 / gamma)))
            * np.sqrt(np.log2(4 / gamma))
        )
    )
    delta_t = 1 / (np.sqrt(2 * t) + np.sqrt(2 * np.log2(5 / gamma)))
    return {"t": t, "count": count, "gamma": gamma, "m": m, "delta_t": delta_t}


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
