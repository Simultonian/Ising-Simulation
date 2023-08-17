import numpy as np
import numpy.testing as npt
from functools import lru_cache
from collections import Counter

from qiskit.quantum_info import SparsePauliOp

from ising.utils import close_state


def calculate_mu(mu_samples, count, coeffs):
    """
    Formula is in the algorithm 1 of the paper
    """
    norm_1_sq = np.linalg.norm(coeffs, ord=1) ** 2
    mu_sum = sum(mu_samples)

    mu = (norm_1_sq * mu_sum) / count
    return mu
