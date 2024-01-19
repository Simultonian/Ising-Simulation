import numpy as np
from ising.hamiltonian import Hamiltonian

"""
Gate complexity for preparing groundstate for `ham` with `eps` error given
the initial state has `eeta` overlap with the same.
"""


def ge_et_al(ham: Hamiltonian, eeta: float, eps: float, obs_norm: float = 1) -> int:
    """
    Ge et al Method
    ---
    Formula: beta * l / (eeta * delta)
    """
    l = len(ham.paulis)
    beta = np.sum(np.abs(ham.coeffs))
    delta = ham.approx_spectral_gap

    nmr = beta * l
    dr = eeta * delta
    return np.ceil(nmr / dr).astype(int)


def phase_estimation(
    ham: Hamiltonian, eeta: float, eps: float, obs_norm: float = 1
) -> int:
    """
    Phase Estimation and Amplitude Amplification Method
    ---
    Formula: beta * l / eeta^2 * delta * eps
    """
    l = len(ham.paulis)
    beta = np.sum(np.abs(ham.coeffs))
    delta = ham.approx_spectral_gap

    nmr = beta * l
    dr = (eeta**2) * delta * eps
    return np.ceil(nmr / dr).astype(int)


def poulin_et_al(ham: Hamiltonian, eeta: float, eps: float, obs_norm: float = 1) -> int:
    """
    Poulin and Wocjan Filtering
    ---
    Formula: beta * l / eeta * delta
    """
    l = len(ham.paulis)
    beta = np.sum(np.abs(ham.coeffs))
    delta = ham.approx_spectral_gap

    nmr = beta * l
    dr = eeta * delta
    return np.ceil(nmr / dr).astype(int)


def lin_et_al(ham: Hamiltonian, eeta: float, eps: float, obs_norm: float = 1) -> int:
    """
    Lin and Tong: Near Optimal Groundstate Preparation
    ---
    Formula: log(1 / eps) * beta * l / eeta * delta
    """
    l = len(ham.paulis)
    beta = np.sum(np.abs(ham.coeffs))
    delta = ham.approx_spectral_gap

    nmr = beta * l * np.log10(1 / eps)
    dr = eeta * delta
    return np.ceil(nmr / dr).astype(int)


def truncated_taylor(
    ham: Hamiltonian, eeta: float, eps: float, obs_norm: float = 1
) -> int:
    """
    Truncated Taylor Series with single-ancilla LCU
    """
    beta = np.sum(np.abs(ham.coeffs))
    delta = ham.approx_spectral_gap

    nmr = (beta**2) * np.log10(obs_norm / (eps * eeta))
    dr = delta**2
    return np.ceil(nmr / dr).astype(int)


def first_order_trotter(
    ham: Hamiltonian, eeta: float, eps: float, obs_norm: float = 1
) -> int:
    """
    Hamiltonian Simulation using First Order Trotter with single-ancilla LCU
    """
    l = len(ham.paulis)
    lambd = np.max(np.abs(ham.coeffs))
    delta = ham.approx_spectral_gap

    nmr = obs_norm * (lambd**2) * (l**3) * np.log10(obs_norm / (eeta * eps))
    dr = eps * eeta**2 * delta**2
    return np.ceil(nmr / dr).astype(int)


def qdrift(ham: Hamiltonian, eeta: float, eps: float, obs_norm: float = 1) -> int:
    """
    Hamiltonian Simulation using qDRIFT with single-ancilla LCU
    """
    beta = np.sum(np.abs(ham.coeffs))
    delta = ham.approx_spectral_gap

    nmr = obs_norm * (beta**2) * np.log10(obs_norm / (eeta * eps))
    dr = eps * eeta**2 * delta**2
    return np.ceil(nmr / dr).astype(int)
