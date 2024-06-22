import numpy as np
import json
from ising.lindbladian.simulation.unraveled import (
    lowering_all_sites,
    lindbladian_operator,
)

from ising.utils import global_phase
from ising.utils.trace import partial_trace

ZERO = np.array([[1], [0]])
ONE = np.array([[0], [1]])
SIGMA_MINUS = np.array([[0, 1], [0, 0]])
SIGMA_PLUS = np.array([[0, 0], [1, 0]])


GAMMAS = [0, 0.1, 0.4, 0.7, 0.9]
TIMES = [0, 0.5, 1, 1.5, 2]
EPS = 0.01


def _round(mat):
    return np.round(mat, decimals=3)


def matrix_exp_no_i(eig_vec, eig_val, eig_vec_inv, time: float):
    return eig_vec @ np.diag(np.exp(time * eig_val)) @ eig_vec_inv


def matrix_exp(eig_vec, eig_val, eig_vec_inv, time: float):
    return eig_vec @ np.diag(np.exp(complex(0, -1) * time * eig_val)) @ eig_vec_inv


def reshape_vec(vec):
    l = vec.shape[0]
    assert vec.shape[1] == 1
    d = int(np.sqrt(l))

    rho = np.zeros((d, d), dtype=complex)

    for i, x in enumerate(vec):
        r, c = i % d, i // d
        rho[r][c] = x[0]

    return rho


def apply_amplitude_damping(rho, gamma, time):
    """
    Applies the Kraus operator and get the final state
    """
    new_gamma = 1 - np.exp(-gamma * time)
    e0 = np.array([[1, 0], [0, np.sqrt(1 - new_gamma)]])
    e1 = np.array([[0, np.sqrt(new_gamma)], [0, 0]])

    return (e0 @ rho @ e0.conj().T) + (e1 @ rho @ e1.conj().T)


def lindblad_evo(rho, ham, gamma, time):
    """
    Function to calculate final state after amplitude damping.

    Inputs:
        - rho: Starting state of system
        - ham: System Hamiltonian
        - gamma: Strength of amplitude damping
        - time: evolution time
    """
    # columnize
    rho_vec = rho.reshape(-1, 1)

    # Hamiltonian is zero
    l_op = lindbladian_operator(ham, [np.sqrt(gamma) * SIGMA_MINUS])

    eig_val, eig_vec = np.linalg.eig(l_op)
    eig_vec_inv = np.linalg.inv(eig_vec)

    op_time_matrix = matrix_exp_no_i(eig_vec, eig_val, eig_vec_inv, time)
    rho_vec_final = op_time_matrix @ rho_vec
    rho_final = reshape_vec(rho_vec_final)

    return rho_final


def ham_evo(rho_sys, rho_env, ham_sys, gamma, time, neu=1000):
    """
    Replicate the Lindbladian evolution of amplitude damping using
    interaction Hamiltonian dynamics.

    Inputs:
        - rho_sys: Initial state of system only
        - rho_env: Initial state of environment only
        - ham_sys: Hamiltonian for system, same size as `rho_sys`
        - gamma: Strength of amplitude damping
        - time: Evolution time to match
    """
    tau = time / neu
    big_ham_sys = np.kron(ham_sys, np.eye(rho_env.shape[0]))

    # interaction Hamiltonian directly from the formula
    ham_int_norm = np.sqrt(gamma)
    ham_int = ham_int_norm * (
        np.kron(SIGMA_PLUS, SIGMA_MINUS) + np.kron(SIGMA_MINUS, SIGMA_PLUS)
    )

    if ham_int.shape != big_ham_sys.shape:
        raise ValueError("Incorrect size: {ham_int.shape}, {big_ham_sys.shape}")

    complete_ham = big_ham_sys + ham_int

    cur_rho_sys = rho_sys
    eig_val, eig_vec = np.linalg.eig(complete_ham)
    eig_vec_inv = np.linalg.inv(eig_vec)

    for _ in range(neu):
        complete_rho = np.kron(cur_rho_sys, rho_env)

        rho_fin = (
            matrix_exp(eig_vec, eig_val, eig_vec_inv, time=np.sqrt(tau))
            @ complete_rho
            @ matrix_exp(eig_vec, eig_val, eig_vec_inv, time=-np.sqrt(tau))
        )
        cur_rho_sys = partial_trace(rho_fin, [1])

    return cur_rho_sys


def _random_psi():
    real_psi = np.random.uniform(-1, 1, 2)
    x = 0
    for a in real_psi:
        x += np.abs(a) ** 2

    real_psi /= np.sqrt(x)
    return real_psi


def test_main():
    np.random.seed(42)

    psi = _random_psi()
    rho_sys = np.outer(psi, psi.conj())
    rho_sys = rho_sys / global_phase(rho_sys)
    ham = np.zeros_like(rho_sys)

    rho_env = np.outer(ZERO, ZERO.conj())

    for gamma in GAMMAS:
        for time in TIMES:
            neu = max(100, int(100 * (time**2) / EPS))
            rho_ham = ham_evo(rho_sys, rho_env, ham, gamma, time, neu)
            rho_amp = apply_amplitude_damping(rho_sys, gamma, time)
            rho_lin = lindblad_evo(rho_sys, ham, gamma, time)

            rho_lin = rho_lin / global_phase(rho_lin)
            rho_amp = rho_amp / global_phase(rho_amp)
            rho_ham = rho_ham / global_phase(rho_ham)

            np.testing.assert_almost_equal(rho_lin, rho_amp, decimal=4)
            np.testing.assert_almost_equal(rho_lin, rho_ham, decimal=4)


if __name__ == "__main__":
    test_main()
