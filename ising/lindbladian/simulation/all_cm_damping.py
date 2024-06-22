import numpy as np
import json
from ising.hamiltonian import parametrized_ising
from ising.utils import close_state
from ising.observables import overall_magnetization
from ising.lindbladian.simulation.unraveled import (
    lowering_all_sites,
    lindbladian_operator,
)

from ising.utils.trace import partial_trace

# log scale
GAMMA_RANGE = (0, -4)
GAMMA_COUNT = 4

# not log scale
TIME_RANGE = (1, 5)
TIME_COUNT = 10

# system params
CHAIN_SIZE = 5
H_VAL = -0.1

# simulation params
OVERLAP = 0.8

ZERO = np.array([[1], [0]])
ONE = np.array([[0], [1]])
SIGMA_MINUS = np.array([[0, 1], [0, 0]])
SIGMA_PLUS = np.array([[0, 0], [1, 0]])


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


def ham_evo(rho_sys, rho_env, ham_sys, gamma, time):
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
    big_ham_sys = np.kron(ham_sys, np.eye(rho_env.shape[0]))

    # interaction Hamiltonian directly from the formula
    g = gamma
    ham_int = g * (np.kron(SIGMA_PLUS, SIGMA_MINUS) + np.kron(SIGMA_MINUS, SIGMA_PLUS))

    if ham_int.shape != big_ham_sys.shape:
        raise ValueError("Incorrect size: {ham_int.shape}, {big_ham_sys.shape}")

    complete_ham = big_ham_sys + ham_int
    complete_rho = np.kron(rho_sys, rho_env)

    eig_val, eig_vec = np.linalg.eig(complete_ham)
    eig_vec_inv = np.linalg.inv(eig_vec)


    rho_fin = (
        matrix_exp(eig_vec, eig_val, eig_vec_inv, time)
        @ complete_rho
        @ matrix_exp(eig_vec, eig_val, eig_vec_inv, -time)
    )

    rho_sys_fin = partial_trace(rho_fin, [1])

    return rho_sys_fin



def test_main():
    rho_sys = np.outer(ONE, ONE.conj())
    ham = np.zeros_like(rho_sys)

    rho_env = np.outer(ZERO, ZERO.conj())

    gamma = 0.2

    rho_linds = [lindblad_evo(rho_sys, ham, gamma, time) for time in [0, 0.5, 1, 1.5]]
    rho_hams = [ham_evo(rho_sys, rho_env, ham, gamma, time) for time in [0, 0.5, 1, 1.5]]

    for rho_lin, rho_ham in zip(rho_linds, rho_hams):
        np.testing.assert_almost_equal(rho_lin, rho_ham)

    
if __name__ == "__main__":
    test_main()
