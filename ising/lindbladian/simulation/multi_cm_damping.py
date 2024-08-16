import numpy as np
from itertools import product
import json
from ising.lindbladian.simulation.unraveled import (
    thermal_lindbladians,
    lowering_all_sites,
    lindbladian_operator,
)

from ising.utils import global_phase
from ising.utils.trace import partial_trace
from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization

from tqdm import tqdm


ZERO = np.array([[1], [0]])
ONE = np.array([[0], [1]])
SIGMA_MINUS = np.array([[0, 1], [0, 0]]) # a_dag
SIGMA_PLUS = np.array([[0, 0], [1, 0]]) # a_n

def calculate_gamma(beta):
    return np.exp(-beta) / (1 + np.exp(-beta))


QUBIT_COUNT = 3
GAMMA = 0.4
TIME_RANGE = (0, 5)
TIME_COUNT = 5
EPS = 0.1

H_VAL = -0.1
COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF"]



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

def _kron_multi(ls):
    prod = ls[0]
    for term in ls[1:]:
        prod = np.kron(prod, term)
    return prod

def apply_amplitude_damping(rho_sys, ham, gamma, time, nue=1000):
    """
    Applies the Kraus operator and get the final state, this is applicable for
    multiple qubits. The damping is applied to all the qubits at the same time
    """
    delta_t = time / nue
    num_qubits = int(np.log2(rho_sys.shape[0]))
    new_gamma = 1 - np.exp(-gamma * delta_t)

    e0 = np.array([[1, 0], [0, np.sqrt(1 - new_gamma)]])
    e1 = np.array([[0, np.sqrt(new_gamma)], [0, 0]])

    cur_rho = rho_sys

    eig_val, eig_vec = np.linalg.eig(ham)
    eig_vec_inv = np.linalg.inv(eig_vec)

    krauses = []
    for kraus_ops in product([e0, e1], repeat=num_qubits):
        krauses.append(_kron_multi(kraus_ops))

    for _ in range(nue):
        final_rho = np.zeros_like(cur_rho)
        for kraus in krauses:
            final_rho += (kraus @ cur_rho @ kraus.conj().T)

        cur_rho = (
            matrix_exp(eig_vec, eig_val, eig_vec_inv, time=delta_t)
            @ final_rho
            @ matrix_exp(eig_vec, eig_val, eig_vec_inv, time=-delta_t)
        )

    return cur_rho


def lindblad_evo(rho, ham, gamma, z, time):
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
    l_op = lindbladian_operator(ham, thermal_lindbladians(QUBIT_COUNT, gamma=gamma, z=z))

    eig_val, eig_vec = np.linalg.eig(l_op)
    eig_vec_inv = np.linalg.inv(eig_vec)

    op_time_matrix = matrix_exp_no_i(eig_vec, eig_val, eig_vec_inv, time)
    rho_vec_final = op_time_matrix @ rho_vec
    rho_final = reshape_vec(rho_vec_final)

    return rho_final

def interaction_hamiltonian(QUBIT_COUNT, gamma):
    """
    Construct a `QUBIT_COUNT + 1` Hamiltonian for each interaction point.
    There will be `QUBIT_COUNT` of them, acting on two qubits each

    Input:
        - QUBIT_COUNT: Size of the chain
        - gamma: Strength of the Hamiltonian
    """
    ham_ints = []
    for _site in range(QUBIT_COUNT):
        sys_site, env_site = _site, QUBIT_COUNT

        ham_int1, ham_int2 = None, None
        for pos in range(QUBIT_COUNT+1):
            cur_op1, cur_op2 = None, None
            if pos == sys_site:
                cur_op1, cur_op2 = SIGMA_PLUS, SIGMA_MINUS
            elif pos == env_site:
                cur_op1, cur_op2 = SIGMA_MINUS, SIGMA_PLUS
            else:
                cur_op1, cur_op2 = np.eye(2), np.eye(2)

            if ham_int1 is None or ham_int2 is None:
                ham_int1, ham_int2 = cur_op1, cur_op2
            else:
                ham_int1 = np.kron(ham_int1, cur_op1)
                ham_int2 = np.kron(ham_int2, cur_op2)

        assert ham_int1 is not None and ham_int2 is not None

        ham_ints.append(np.sqrt(gamma) * (ham_int1 + ham_int2))

    return ham_ints


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
    big_ham_sys = np.kron(ham_sys, np.eye(2))

    cur_rho_sys = rho_sys

    ham_ints = interaction_hamiltonian(QUBIT_COUNT, gamma=gamma)

    hams = []
    print(f"Running for time:{time}, neu:{neu}")
    for ham_int in ham_ints:
        ham = (np.sqrt(tau) * big_ham_sys / QUBIT_COUNT) + ham_int
        eig_val, eig_vec = np.linalg.eig(ham)
        eig_vec_inv = np.linalg.inv(eig_vec)
        hams.append((eig_vec, eig_val, eig_vec_inv))
    print("Ham operator calculation complete")

    us, u_dags = [], []
    for eig_vec, eig_val, eig_vec_inv in hams:
        u = matrix_exp(eig_vec, eig_val, eig_vec_inv, time=np.sqrt(tau))
        u_dag = matrix_exp(eig_vec, eig_val, eig_vec_inv, time=-np.sqrt(tau))
        us.append(u)
        u_dags.append(u_dag)

    with tqdm(total=neu) as pbar:
        for _ in range(neu):
            pbar.update(1)
            for u, u_dag in zip(us, u_dags):
                complete_rho = np.kron(cur_rho_sys, rho_env)
                rho_fin = (
                    u @ complete_rho @ u_dag 
                )
                cur_rho_sys = partial_trace(rho_fin, list(range(QUBIT_COUNT, QUBIT_COUNT + 1)))

    return cur_rho_sys


def _random_psi(qubit_count):
    real_psi = np.random.uniform(-1, 1, 2**qubit_count)
    norm = sum([np.abs(x) for x in real_psi]) ** 0.5
    return real_psi / norm


def test_main():
    np.random.seed(42)

    psi = _random_psi(qubit_count=QUBIT_COUNT)
    rho_sys = np.outer(psi, psi.conj())
    # ham = np.zeros_like(rho_sys)
    ham = parametrized_ising(QUBIT_COUNT, H_VAL).matrix

    # Environment qubit is always in ZERO, and it is always only one qubit each
    inv_temp = 10
    alpha, beta = 1, np.exp(-inv_temp) 
    rho_env1 = (alpha * np.outer(ZERO, ZERO) + beta * np.outer(ONE, ONE)) / (alpha + beta)

    observable = overall_magnetization(QUBIT_COUNT).matrix

    z = calculate_gamma(inv_temp)
    gamma = GAMMA

    results = {"interaction": {}, "lindbladian": {}}
    times = np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)
    for time in times:
        neu = max(100, int(10 * (time**2) / EPS))
        rho_ham = ham_evo(rho_sys, rho_env1, ham, gamma, time, neu)
        rho_lin = lindblad_evo(rho_sys, ham, gamma, z, time)

        rho_ham = rho_ham / global_phase(rho_ham)
        rho_lin = rho_lin / global_phase(rho_lin)

        results["interaction"][time] = np.trace(np.abs(observable @ rho_ham))
        results["lindbladian"][time] = np.trace(np.abs(observable @ rho_lin))

    file_name = f"data/lindbladian/time_vs_magn/size_{QUBIT_COUNT}.json"
    with open(file_name, "w") as file:
        json.dump(results, file)

    print(f"saved the data to {file_name}")


if __name__ == "__main__":
    test_main()
