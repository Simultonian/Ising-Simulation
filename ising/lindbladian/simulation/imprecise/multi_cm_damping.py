import numpy as np
from itertools import product
import json
from qiskit.quantum_info import Pauli
from ising.lindbladian.simulation.unraveled import (
    lowering_all_sites,
    lindbladian_operator,
)

from ising.lindbladian.simulation.imprecise import collision_model_evo
from ising.lindbladian.simulation.utils import load_interaction_hams

from ising.utils import global_phase
from ising.utils.trace import partial_trace
from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization
from ising.lindbladian.simulation.multi_cm_damping import lindblad_evo, interaction_hamiltonian

from tqdm import tqdm


ZERO = np.array([[1], [0]])
ONE = np.array([[0], [1]])
SIGMA_MINUS = np.array([[0, 1], [0, 0]])
SIGMA_PLUS = np.array([[0, 0], [1, 0]])


QUBIT_COUNT = 3
GAMMAS = [0, 0.1, 0.4, 0.7, 0.9]
GAMMA = 0
TIME_RANGE = (1, 2)
TIME_COUNT = 10
EPS = 0.1
DELTA = 0.9

SAL_RUNS = int(
            8
            * np.ceil(
                ((np.log(2 / (1 - DELTA)))) / (EPS ** 2)
            )
        )

SAL_RUNS = 100
H_VAL = -0.1
COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF"]



def _round(mat):
    return np.round(mat, decimals=3)


def matrix_exp(eig_vec, eig_val, eig_vec_inv, time: float):
    return eig_vec @ np.diag(np.exp(complex(0, -1) * time * eig_val)) @ eig_vec_inv


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

    return cur_rho_sys


def taylor_evo(rho_sys, rho_env, observable, gamma, time, error, neu=10):
    """
    Replicate the Lindbladian evolution of amplitude damping using
    interaction Hamiltonian dynamics.

    Inputs:
        - rho_sys: Initial state of system only
        - rho_env: Initial state of environment only
        - gamma: Strength of amplitude damping
        - time: Evolution time to match
    """
    pre_gamma_ham_ints = load_interaction_hams(QUBIT_COUNT)
    ham_ints = []

    for sparse_repr in pre_gamma_ham_ints:
        paulis, coeffs = sparse_repr.paulis, sparse_repr.coeffs
        coeffs = [gamma * coeff for coeff in coeffs]
        ham_ints.append((paulis, coeffs))

    ham_sys = parametrized_ising(QUBIT_COUNT, H_VAL)
    pauli_sys, coeff_sys = ham_sys.paulis, ham_sys.coeffs

    # single environment qubit
    pauli_sys = [Pauli(pauli.to_label() + "I") for pauli in pauli_sys]

    magn_h = collision_model_evo(
        rho_sys = rho_sys,
        rho_env = rho_env,
        ham_ints = ham_ints,
        ham_sys = (pauli_sys, coeff_sys),
        tau = tau,
        error = error,
        neu = neu,
        runs = SAL_RUNS,
        observable = observable,
    )


    return magn_h 

def _random_psi(qubit_count):
    real_psi = np.random.uniform(-1, 1, 2**qubit_count)
    norm = sum([np.abs(x) for x in real_psi]) ** 0.5
    return real_psi / norm


def test_main():
    np.random.seed(42)
    # results = {"interaction": {}, "lindbladian": {}, "sal": {}}
    results = {"interaction": {}, "lindbladian": {}}

    gamma = GAMMA
    times = np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)

    psi = _random_psi(qubit_count=QUBIT_COUNT)
    rho_sys = np.outer(psi, psi.conj())

    # Environment qubit is always in ZERO, and it is always only one qubit each
    rho_env = np.outer(ZERO, ZERO.conj())

    # QUBIT_COUNT size Hamiltonians
    ham_sys = parametrized_ising(QUBIT_COUNT, H_VAL)

    # QUBIT_COUNT + 1 size Hamiltonians
    ham_ints = interaction_hamiltonian(QUBIT_COUNT, gamma=1)

    big_ham_sys = np.kron(ham_sys.matrix, np.eye(2))
    observable = overall_magnetization(QUBIT_COUNT)

    for time in times:
        neu = max(10, int(10 * (time**2) / EPS))
        tau = time / neu

        print(f"Running for time:{time}, neu:{neu}")
        us, u_dags = [], []
        for ham_int in ham_ints:
            ham = (np.sqrt(tau) * big_ham_sys / QUBIT_COUNT) + (GAMMA * ham_int)
            # Also decompose ham into (paulis, coeffs)
            eig_val, eig_vec = np.linalg.eig(ham)
            eig_vec_inv = np.linalg.inv(eig_vec)

            u = matrix_exp(eig_vec, eig_val, eig_vec_inv, time=np.sqrt(tau))
            u_dag = matrix_exp(eig_vec, eig_val, eig_vec_inv, time=-np.sqrt(tau))
            us.append(u)
            u_dags.append(u_dag)
        print("Ham operator calculation complete")

        cur_rho_sys = rho_sys.copy()
        with tqdm(total=neu) as pbar:
            for _ in range(neu):
                pbar.update(1)
                for u, u_dag in zip(us, u_dags):
                    complete_rho = np.kron(cur_rho_sys, rho_env)
                    rho_fin = (
                        u @ complete_rho @ u_dag 
                    )
                    cur_rho_sys = partial_trace(rho_fin, list(range(QUBIT_COUNT, QUBIT_COUNT + 1)))

        results["interaction"][time] = np.trace(np.abs(observable.matrix @ cur_rho_sys))


    for time in times:
        rho_lin = lindblad_evo(rho_sys, ham_sys.matrix, gamma, time)
        rho_lin = rho_lin / global_phase(rho_lin)
        results["lindbladian"][time] = np.trace(np.abs(observable.matrix @ rho_lin))

    file_name = f"data/lindbladian/time_vs_magn/size_{QUBIT_COUNT}.json"
    with open(file_name, "w") as file:
        json.dump(results, file)

    print(f"saved the data to {file_name}")


if __name__ == "__main__":
    test_main()
