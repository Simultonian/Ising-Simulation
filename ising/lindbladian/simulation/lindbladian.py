import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from ising.utils import global_phase, hache
from ising.utils.trace import partial_trace
from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization

from tqdm import tqdm
import json
import hashlib


ZERO = np.array([[1], [0]])
ONE = np.array([[0], [1]])
SIGMA_MINUS = np.array([[0, 1], [0, 0]]) # a_dag, lowering
SIGMA_PLUS = np.array([[0, 0], [1, 0]]) # a_n, raising

def calculate_gamma(beta):
    return np.exp(-beta) / (1 + np.exp(-beta))


"""
Hamiltonian constants
"""
QUBIT_COUNT = 4
H_VAL = -0.1

"""
Environment constants
"""
LAMBDAS = [0.0005, 0.001, 0.01, 0.1, 1, 0.5, 0.1, 0.8, 10.0]
INV_TEMPS = [0.1, 0.5, 1, 5, 10]
# PARTIAL_SWAPS = [0, 0.1, 0.5, 0.9, 0.99, 1.0]

"""
Simulation constants
"""
TIME_RANGE = (0, 5)
TIME_COUNT = 5
EPS = 0.1


COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF", 
          "#3ABEFF", "#FFB743", "#00CC99", "#FF6B6B", "#845EC2", 
          "#F9F871", "#00C9A7", "#C34A36", "#4ECDC4", "#FF9671", 
          "#FFC75F", "#008080", "#D65DB1", "#4D8076", "#FF8066"]
DIR = "plots/lindbladian/new_simulation"

def _is_valid_rho(rho):
    assert np.allclose(np.sum(np.diag(rho)), 1)
    assert np.all(np.diag(rho) >= 0)
    assert not np.isnan(rho).any()

def make_valid_rho(rho):
    # Round to 6 decimal places
    rho = np.round(rho, decimals=6)
    
    # Normalize the matrix
    row_sums = np.sum(np.diag(rho))
    rho_normalized = rho / row_sums
    
    # Round again to ensure precision after normalization
    rho_normalized = np.round(rho_normalized, decimals=6)

    for ind, d in enumerate(np.diag(rho_normalized)):
        if d < 0:
            if abs(d) < 1e-3:
                rho_normalized[ind][ind] = abs(d)
            
    # Round again to ensure precision after normalization
    rho_normalized = np.round(rho_normalized, decimals=6)

    row_sums = np.sum(np.diag(rho_normalized))
    rho_normalized = rho_normalized / row_sums

    _is_valid_rho(rho_normalized)
    
    
    return rho_normalized

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

def environment_state(inverse_temperature):
    beta = np.exp(-inverse_temperature) 
    rho_env = (1 * np.outer(ZERO, ZERO) + beta * np.outer(ONE, ONE)) / (1 + beta)
    return make_valid_rho(rho_env)


def thermal_jump_operators(system_size: int, coupling_parameter: float, occupation_factor: float):
    """
    Returns the list for lindbladian operators for the lowering operator
    on all sites.
    """

    l_ops = []
    for site in range(system_size):
        l, r = site, system_size - (site + 1)

        if l == 0:
            op = SIGMA_MINUS.copy()
        else:
            op = np.kron(np.eye(2**l), SIGMA_MINUS)

        if r > 0:
            op = np.kron(op, np.eye(2**r))

        l_ops.append(coupling_parameter * np.sqrt(1 - occupation_factor) * op)

    for site in range(system_size):
        l, r = site, system_size - (site + 1)

        if l == 0:
            op = SIGMA_PLUS.copy()
        else:
            op = np.kron(np.eye(2**l), SIGMA_PLUS)

        if r > 0:
            op = np.kron(op, np.eye(2**r))

        l_ops.append(coupling_parameter * np.sqrt(occupation_factor) * op)

    return l_ops


def lindbladian_operator(system_hamiltonian, jump_operators):
    """
    Gives back the Lindbladian operator for the given system Hamiltonian and
    the Lindbladian operators
    """

    identity = np.eye(system_hamiltonian.shape[0])

    term1 = -1j * np.kron(identity, system_hamiltonian)

    # Transpose of Ising Chain does not make any change
    term2 = 1j * np.kron(system_hamiltonian.T, identity)

    term3 = np.zeros_like(term1)

    for l_op in jump_operators:
        t1 = np.kron(l_op.conj(), l_op)
        t2 = np.kron(identity, (l_op.T.conj() @ l_op))
        t3 = np.kron(l_op.T @ l_op.conj(), identity)

        term3 += t1 - 0.5 * (t2 + t3)

    return term1 + term2 + term3


@hache(blob_type=float, max_size=1000)
def lindbladian_simulation(
    system_size,
    time,
    coupling_parameter,
    inverse_temperature,
    rho_system,
    system_hamiltonian,
    observable
    ):
    rho_system_reshape = rho_system.reshape(-1, 1)

    occupation_factor = np.exp(-inverse_temperature) / (1 + np.exp(-inverse_temperature))

    # Hamiltonian is zero
    l_op = lindbladian_operator(system_hamiltonian, 
                thermal_jump_operators(
                    system_size=system_size,
                    coupling_parameter=coupling_parameter,
                    occupation_factor=occupation_factor))

    eig_val, eig_vec = np.linalg.eig(l_op)
    eig_vec_inv = np.linalg.inv(eig_vec)

    op_time_matrix = matrix_exp_no_i(eig_vec, eig_val, eig_vec_inv, time)
    rho_vec_final = op_time_matrix @ rho_system_reshape
    rho_final = reshape_vec(rho_vec_final)

    rho_lin = np.round(rho_final, decimals=6)
    rho_lin = rho_lin / global_phase(rho_lin)

    return np.trace(np.abs(observable @ rho_lin))


def get_interaction_hamiltonians(system_size, coupling_parameter):
    """
    Construct a `system_size + 1` Hamiltonian for each interaction point.
    There will be `system_size` of them, acting on two qubits each

    Input:
        - system_size: System size
        - coupling_parameter: Strength of the Hamiltonian
    """
    ham_ints = []
    for _site in range(system_size):
        sys_site, env_site = _site, system_size

        ham_int1, ham_int2 = None, None
        for pos in range(system_size+1):
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

        ham_ints.append(coupling_parameter * (ham_int1 + ham_int2))

    return ham_ints


@hache(blob_type=float, max_size=1000)
def collision_simulation(
    system_size,
    rho_system,
    rho_environment,
    delta_t,
    nu_prime,
    observable,
    system_hamiltonian,
    coupling_parameter
):
    interaction_hamiltonians = get_interaction_hamiltonians(system_size, coupling_parameter)
    interaction_count = len(interaction_hamiltonians)
    system_hamiltonian_extended = np.kron(system_hamiltonian, np.eye(2))

    print(f"Running for delta time:{delta_t}, neu:{nu_prime}")

    us = []
    for interaction_hamiltonian in interaction_hamiltonians:
        ham = (delta_t * system_hamiltonian_extended / interaction_count) + interaction_hamiltonian
        eig_val, eig_vec = np.linalg.eig(ham)
        eig_vec_inv = np.linalg.inv(eig_vec)

        u = matrix_exp(eig_vec, eig_val, eig_vec_inv, time=delta_t)
        us.append(u)

    cur_rho_sys = rho_system
    with tqdm(total=nu_prime) as pbar:
        for _ in range(nu_prime):
            pbar.update(1)
            for u in us:
                complete_rho = np.kron(cur_rho_sys, rho_environment)
                rho_fin = (
                    u @ complete_rho @ u.conj().T
                )
                cur_rho_sys = partial_trace(rho_fin, list(range(system_size, system_size + 1)))

    rho_ham = np.round(cur_rho_sys, decimals=6)
    rho_ham = rho_ham / global_phase(rho_ham)

    return np.trace(np.abs(observable @ rho_ham))

import time as time_lib
import os

def test_main():
    np.random.seed(42)

    random_hash = hashlib.md5(str(time_lib.time()).encode()).hexdigest()[:8]
    os.makedirs(f"{DIR}/{random_hash}/", exist_ok=True)

    system_hamiltonian = parametrized_ising(QUBIT_COUNT, H_VAL).matrix

    # 
    psi = np.zeros(2 ** QUBIT_COUNT)
    psi[0] = 1
    rho_system = np.outer(psi, psi.conj())
    observable = overall_magnetization(QUBIT_COUNT).matrix

    times = np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)
    t_max = max(times)

    results = {}
    results["times"] = times.tolist()
    plt.clf()
    for inverse_temperature in INV_TEMPS:
        run_hash = hashlib.md5(str(time_lib.time()).encode()).hexdigest()[:8]
        results[run_hash] = {}
        results[run_hash]["inverse temperature"] = inverse_temperature
        results[run_hash]["interaction strengths"] = []
        results[run_hash]["delta t's"] = []
        results[run_hash]["interaction lambda's"] = []

        for ind, interaction_strength in enumerate(LAMBDAS):
            neu_max = max(((t_max ** 2) * (30 / EPS) * interaction_strength), 1)
            delta_t = t_max / neu_max
            interaction_lambda = 1 / delta_t

            interaction_string = str(interaction_strength).replace('.', '')
            results[run_hash]["interaction strengths"].append(interaction_strength)
            results[run_hash]["delta t's"].append(delta_t)
            results[run_hash]["interaction lambda's"].append(interaction_lambda)

            rho_environment = environment_state(inverse_temperature=inverse_temperature)

            lindbladian_results = []
            interaction_results = []
            for time in times:
                nu_prime = int(neu_max * (time / t_max))

                lindbladian_result = lindbladian_simulation(
                    system_size=QUBIT_COUNT,
                    time=time,
                    coupling_parameter=interaction_lambda,
                    rho_system=rho_system,
                    inverse_temperature=inverse_temperature,
                    system_hamiltonian=system_hamiltonian,
                    observable=observable,
                    )
                lindbladian_results.append(lindbladian_result)
                
                interaction_result = collision_simulation(
                                        system_size=QUBIT_COUNT,
                                        rho_system=rho_system,
                                        rho_environment=rho_environment,
                                        delta_t=delta_t,
                                        nu_prime=nu_prime,
                                        observable=observable,
                                        system_hamiltonian=system_hamiltonian,
                                        coupling_parameter=interaction_lambda
                                    )
                interaction_results.append(interaction_result)

            results[run_hash][interaction_string] = {}
            results[run_hash][interaction_string]["lindbladian"] = lindbladian_results
            results[run_hash][interaction_string]["interaction"] = interaction_results
            ax = sns.lineplot(x=times, y=lindbladian_results, label=f"Lindbladian", color=COLORS[ind])
            ax = sns.scatterplot(x=times, y=interaction_results, color=COLORS[ind])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.ylabel(r"Overall Magnetization")
        plt.xlabel(r"Evolution time")

        os.makedirs(f"{DIR}/{random_hash}/", exist_ok=True)
        file_name = f"{DIR}/{random_hash}/{run_hash}.png"
        plt.savefig(file_name, dpi=450)

    plt.close()
    # Save parameter mapping to JSON after every run
    with open(f'{DIR}/{random_hash}/param_mapping.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Generated all plots and saved parameter mapping")


if __name__ == "__main__":
    test_main()
