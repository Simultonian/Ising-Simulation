import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from ising.lindbladian.simulation.unraveled import (
    thermal_lindbladians,
    lindbladian_operator,
)

from ising.utils import global_phase, hache
from ising.utils.trace import partial_trace
from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization

from tqdm import tqdm


ZERO = np.array([[1], [0]])
ONE = np.array([[0], [1]])
SIGMA_MINUS = np.array([[0, 1], [0, 0]]) # a_dag
SIGMA_PLUS = np.array([[0, 0], [1, 0]]) # a_n

QUBIT_COUNT = 6
GAMMAS = [1, 0.5, 0.1, 0.8, 10.0]
TIME_RANGE = (0, 20)
TIME_COUNT = 20
EPS = 0.1

H_VAL = -0.1
COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF", 
          "#3ABEFF", "#FFB743", "#00CC99", "#FF6B6B", "#845EC2", 
          "#F9F871", "#00C9A7", "#C34A36", "#4ECDC4", "#FF9671", 
          "#FFC75F", "#008080", "#D65DB1", "#4D8076", "#FF8066"]


def calculate_gamma(beta):
    return np.exp(-beta) / (1 + np.exp(-beta))

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


@hache(blob_type=float, max_size=1000)
def lindblad_evo(rho, ham, gamma, z, time, observable):
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

    rho_lin = np.round(rho_final, decimals=6)
    rho_lin = rho_lin / global_phase(rho_lin)

    return np.trace(np.abs(observable @ rho_lin))

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


@hache(blob_type=float, max_size=1000)
def ham_evo(rho_sys, rho_env, ham_sys, gamma, time, neu, observable):
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


    ham_ints = interaction_hamiltonian(QUBIT_COUNT, gamma=gamma)

    us = []
    print(f"Running for time:{time}, neu:{neu}")
    for ham_int in ham_ints:
        ham = (np.sqrt(tau) * big_ham_sys / QUBIT_COUNT) + ham_int
        eig_val, eig_vec = np.linalg.eig(ham)
        eig_vec_inv = np.linalg.inv(eig_vec)

        u = matrix_exp(eig_vec, eig_val, eig_vec_inv, time=np.sqrt(tau))
        us.append(u)

    print("Ham operator calculation complete")

    cur_rho_sys = rho_sys
    with tqdm(total=neu) as pbar:
        for _ in range(neu):
            pbar.update(1)
            for u in us:
                complete_rho = np.kron(cur_rho_sys, rho_env)
                rho_fin = (
                    u @ complete_rho @ u.conj().T
                )
                cur_rho_sys = partial_trace(rho_fin, list(range(QUBIT_COUNT, QUBIT_COUNT + 1)))

    rho_ham = np.round(cur_rho_sys, decimals=6)
    rho_ham = rho_ham / global_phase(rho_ham)

    return np.trace(np.abs(observable @ rho_ham))


def _random_psi(qubit_count):
    psi = np.zeros(2 ** qubit_count)
    psi[0] = 1
    return psi

import json
import hashlib
import time as time_lib
import os

def test_main():

    np.random.seed(42)

    # Create a random hash based on current time
    random_hash = hashlib.md5(str(time_lib.time()).encode()).hexdigest()[:8]
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots/lindbladian/simulation", exist_ok=True)

    saved_dict = {}

    psi = _random_psi(qubit_count=QUBIT_COUNT)
    rho_sys = np.outer(psi, psi.conj())
    # ham = np.zeros_like(rho_sys)
    ham = parametrized_ising(QUBIT_COUNT, H_VAL).matrix

    inv_temp = 1000000
    saved_dict["inv_temp"] = inv_temp
    saved_dict["qubits"] = QUBIT_COUNT
    saved_dict["h_val"] = H_VAL
    saved_dict["eps"] = EPS
    beta = np.exp(-inv_temp) 
    saved_dict["beta"] = beta
    rho_env1 = (np.outer(ZERO, ZERO) + beta * np.outer(ONE, ONE)) / (1 + beta)

    observable = overall_magnetization(QUBIT_COUNT).matrix

    z = calculate_gamma(inv_temp)
    saved_dict["gamma"] = z

    times = np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)
    saved_dict["times"] = times.tolist()  # Convert numpy array to list for JSON serialization

    saved_dict["results"] = {}
    for gamma_ind, gamma in enumerate(GAMMAS):
        gamma_key = str(gamma)  # Convert gamma to string for JSON key
        saved_dict["results"][gamma_key] = {}
        saved_dict["results"][gamma_key]["taus"] = {}
        interaction, lindbladian = [], []
        for time in times:
            time_key = str(time)
            neu = max(100, int(400 * (time**2) / EPS))
            tau = time / neu

            saved_dict["results"][gamma_key]["taus"][time_key] = tau

            interaction.append(float(ham_evo(rho_sys, rho_env1, ham, gamma, time, neu, observable)))
            lindbladian.append(float(lindblad_evo(rho_sys, ham, gamma, z, time, observable)))
        saved_dict["results"][gamma_key]["interaction"] = interaction
        saved_dict["results"][gamma_key]["lindbladian"] = lindbladian

        ax = sns.lineplot(
            x=times,
            y=lindbladian,
            label=f"Lindbladian {gamma}",
            color=COLORS[gamma_ind],
        )
        ax = sns.scatterplot(
            x=times,
            y=interaction,
            label=f"Single Ancilla LCU {gamma}",
            s=35,
            color=COLORS[gamma_ind],
        )

    # Remove the top and right border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add labels for each group
    plt.ylabel(r"Overall Magnetization")
    plt.xlabel(r"Evolution time")

    # Base filename with hash
    base_file_name = f"plots/lindbladian/simulation/multi_cm_magn_gamma_{random_hash}"
    
    # Version with legend
    plt.legend(loc="upper center", bbox_to_anchor=(0.48, 1.15), ncol=3, fontsize=10)
    plt.savefig(f"{base_file_name}_with_legend.png", dpi=300)
    print(f"Saved the plot with legend to {base_file_name}_with_legend.png")
    
    # Version without legend
    ax.get_legend().remove()
    plt.savefig(f"{base_file_name}_no_legend.png", dpi=300)
    print(f"Saved the plot without legend to {base_file_name}_no_legend.png")
    
    # Save the dictionary as JSON
    with open(f"{base_file_name}_data.json", "w") as f:
        json.dump(saved_dict, f, indent=4)
    print(f"Saved the dictionary to {base_file_name}_data.json")


if __name__ == "__main__":
    test_main()
