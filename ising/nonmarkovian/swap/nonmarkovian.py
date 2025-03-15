import numpy as np
import json
import hashlib
import time as time_lib
import os
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from ising.lindbladian.simulation.old_lindbladian import (
    interaction_hamiltonian,
    _random_psi,
    matrix_exp
)
from ising.nonmarkovian.swap import swap_channel

from ising.utils import global_phase, hache
from ising.utils.trace import partial_trace
from ising.hamiltonian import parametrized_ising
from ising.observables import overall_magnetization



ZERO = np.array([[1], [0]])
ONE = np.array([[0], [1]])
SIGMA_MINUS = np.array([[0, 1], [0, 0]]) # a_dag
SIGMA_PLUS = np.array([[0, 0], [1, 0]]) # a_n

QUBIT_COUNT = 6
# GAMMAS = [1, 0.5, 0.1, 10.0]
GAMMAS = [20]
PARTIAL_SWAPS = [0, 0.9, 0.99, 0.999, 1]
COLLISION_RANGE = (0, 200)
COLLISION_COUNT = 20
DELTA_T = 0.0025

H_VAL = -0.1
COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF", 
          "#3ABEFF", "#FFB743", "#00CC99", "#FF6B6B", "#845EC2", 
          "#F9F871", "#00C9A7", "#C34A36", "#4ECDC4", "#FF9671", 
          "#FFC75F", "#008080", "#D65DB1", "#4D8076", "#FF8066"]
DIR = "plots/nonmarkovian/new_partial_swap/"


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

@hache(blob_type=float, max_size=1000)
def ham_evo_nonmarkovian(system_size, rho_sys, rho_env, ham_sys, gamma, delta_t, neu, observable, partial_swap):
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
    big_ham_sys = np.kron(ham_sys, np.eye(2))


    ham_ints = interaction_hamiltonian(system_size, gamma=gamma)

    us = []
    udags = []
    print(f"Running for collision count:{neu}")
    for ham_int in ham_ints:
        ham = (np.sqrt(delta_t) * big_ham_sys / system_size) + ham_int
        eig_val, eig_vec = np.linalg.eig(ham)
        eig_vec_inv = np.linalg.inv(eig_vec)

        u = matrix_exp(eig_vec, eig_val, eig_vec_inv, time=np.sqrt(delta_t))
        us.append(u)
        udags.append(u.conj().T)

    print("Ham operator calculation complete")

    cur_rho_sys = rho_sys
    carry_rho_env = rho_env
    with tqdm(total=neu) as pbar:
        for _ in range(neu):
            pbar.update(1)
            for ind, u in enumerate(us):
                complete_rho = np.kron(cur_rho_sys, carry_rho_env)
                rho_fin = (
                    u @ complete_rho @ udags[ind]
                )
                rho_fin = make_valid_rho(rho_fin)
                _is_valid_rho(rho_fin)

                cur_rho_sys = partial_trace(rho_fin, list(range(system_size, system_size + 1)))
                cur_rho_sys = make_valid_rho(cur_rho_sys)
                _is_valid_rho(cur_rho_sys)
                # rho_env after interaction with the system
                cur_rho_env = partial_trace(rho_fin, list(range(0, system_size)))
                cur_rho_env = make_valid_rho(cur_rho_env)
                _is_valid_rho(cur_rho_env)

                # Bring in a fresh environment qubit and perform partial swap with current rho_env
                combined_rho_env = swap_channel(np.kron(cur_rho_env, rho_env), partial_swap)
                # the second environment after partial swap is the carry_rho_env
                carry_rho_env = partial_trace(combined_rho_env, [0])
                carry_rho_env = make_valid_rho(carry_rho_env)
                _is_valid_rho(carry_rho_env)

    rho_ham = np.round(cur_rho_sys, decimals=6)
    rho_ham = rho_ham / global_phase(rho_ham)

    return np.trace(np.abs(observable @ rho_ham))


def test_main():
    np.random.seed(42)

    # Create a random hash based on current time
    random_hash = hashlib.md5(str(time_lib.time()).encode()).hexdigest()[:8]
    
    # Create plots directory if it doesn't exist
    os.makedirs(DIR, exist_ok=True)
    base_dir = f"{DIR}/{random_hash}/"
    os.makedirs(base_dir, exist_ok=True)
            
    saved_dict = {}

    psi = _random_psi(qubit_count=QUBIT_COUNT)
    rho_sys = np.outer(psi, psi.conj())
    # ham = np.zeros_like(rho_sys)
    ham = parametrized_ising(QUBIT_COUNT, H_VAL).matrix

    inv_temp = 1000000
    system_size = QUBIT_COUNT

    saved_dict["inv_temp"] = inv_temp
    saved_dict["qubits"] = system_size
    saved_dict["h_val"] = H_VAL
    beta = np.exp(-inv_temp) 
    saved_dict["beta"] = beta
    rho_env1 = (np.outer(ZERO, ZERO) + beta * np.outer(ONE, ONE)) / (1 + beta)

    observable = overall_magnetization(QUBIT_COUNT).matrix

    collisions = [int(x) for x in np.linspace(COLLISION_RANGE[0], COLLISION_RANGE[1], COLLISION_COUNT)]
    saved_dict["collision count"] = collisions

    saved_dict["results"] = {}
    for gamma_ind, gamma in enumerate(GAMMAS):
        gamma_key = str(gamma)  # Convert gamma to string for JSON key
        saved_dict["results"][gamma_key] = {}
        for p_ind, partial_swap in enumerate(PARTIAL_SWAPS):
            swap_key = str(partial_swap)
            saved_dict["results"][gamma_key][swap_key] = {}
            saved_dict["results"][gamma_key][swap_key]["taus"] = {}

            interaction = []
            for collision_count in collisions:
                collision_key = str(collision_count)
                interaction.append(float(ham_evo_nonmarkovian(system_size, rho_sys, rho_env1, ham, gamma, DELTA_T, collision_count, observable, partial_swap)))
            saved_dict["results"][gamma_key][swap_key]["observable"] = interaction

            swap_hash = hashlib.md5(str(time_lib.time()).encode()).hexdigest()[:8]
            saved_dict["results"][gamma_key][swap_key]["swap_hash"] = swap_hash

            ax = sns.lineplot(
                x=collisions,
                y=interaction,
                label=fr"$p={partial_swap}$",
                color=COLORS[p_ind],
            )
            ax = sns.scatterplot(
                x=collisions,
                y=interaction,
                # label=f"Single Ancilla LCU {gamma}",
                s=25,
                color=COLORS[p_ind],
            )
        # Remove the top and right border
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add labels for each group
        plt.ylabel(r"Average Magnetization")
        plt.xlabel(r"Number of Collisions")

        # Base filename with hash
        # Version with legend
        plt.legend(loc="upper center", bbox_to_anchor=(0.48, 1.15), ncol=3, fontsize=10)
        plt.savefig(f"{base_dir}plot_with_legend.png", dpi=300)
        print(f"Saved the plot with legend to {base_dir}{swap_hash}_plot_with_legend.png")
        
        # Version without legend
        ax.get_legend().remove()
        plt.savefig(f"{base_dir}plot_no_legend.png", dpi=300)
        print(f"Saved the plot without legend to {base_dir}{swap_hash}_plot_no_legend.png")
        plt.clf()  # Clear figure for new plot



    
    # Save the dictionary as JSON
    with open(f"{base_dir}data.json", "w") as f:
        json.dump(saved_dict, f, indent=4)
    print(f"Saved the dictionary to {base_dir}data.json")


if __name__ == "__main__":
    test_main()