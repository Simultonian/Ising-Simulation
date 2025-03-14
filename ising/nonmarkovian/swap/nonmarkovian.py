from ising.lindbladian.simulation.lindbladian import (environment_state, get_interaction_hamiltonians, matrix_exp)

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


"""
Simulation constants
"""
COLLISION_RANGE = (1, 10000)
COLLISION_COUNT = 10
QUBIT_COUNT = 6

"""
System constants
"""
H_VAL = -0.1

"""
Environment constants
"""
LAMBDAS = [0.0005, 0.001, 0.01, 0.1, 1, 0.5, 0.1, 0.8, 10.0]
INV_TEMPS = [0.1, 0.5, 1, 5, 10]
PARTIAL_SWAPS = [0, 0.1, 0.5, 0.9, 0.99, 1.0]

"""
Simulation constants
"""
COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF", 
          "#3ABEFF", "#FFB743", "#00CC99", "#FF6B6B", "#845EC2", 
          "#F9F871", "#00C9A7", "#C34A36", "#4ECDC4", "#FF9671", 
          "#FFC75F", "#008080", "#D65DB1", "#4D8076", "#FF8066"]
DIR = "plots/nonmarkovian/new_simulation"

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

def test_main():




if __name__ == "__main__":
    test_main()