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
from ising.nonmarkovian.swap import swap_channel

from tqdm import tqdm


ZERO = np.array([[1], [0]])
ONE = np.array([[0], [1]])
SIGMA_MINUS = np.array([[0, 1], [0, 0]]) # a_dag
SIGMA_PLUS = np.array([[0, 0], [1, 0]]) # a_n

def calculate_gamma(beta):
    return np.exp(-beta) / (1 + np.exp(-beta))


QUBIT_COUNT = 6
TIME_RANGE = (0, 20)
TIME_COUNT = 20
EPS = 1

# GAMMAS = [0.01, 0.1, 0.5, 0.9, 2, 10]
PS_STRENGTHS = [0, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1.0]
INV_TEMP = 1
GAMMA = 2.0

H_VAL = -0.1
COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#FFD700", "#00CED1", "#FF1493", "#8B0000"]
FOLDER = "partial_swap"


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
def ham_evo_nonmarkovian(rho_sys, rho_env, ham_sys, partial_swap, gamma, time, neu, observable):
    """
    Perform nonmarkovian evolution with the addition of partial swap of 
    post-interaction environment qubit and the fresh qubit before used in the
    next interaction

    Inputs:
        - rho_sys: Initial state of system only
        - rho_env: Initial state of environment only
        - ham_sys: Hamiltonian for system, same size as `rho_sys`
        - partial_swap: Strength of the partial swap
        - gamma: Strength of amplitude damping
        - time: Evolution time to match
    """
    if time == 0:
        rho_sys_norm = rho_sys / global_phase(rho_sys)
        _is_valid_rho(rho_sys_norm)
        return np.trace(np.abs(observable @ rho_sys_norm))

    tau = time / neu
    big_ham_sys = np.kron(ham_sys, np.eye(2))


    ham_ints = interaction_hamiltonian(QUBIT_COUNT, gamma=gamma)

    us = []
    udags = []
    print(f"Running for time:{time}, neu:{neu}")
    for ham_int in ham_ints:
        ham = (np.sqrt(tau) * big_ham_sys / QUBIT_COUNT) + ham_int
        eig_val, eig_vec = np.linalg.eig(ham)
        eig_vec_inv = np.linalg.inv(eig_vec)

        u = matrix_exp(eig_vec, eig_val, eig_vec_inv, time=np.sqrt(tau))
        udags.append(matrix_exp(eig_vec, eig_val, eig_vec_inv, time=-np.sqrt(tau)))
        us.append(u)

    print("Ham operator calculation complete")
    cur_rho_sys = rho_sys
    cur_rho_env = rho_env

    _is_valid_rho(rho_sys)
    _is_valid_rho(rho_env)

    # Scale the number of collisions with a factor of qubit_count to match evolution time
    with tqdm(total=neu) as pbar:
        for sys_ind in range(QUBIT_COUNT * neu):
            pbar.update(1)
            # pick interaction hamiltonian based on the collision number
            u = us[sys_ind % QUBIT_COUNT]
            udag = udags[sys_ind % QUBIT_COUNT]
            complete_rho = np.kron(cur_rho_sys, cur_rho_env)
            complete_rho = make_valid_rho(complete_rho)
            _is_valid_rho(complete_rho)
            rho_fin = (
                u @ complete_rho @ udag
            )
            rho_fin = make_valid_rho(rho_fin)
            _is_valid_rho(rho_fin)

            """
            The remaining loop consists of getting the system state and the
            remains of the environment state. The environment state will
            interact with the next fresh environment, let this be second 
            environment. The interaction will be a partial swap. After the 
            interaction we will use the second environment in the next run.
            """

            cur_rho_sys = partial_trace(rho_fin, list(range(QUBIT_COUNT, QUBIT_COUNT + 1)))
            cur_rho_sys = make_valid_rho(cur_rho_sys)
            _is_valid_rho(cur_rho_sys)
            # trace out the system and you get the environment qubit
            new_rho_env = partial_trace(rho_fin, list(range(0, QUBIT_COUNT)))
            new_rho_env = make_valid_rho(new_rho_env)
            _is_valid_rho(new_rho_env)

            # Bring in the fresh environment qubit and perform partial swap
            rho_env_fin = swap_channel(np.kron(new_rho_env, rho_env), partial_swap)
            rho_env_fin = make_valid_rho(rho_env_fin)
            _is_valid_rho(rho_env_fin)
            # THe next environemnt qubit would be the second qubit after partial swap
            cur_rho_env = partial_trace(rho_env_fin, [0])
            cur_rho_env = make_valid_rho(cur_rho_env)
            _is_valid_rho(cur_rho_env)

    # Get the final 
    rho_ham = np.round(cur_rho_sys, decimals=6)
    rho_ham = make_valid_rho(rho_ham)
    _is_valid_rho(rho_ham)
    rho_ham_norm = rho_ham / global_phase(rho_ham)
    _is_valid_rho(rho_ham_norm)

    return np.trace(np.abs(observable @ rho_ham_norm))


def _random_psi(num_qubits):
    # Calculate the dimension of the state vector
    dim = 2**num_qubits
    
    # Generate complex random numbers for real and imaginary parts
    real_parts = np.random.randn(dim)
    imag_parts = np.random.randn(dim)
    
    # Combine into a complex state vector
    psi = real_parts + 1j * imag_parts
    
    # Normalize the state vector
    psi = psi / np.linalg.norm(psi)
    
    return psi


def test_main():
    np.random.seed(42)
    import json
    import os
    import hashlib
    from datetime import datetime

    # Create directory if it doesn't exist
    os.makedirs(f'plots/nonmarkovian/{FOLDER}', exist_ok=True)
    
    # Initialize parameter mapping dictionary
    param_mapping = {}

    psi = np.zeros(2 ** QUBIT_COUNT)
    psi[0] = 1
    rho_sys = np.outer(psi, psi.conj())
    ham = parametrized_ising(QUBIT_COUNT, H_VAL).matrix
    observable = overall_magnetization(QUBIT_COUNT).matrix
    times = np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)

    plt.clf()  # Clear figure for new plot
    for ind, ps_strength in enumerate(PS_STRENGTHS):
        
        # for ind, inv_temp in enumerate(INV_TEMPS):
        alpha, beta = 1, np.exp(-INV_TEMP) 
        rho_env = (alpha * np.outer(ZERO, ZERO) + beta * np.outer(ONE, ONE)) / (alpha + beta)
        rho_env = make_valid_rho(rho_env)

        interaction = []
        neus = []

        for time in times:
            neu = max(10, int(10 * (time**2) / EPS))
            neus.append(neu)
            interaction.append(ham_evo_nonmarkovian(rho_sys, rho_env, ham, ps_strength, GAMMA, time, neu, observable))

        # Generate unique hash for this parameter combination
        ax = sns.lineplot(x=neus, y=interaction, label=f"$p = {ps_strength}$", color=COLORS[ind])
        
    param_str = f"{GAMMA}_{datetime.now()}"
    hash_id = hashlib.md5(param_str.encode()).hexdigest()[:8]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.ylabel(r"Overall Magnetization")
    plt.xlabel(r"Number of collisions")
    # plt.legend('', frameon=False)

    # Save plot with hash
    file_name = f"plots/nonmarkovian/{FOLDER}/plot_{hash_id}.png"
    plt.savefig(file_name, dpi=450)
    
    # Store parameter mapping
    param_mapping[hash_id] = {
        'gamma': float(GAMMA),
        'ps_strength': float(ps_strength),
        'filename': file_name
    }
    
    plt.close()
    # Save parameter mapping to JSON after every run
    with open(f'plots/nonmarkovian/{FOLDER}/param_mapping.json', 'w') as f:
        json.dump(param_mapping, f, indent=4)

    print("Generated all plots and saved parameter mapping")

if __name__ == "__main__":
    test_main()