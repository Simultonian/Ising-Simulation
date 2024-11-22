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
from ising.nonmarkovian.swap import parameterized_swap

from tqdm import tqdm


ZERO = np.array([[1], [0]])
ONE = np.array([[0], [1]])
SIGMA_MINUS = np.array([[0, 1], [0, 0]]) # a_dag
SIGMA_PLUS = np.array([[0, 0], [1, 0]]) # a_n

def calculate_gamma(beta):
    return np.exp(-beta) / (1 + np.exp(-beta))


QUBIT_COUNT = 4
GAMMA = 0.1
PS_COUNT = 5
PS_STRENGTH = np.pi/2 - 0.3
TIME_RANGE = (0, 20)
TIME_COUNT = 40
EPS = 1
# INV_TEMP = 1

INV_TEMPS = [0.1, 1, 2, 5]

H_VAL = -0.1
COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF"]



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
def ham_evo_nonmarkovian(ham_sys, partial_swap, gamma, time, neu, observable):
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

    ps_u = parameterized_swap(partial_swap)

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
            rho_env_fin = ps_u @ np.kron(new_rho_env, rho_env) @ ps_u.conj().T
            rho_env_fin = make_valid_rho(rho_env_fin )
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


def test_main():
    np.random.seed(42)

    psi = _random_psi(QUBIT_COUNT)
    rho_sys = np.outer(psi, psi.conj())
    # ham = np.zeros_like(rho_sys)
    ham = parametrized_ising(QUBIT_COUNT, H_VAL).matrix
    observable = overall_magnetization(QUBIT_COUNT).matrix
    times = np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)

    interaction_og = []
    count = 20
    plt.clf()
    for ind, inv_temp in enumerate(INV_TEMPS):
        alpha, beta = 1, np.exp(-inv_temp) 
        rho_env = (alpha * np.outer(ZERO, ZERO) + beta * np.outer(ONE, ONE)) / (alpha + beta)
        rho_env = make_valid_rho(rho_env)

        interaction = []
        neus = []
        for time in times:
            neu = max(10, int(10 * (time**2) / EPS))
            neus.append(neu)
            interaction.append(ham_evo_nonmarkovian(rho_sys, rho_env, ham, PS_STRENGTH, GAMMA, time, neu, observable))


        if len(interaction_og) == 0:
            interaction_og = [interaction[0]]


        print(interaction_og + interaction[5:count])
        ax = sns.lineplot(
            x=[0] + neus[5:count],
            y=interaction_og + interaction[5:count],
            label=f"SAL inv_temp={_round(inv_temp)}",
            # s=35,
            color=COLORS[ind],
            # alpha = 1 - opacity[ps_ind]
        )

    # Remove the top and right border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add labels for each group
    plt.ylabel(r"Overall Magnetization")
    plt.xlabel(r"Number of collisions")

    file_name = f"plots/nonmarkovian/swap/no_label_multi_temp_{QUBIT_COUNT}.png"

    ax.get_legend().remove()
    # plt.legend(loc="upper right", bbox_to_anchor=(0.48, 1.15), ncol=1, fontsize=10)
    # plt.legend(ncol=1, fontsize=7)
    plt.savefig(file_name, dpi=450)
    print(f"saved the plot to {file_name}")


if __name__ == "__main__":
    test_main()
