import numpy as np
import json
from qiskit.quantum_info import Pauli, SparsePauliOp

from ising.utils import global_phase
from ising.utils.trace import partial_trace
from ising.hamiltonian import parametrized_ising, Hamiltonian
from ising.observables import overall_magnetization
from ising.lindbladian.simulation.multi_cm_damping import lindblad_evo, interaction_hamiltonian
from ising.lindbladian.simulation.imprecise import GroupedLieCircuit, Taylor

from ising.lindbladian.simulation.utils import load_interaction_hams
from ising.simulation.taylor.utils import calculate_mu

from tqdm import tqdm


ZERO = np.array([[1], [0]])
ONE = np.array([[0], [1]])
SIGMA_MINUS = np.array([[0, 1], [0, 0]])
SIGMA_PLUS = np.array([[0, 0], [1, 0]])
PSI_PLUS = np.array([[1], [1]]) / np.sqrt(2)
RHO_PLUS = np.outer(PSI_PLUS, PSI_PLUS.conj())


QUBIT_COUNT = 3
GAMMA = 0.5
INV_TEMP = 1
TIME_RANGE = (1, 5)
TIME_COUNT = 10
EPS = 0.5
DELTA = 0.9

SAL_RUNS = int(
            8
            * np.ceil(
                ((np.log(2 / (1 - DELTA)))) / (EPS ** 2)
            )
        )

# SAL_RUNS = 100
H_VAL = -0.1
COLORS = ["#DC5B5A", "#625FE1", "#94E574", "#2A2A2A", "#D575EF"]



def matrix_exp(eig_vec, eig_val, eig_vec_inv, time: float):
    return eig_vec @ np.diag(np.exp(complex(0, -1) * time * eig_val)) @ eig_vec_inv


def _random_psi(qubit_count):
    real_psi = np.random.uniform(-1, 1, 2**qubit_count)
    norm = sum([np.abs(x) for x in real_psi]) ** 0.5
    return real_psi / norm


def _get_thermal_state(beta):
    alpha, beta = 1, np.exp(-beta) 
    return (alpha * np.outer(ZERO, ZERO) + beta * np.outer(ONE, ONE)) / (alpha + beta)

def _calculate_gamma(beta):
    return np.exp(-beta) / (1 + np.exp(-beta))

from ising.lindbladian.simulation.utils import save_interaction_hams, load_interaction_hams, interaction_hamiltonian, interaction_hamiltonian_sparse

def test_main():
    np.random.seed(42)
    results = {"interaction": {}, "lindbladian": {}, "sal": {}}

    gamma = GAMMA
    inv_temp = INV_TEMP

    times = np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)

    psi = _random_psi(qubit_count=QUBIT_COUNT)
    rho_sys = np.outer(psi, psi.conj())

    # Environment qubit is always in ZERO, and it is always only one qubit each
    rho_env = _get_thermal_state(inv_temp)
    z = _calculate_gamma(inv_temp)

    # QUBIT_COUNT size Hamiltonians
    ham_sys = parametrized_ising(QUBIT_COUNT, H_VAL).sparse_repr

    # QUBIT_COUNT + 1 size Hamiltonians
    ham_ints_sparse = load_interaction_hams(QUBIT_COUNT, gamma)

    big_ham_sys = SparsePauliOp([x.to_label() + "I" for x in ham_sys.paulis], ham_sys.coeffs)
    observable = overall_magnetization(QUBIT_COUNT)


    for time in times:
        rho_lin = lindblad_evo(rho_sys, ham_sys.to_matrix(), gamma, z, time)
        rho_lin = rho_lin / global_phase(rho_lin)
        results["lindbladian"][time] = np.trace(np.abs(observable.matrix @ rho_lin))

    for time in times:
        neu = max(100, int((time**2) / EPS))
        tau = time / neu

        # QUBIT_COUNT are the number of collision sites
        ham_sim_error = EPS / (9 * QUBIT_COUNT * neu)

        print(f"Running for time:{time}, neu:{neu}")
        us = []
        taylors = []
        for ham_int in ham_ints_sparse:
            ham_sparse = (np.sqrt(tau) * big_ham_sys / QUBIT_COUNT) + ham_int

            eig_val, eig_vec = np.linalg.eig(ham_sparse.to_matrix())
            eig_vec_inv = np.linalg.inv(eig_vec)

            u = matrix_exp(eig_vec, eig_val, eig_vec_inv, time=np.sqrt(tau))
            us.append(u)
            
            """
            This is constructing the Taylor Circuits
            """
            taylor = Taylor(ham_sparse.paulis, ham_sparse.coeffs, ham_sim_error)
            taylor.setup_time(np.sqrt(tau))
            taylors.append(taylor)

        print("Ham operator calculation complete")

        rho_sys_exact = rho_sys.copy()
        with tqdm(total=neu) as pbar:
            for _ in range(neu):
                pbar.update(1)
                for u in us:
                    complete_rho = np.kron(rho_sys_exact, rho_env)
                    rho_fin = (
                        u @ complete_rho @ u.conj().T
                    )
                    rho_sys_exact = partial_trace(rho_fin, list(range(QUBIT_COUNT, QUBIT_COUNT + 1)))

        results["interaction"][time] = np.trace(np.abs(observable.matrix @ rho_sys_exact))

        print("Exact complete, sampling from SAL now")
        with tqdm(total=SAL_RUNS, desc="SAL runs") as pbar:
            sampling_results = []
            for _ in range (SAL_RUNS):
                pbar.update(1)
                rho_sys_sal = rho_sys.copy()
                with tqdm(total=neu, desc="Neu bar", leave=False) as pbar_inner:
                    for _ in range(neu):
                        pbar_inner.update(1)
                        for taylor in taylors:
                            complete_rho = np.kron(rho_sys_sal, rho_env)
                            complete_rho = np.kron(RHO_PLUS, complete_rho)

                            u = taylor.sample_matrix()

                            rho_fin = (
                                u @ complete_rho @ u.conj().T
                            )
                            rho_fin = partial_trace(rho_fin, [0])

                            rho_sys_sal = partial_trace(rho_fin, list(range(QUBIT_COUNT, QUBIT_COUNT + 1)))

                result = np.trace(np.abs(observable.matrix @ rho_sys_sal))
                sampling_results.append(result)

            results["sal"][time] = calculate_mu(sampling_results, SAL_RUNS, [1])



    file_name = f"data/lindbladian/time_vs_magn/size_{QUBIT_COUNT}.json"
    with open(file_name, "w") as file:
        json.dump(results, file)

    print(f"saved the data to {file_name}")


if __name__ == "__main__":
    test_main()
