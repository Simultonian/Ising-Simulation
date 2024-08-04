import os
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from tqdm import tqdm

from qiskit.quantum_info import SparsePauliOp
from ising.hamiltonian import parametrized_ising_power, parametrized_ising
from ising.benchmark.gates import (
    TaylorBenchmarkTime,
    TrotterBenchmarkTime,
    QDriftBenchmarkTime,
    KTrotterBenchmarkTime,
)
from ising.hamiltonian.ising_one import qdrift_count
from ising.utils.commutator.commutator_hueristic import (
    r_first_order,
    r_second_order,
    alpha_commutator_second_order,
    alpha_commutator_first_order,
)
from ising.groundstate.simulation.utils import (
    ground_state_constants,
    ground_state_maximum_time,
)
import json
from ising.hamiltonian import parse
from ising.lindbladian.simulation.utils import load_interaction_hams


QUBIT_COUNT = 25
GAMMA = 0.5
H_VAL = 0.1
ERROR = 0.1
OVERLAP = 0.1
PROBABILITY = 0.1
TIME_RANGE = (1, 5)
TIME_COUNT = 10


# ERROR, TIME
OBS_NORM = 1
FILE_NAME = f"lindi_{QUBIT_COUNT}"

def test_main2():
    np.random.seed(42)
    results = {"trotter": {}, "sal": {}, "qdrift": {}, "ktrotter": {}}

    times = np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)

    # QUBIT_COUNT size Hamiltonians
    ham_sys = parametrized_ising(QUBIT_COUNT, H_VAL).sparse_repr

    # QUBIT_COUNT + 1 size Hamiltonians
    ham_ints_sparse = load_interaction_hams(QUBIT_COUNT)

    big_ham_sys = SparsePauliOp([x.to_label() + "I" for x in ham_sys.paulis], ham_sys.coeffs)

    for time in times:
        neu = max(10, int(10 * (time**2) / ERROR))
        tau = time / neu

        # QUBIT_COUNT are the number of collision sites
        ham_sim_error = ERROR / (9 * QUBIT_COUNT * neu)

        """
        This section is for the Hamiltonian Simulation that is exact
        """
        print(f"Running for time:{time}, neu:{neu}")
        taylors, trotters, ktrotters, qdrifts = [], [], [], []
        alpha_first_comms = []
        alpha_second_comms = []
        for ham_int_sparse in ham_ints_sparse:
            ham_sparse = (np.sqrt(tau) * big_ham_sys / QUBIT_COUNT) + (GAMMA * ham_int_sparse)
            sorted_pairs = list(
                sorted(
                    [(x, y.real) for (x, y) in zip(ham_sparse.paulis, ham_sparse.coeffs)],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
            )
            alpha_com_first = alpha_commutator_first_order(
                sorted_pairs, ERROR, delta=0, cutoff_count = -1
            )
            alpha_first_comms.append(alpha_com_first)

            alpha_com_second = alpha_commutator_second_order(
                sorted_pairs, ERROR, delta=0, cutoff_count = -1
            )
            alpha_second_comms.append(alpha_com_second)
            taylors.append(TaylorBenchmarkTime(ham_sparse))
            trotters.append(TrotterBenchmarkTime(ham_sparse, system=FILE_NAME))
            qdrifts.append(QDriftBenchmarkTime(ham_sparse))
            ktrotters.append(KTrotterBenchmarkTime(ham_sparse, system=FILE_NAME, order=2))

        print("Ham operator calculation complete")

        with tqdm(total=neu) as pbar:
            for _ in range(neu):
                pbar.update(1)

                """
                This section is for the Hamiltonian Simulation using Trotterization
                """
                for u in lie_us:
                    rho_fin = (
                        u @ complete_rho @ u.conj().T
                    )

        print("Trotter and Exact complete, sampling from SAL now")
        with tqdm(total=neu, desc="Neu bar", leave=False) as pbar_inner:
            for _ in range(neu):
                pbar_inner.update(1)
                for taylor in taylors:
                    u = taylor.sample_matrix()

        results["interaction"][time] = np.trace(np.abs(observable.matrix @ rho_sys_exact))
        results["trotter"][time] = np.trace(np.abs(observable.matrix @ rho_sys_trotter))


    file_name = f"data/lindbladian/time_vs_magn/size_{QUBIT_COUNT}.json"
    with open(file_name, "w") as file:
        json.dump(results, file)

    print(f"saved the data to {file_name}")

def test_main():

    # One row is fixed time
    error_points = [10**x for x in np.linspace(ERROR_RANGE[0], ERROR_RANGE[1], ERROR_COUNT)]

    # ham = parse(FILE_NAME)
    ham = parametrized_ising(QUBIT, H_VAL)

    sorted_pairs = list(
        sorted(
            [(x, y.real) for (x, y) in zip(ham.paulis, ham.coeffs)],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    )

    lambd = np.sum(np.abs(ham.coeffs))

    taylor_bench = TaylorBenchmarkTime(ham)
    trotter_bench = TrotterBenchmarkTime(ham, system=FILE_NAME)
    qdrift_bench = QDriftBenchmarkTime(ham)
    ktrotter_bench = KTrotterBenchmarkTime(ham, system=FILE_NAME, order=2)

    # Calculate the alpha commutators for both first and second order

    alpha_name = f"data/alphacomm/{FILE_NAME}.json"
    min_error = min(error_points)

    results = {
        "taylor": {},
        "trotter": {},
        "qdrift": {},
        "ktrotter": {},
    }

    for error in error_points:
        ground_params = ground_state_constants(ham.spectral_gap, OVERLAP, error, PROBABILITY, obs_norm=1)
        time = ground_state_maximum_time(ground_params)

        trotter_rep = r_first_order(
            sorted_pairs, time, error, alpha_com=alpha_com_first
        )
        trotter_counts = trotter_bench.controlled_gate_count(time, trotter_rep)["cx"]
        print(f"Trotter:{trotter_counts}")

        k = int(
            np.floor(
                np.log(lambd * time / error) / np.log(np.log(lambd * time / error))
            )
        )

        taylor_counts = taylor_bench.simulation_gate_count(time, k)
        print(f"Taylor:{taylor_counts}")


        qdrift_rep = qdrift_count(lambd, time, error)
        qdrift_counts = qdrift_bench.simulation_gate_count(time, qdrift_rep)
        print(f"qDRIFT:{qdrift_counts}")

        ktrotter_reps = r_second_order(
            sorted_pairs, time, error, alpha_com=alpha_com_second
        )
        ktrotter_counts = ktrotter_bench.controlled_gate_count(time, ktrotter_reps)["cx"]
        print(f"kTrotter:{ktrotter_counts}")

        results["taylor"][error] = str(taylor_counts["cx"])
        results["trotter"][error] = str(trotter_counts)
        results["qdrift"][error] = str(qdrift_counts["cx"])
        results["ktrotter"][error] = str(ktrotter_counts)

        print("-----------------")

    json_name = f"data/benchmark/line/error/{FILE_NAME}_{ERROR_RANGE[0]}.json"

    print(results)
    print(f"Saving results at: {json_name}")
    with open(json_name, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    test_main()
