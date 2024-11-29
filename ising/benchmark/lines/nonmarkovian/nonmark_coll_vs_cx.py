import numpy as np
from tqdm import tqdm

from qiskit.quantum_info import SparsePauliOp
from ising.hamiltonian import Hamiltonian, parametrized_ising_power, parametrized_ising
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
import json
from ising.lindbladian.simulation.utils import load_interaction_hams


QUBIT_COUNT = 10
GAMMA = 0.5
H_VAL = 0.1
ERROR = 0.1
TIME_RANGE = (1, 1.5)
TIME_COUNT = 10


def test_main():
    np.random.seed(42)

    times = [10 ** x for x in np.linspace(TIME_RANGE[0], TIME_RANGE[1], TIME_COUNT)]

    # QUBIT_COUNT size Hamiltonians
    ham_sys = parametrized_ising(QUBIT_COUNT, H_VAL).sparse_repr

    # QUBIT_COUNT + 1 size Hamiltonians
    ham_ints_sparse = load_interaction_hams(QUBIT_COUNT, 1.0)

    big_ham_sys = SparsePauliOp([x.to_label() + "I" for x in ham_sys.paulis], ham_sys.coeffs)

    results = {"trotter": {}, "sal": {}, "qdrift": {}, "ktrotter": {}}
    for time in times:
        neu = max(10, int(10 * (time**2) / ERROR))
        time_str = str(neu)
        tau = time / neu
        evo_time = np.sqrt(tau)

        # QUBIT_COUNT are the number of collision sites
        ham_sim_error = ERROR / (9 * QUBIT_COUNT * neu)

        """
        This section is for the Hamiltonian Simulation that is exact
        """
        print(f"Running for time:{time}, neu:{neu}")
        taylors, trotters, ktrotters, qdrifts = [], [], [], []
        alpha_com_firsts = []
        alpha_com_seconds = []
        lambds = []
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
                sorted_pairs, ham_sim_error, delta=0, cutoff_count = -1
            )
            alpha_com_firsts.append(alpha_com_first)

            alpha_com_second = alpha_commutator_second_order(
                sorted_pairs, ham_sim_error, delta=0, cutoff_count = -1
            )
            alpha_com_seconds.append(alpha_com_second)


            lambds.append(np.sum(np.abs(ham_sparse.coeffs)))

            taylors.append(TaylorBenchmarkTime(Hamiltonian(sparse_repr=ham_sparse)))
            trotters.append(TrotterBenchmarkTime(Hamiltonian(sparse_repr=ham_sparse), system=""))
            qdrifts.append(QDriftBenchmarkTime(Hamiltonian(sparse_repr=ham_sparse)))
            ktrotters.append(KTrotterBenchmarkTime(Hamiltonian(sparse_repr=ham_sparse), system="", order=2))

        print("Ham operator calculation complete")

        trotter_cx, ktrotter_cx, qdrift_cx, taylor_cx = 0, 0, 0, 0
        """
        This section is for the Hamiltonian Simulation using Trotterization
        """
        for trotter, alpha_com_first in zip(trotters, alpha_com_firsts):
            # If alpha_com is given then sorted_pairs are not needed
            trotter_rep = r_first_order(
                sorted_pairs=[], time=evo_time, error=ham_sim_error, alpha_com=alpha_com_first
            )
            trotter_cx += trotter.controlled_gate_count(evo_time, trotter_rep).get("cx", 0)

        for ktrotter, alpha_com_second in zip(ktrotters, alpha_com_seconds):
            # If alpha_com is given then sorted_pairs are not needed
            ktrotter_rep = r_second_order(
                sorted_pairs=[], time=evo_time, error=ham_sim_error, alpha_com=alpha_com_second
            )
            ktrotter_cx += ktrotter.controlled_gate_count(evo_time, ktrotter_rep).get("cx", 0)

        for taylor, lambd in zip(taylors, lambds):
            k = int(
                np.floor(
                    np.log(lambd * evo_time / ham_sim_error) / np.log(np.log(lambd * evo_time / ham_sim_error))
                )
            )

            taylor_cx += taylor.simulation_gate_count(evo_time, k).get("cx", 0)

        for qdrift, lambd in zip(qdrifts, lambds):
            qdrift_rep = qdrift_count(lambd, evo_time, ham_sim_error)
            qdrift_cx += qdrift.simulation_gate_count(evo_time, qdrift_rep).get("cx", 0)

        results["sal"][time_str] = str(abs(taylor_cx * neu))
        results["trotter"][time_str] = str(abs(trotter_cx * neu))
        results["ktrotter"][time_str] = str(abs(ktrotter_cx * neu))
        results["qdrift"][time_str] = str(abs(qdrift_cx * neu))


    file_name = f"data/benchmark/line/nonmarkovian/size_{QUBIT_COUNT}.json"
    with open(file_name, "w") as file:
        json.dump(results, file)

    print(f"saved the data to {file_name}")

if __name__ == "__main__":
    test_main()
