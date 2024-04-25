import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ising.hamiltonian import parametrized_ising_power
from ising.benchmark.gates import (
    TaylorBenchmarkTime,
    TrotterBenchmarkTime,
    QDriftBenchmarkTime,
    KTrotterBenchmarkTime,
)
from ising.hamiltonian.ising_one import qdrift_count
from ising.utils.commutator import (
    commutator_r_second_order,
    alpha_commutator_second_order,
    alpha_commutator_first_order,
    commutator_r_first_order,
)
import json


def plot_gate_error(
    qubit, h_val, err_pair, point_count_pair, obs_norm, time_pair, file_name
):
    fig, ax = plt.subplots()

    # One col is fixed error
    error_points = [
        10**x for x in np.linspace(err_pair[0], err_pair[1], point_count_pair[0])
    ]
    # One row is fixed time
    time_points = [
        x for x in np.linspace(time_pair[0], time_pair[1], point_count_pair[1])
    ]

    # 2D Arrays where the first dim is time and second is error
    taylor, trotter, qdrift, ktrotter = [], [], [], []

    ham = parametrized_ising_power(qubits=qubit, h=h_val)
    lambd = np.sum(np.abs(ham.coeffs))

    taylor_bench = TaylorBenchmarkTime(ham)
    trotter_bench = TrotterBenchmarkTime(ham)
    qdrift_bench = QDriftBenchmarkTime(ham)
    ktrotter_bench = KTrotterBenchmarkTime(ham, order=2)

    # Calculate the alpha commutators for both first and second order

    print("Calculating the alpha commutators")
    alpha_com_second = alpha_commutator_second_order(ham.sparse_repr)
    alpha_com_first = alpha_commutator_first_order(ham.sparse_repr)
    print("Completed the calculation")

    nrows, ncols = len(time_points), len(error_points)
    for time in time_points:
        taylor.append([])
        trotter.append([])
        qdrift.append([])
        ktrotter.append([])

        for error in error_points:
            k = int(
                np.floor(
                    np.log(lambd * time / error) / np.log(np.log(lambd * time / error))
                )
            )

            taylor_counts = taylor_bench.simulation_gate_count(time, k)
            print(f"Taylor:{taylor_counts}")

            trotter_rep = commutator_r_first_order(
                ham.sparse_repr, time, error, alpha_com_first
            )
            trotter_counts = trotter_bench.simulation_gate_count(time, trotter_rep)
            print(f"Trotter:{trotter_counts}")

            qdrift_rep = qdrift_count(lambd, time, error)
            qdrift_counts = qdrift_bench.simulation_gate_count(time, qdrift_rep)
            print(f"qDRIFT:{qdrift_counts}")

            ktrotter_reps = commutator_r_second_order(
                ham.sparse_repr, time, error, alpha_com_second
            )
            ktrotter_counts = ktrotter_bench.simulation_gate_count(time, ktrotter_reps)
            print(f"kTrotter:{ktrotter_counts}")

            taylor[-1].append(np.log2(taylor_counts["cx"]))
            trotter[-1].append(np.log2(trotter_counts["cx"]))
            qdrift[-1].append(np.log2(qdrift_counts["cx"]))
            ktrotter[-1].append(np.log2(ktrotter_counts["cx"]))

            print("-----------------")

    json_name = f"data/benchmark/heat/{file_name}_gate_err_data.json"
    results = {
        "taylor": taylor,
        "trotter": trotter,
        "qdrift": qdrift,
        "ktrotter": ktrotter,
    }

    print(f"Saving results at: {json_name}")
    with open(json_name, "w") as file:
        json.dump(results, file)

    print(f"Saving data at:{json_name}")


if __name__ == "__main__":
    qubit = 5
    h_val = 0.1
    err_pair = (-1, -5)

    # error, time
    point_count = (3, 10)
    obs_norm = 1
    time_pair = (1, 10)
    file_name = f"ising_power_{qubit}"
    plot_gate_error(qubit, h_val, err_pair, point_count, obs_norm, time_pair, file_name)
