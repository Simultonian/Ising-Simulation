import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
import json
from ising.hamiltonian import parse


def plot_gate_error(
    qubit, h_val, err_pair, delta, point_count_pair, obs_norm, time_pair, file_name
):
    fig, ax = plt.subplots()

    # One col is fixed error
    error_points = [
        10**x for x in np.linspace(err_pair[0], err_pair[1], point_count_pair[0])
    ]
    min_error = min(error_points)
    # One row is fixed time
    time_points = [
        10**x for x in np.linspace(time_pair[0], time_pair[1], point_count_pair[1])
    ]

    # 2D Arrays where the first dim is time and second is error
    taylor, trotter, qdrift, ktrotter = [], [], [], []

    ham = parse(file_name)
    # ham = parametrized_ising(qubit, h_val)

    sorted_pairs = list(
        sorted(
            [(x, y.real) for (x, y) in zip(ham.paulis, ham.coeffs)],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    )

    lambd = np.sum(np.abs(ham.coeffs))

    taylor_bench = TaylorBenchmarkTime(ham)
    trotter_bench = TrotterBenchmarkTime(ham)
    qdrift_bench = QDriftBenchmarkTime(ham)
    ktrotter_bench = KTrotterBenchmarkTime(ham, order=2)

    # Calculate the alpha commutators for both first and second order

    alpha_name = f"data/alphacomm/{file_name}.json"

    if os.path.exists(alpha_name):
        with open(alpha_name, "r") as alpha_file:
            data = json.load(alpha_file)
        alpha_com_first = data["alpha_1"]
        alpha_com_second = data["alpha_2"]
        print("alpha file file, not recomputing")
        print(f"com1:{alpha_com_first} \ncom2:{alpha_com_second}")
    else:
        ll = len(sorted_pairs)
        print("Calculating the alpha commutators")
        alpha_com_first = alpha_commutator_first_order(
            sorted_pairs, min_error, delta=0, cutoff_count=ll**2
        )
        alpha_com_second = alpha_commutator_second_order(
            sorted_pairs, min_error, delta=0, cutoff_count=ll
        )
        print(f"com1:{alpha_com_first} \ncom2:{alpha_com_second}")

        save_alpha = {"alpha_1": alpha_com_first, "alpha_2": alpha_com_second}
        print(f"Saving results at: {alpha_name}")

        with open(alpha_name, "w") as file:
            json.dump(save_alpha, file)

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

            trotter_rep = r_first_order(
                sorted_pairs, time, error, alpha_com=alpha_com_first
            )
            trotter_counts = trotter_bench.circuit_gate_count("cx", trotter_rep)
            print(f"Trotter:{trotter_counts}")

            qdrift_rep = qdrift_count(lambd, time, error)
            qdrift_counts = qdrift_bench.simulation_gate_count(time, qdrift_rep)
            print(f"qDRIFT:{qdrift_counts}")

            ktrotter_reps = r_second_order(
                sorted_pairs, time, error, alpha_com=alpha_com_second
            )
            ktrotter_counts = ktrotter_bench.circuit_gate_count("cx", ktrotter_reps)
            print(f"kTrotter:{ktrotter_counts}")

            taylor[-1].append(np.log2(taylor_counts["cx"]))
            trotter[-1].append(np.log2(trotter_counts))
            qdrift[-1].append(np.log2(qdrift_counts["cx"]))
            ktrotter[-1].append(np.log2(ktrotter_counts))

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


if __name__ == "__main__":
    qubit = 25
    h_val = 0.1
    err_pair = (-1, -5)
    delta = 0.1

    # error, time
    point_count = (5, 20)
    obs_norm = 1
    # log scale
    time_pair = (0, 2)
    file_name = f"methane"
    plot_gate_error(
        qubit, h_val, err_pair, delta, point_count, obs_norm, time_pair, file_name
    )
