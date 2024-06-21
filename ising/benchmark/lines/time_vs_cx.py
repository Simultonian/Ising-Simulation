import os
import matplotlib.pyplot as plt
import numpy as np
import math
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


QUBIT = 25
H_VAL = 0.1
ERROR = 0.01
DELTA = 0.1

# ERROR, TIME
OBS_NORM = 1
TIME_PAIR = (0, 2)
TIME_COUNT = 8
FILE_NAME = f"ising_{QUBIT}"


def test_main():
    # One row is fixed time
    time_points = [10**x for x in np.linspace(TIME_PAIR[0], TIME_PAIR[1], TIME_COUNT)]

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
    trotter_bench = TrotterBenchmarkTime(ham)
    qdrift_bench = QDriftBenchmarkTime(ham)
    ktrotter_bench = KTrotterBenchmarkTime(ham, order=2)

    # Calculate the alpha commutators for both first and second order

    alpha_name = f"data/alphacomm/{FILE_NAME}.json"

    if os.path.exists(alpha_name):
        with open(alpha_name, "r") as alpha_file:
            data = json.load(alpha_file)
        alpha_com_first = data["alpha_1"]
        alpha_com_second = data["alpha_2"]
        print("alpha file found, not recomputing")
        print(f"com1:{alpha_com_first} \ncom2:{alpha_com_second}")
    else:
        ll = len(sorted_pairs)
        print("Calculating the alpha commutators")
        alpha_com_first = alpha_commutator_first_order(
            sorted_pairs, ERROR, delta=0, cutoff_count=ll**2
        )
        alpha_com_second = alpha_commutator_second_order(
            sorted_pairs, ERROR, delta=0, cutoff_count=ll
        )
        print(f"com1:{alpha_com_first} \ncom2:{alpha_com_second}")

        save_alpha = {"alpha_1": alpha_com_first, "alpha_2": alpha_com_second}
        print(f"Saving results at: {alpha_name}")

        with open(alpha_name, "w") as file:
            json.dump(save_alpha, file)

    results = {
        "taylor": {},
        "trotter": {},
        "qdrift": {},
        "ktrotter": {},
    }

    for time in time_points:
        k = int(
            np.floor(
                np.log(lambd * time / ERROR) / np.log(np.log(lambd * time / ERROR))
            )
        )

        taylor_counts = taylor_bench.simulation_gate_count(time, k)
        print(f"Taylor:{taylor_counts}")

        trotter_rep = r_first_order(
            sorted_pairs, time, ERROR, alpha_com=alpha_com_first
        )
        trotter_counts = trotter_bench.circuit_gate_count("cx", trotter_rep)
        print(f"Trotter:{trotter_counts}")

        qdrift_rep = qdrift_count(lambd, time, ERROR)
        qdrift_counts = qdrift_bench.simulation_gate_count(time, qdrift_rep)
        print(f"qDRIFT:{qdrift_counts}")

        ktrotter_reps = r_second_order(
            sorted_pairs, time, ERROR, alpha_com=alpha_com_second
        )
        ktrotter_counts = ktrotter_bench.circuit_gate_count("cx", ktrotter_reps)
        print(f"kTrotter:{ktrotter_counts}")

        results["taylor"][time] = str(taylor_counts["cx"])
        results["trotter"][time] = str(trotter_counts)
        results["qdrift"][time] = str(qdrift_counts["cx"])
        results["ktrotter"][time] = str(ktrotter_counts)

        print("-----------------")

    json_name = f"data/benchmark/line/time/{FILE_NAME}.json"

    print(f"Saving results at: {json_name}")
    with open(json_name, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    test_main()
