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
from ising.groundstate.simulation.utils import (
    ground_state_constants,
    ground_state_maximum_time,
)
import json
from ising.hamiltonian import parse


QUBIT = 5
H_VAL = 0.1
ERROR_RANGE = (-1, -4)
ERROR_COUNT = 5
OVERLAP = 0.1
PROBABILITY = 0.1


# ERROR, TIME
OBS_NORM = 1
FILE_NAME = f"methane"


def test_main():
    # One row is fixed time
    error_points = [10**x for x in np.linspace(ERROR_RANGE[0], ERROR_RANGE[1], ERROR_COUNT)]

    ham = parse(FILE_NAME)
    # ham = parametrized_ising(QUBIT, H_VAL)

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
