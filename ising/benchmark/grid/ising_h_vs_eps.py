import numpy as np
from ising.benchmark.gates import (
    TaylorBenchmarkTime,
    TrotterBenchmarkTime,
    QDriftBenchmarkTime,
    KTrotterBenchmarkTime,
)
from ising.hamiltonian.ising_one import qdrift_count
from ising.utils.commutator.commutator_hueristic import (
    r_second_order,
    alpha_commutator_second_order,
    alpha_commutator_first_order,
    r_first_order,
)
import json

QUBITS = 25
TIME = 10.0
ORDER = 2

# THE BELOW ARE LOGARITHMIC WRT 10
H_RANGE = (-3, 2)
H_COUNT = 20

ERROR_RANGE = (-1, -5)
ERROR_COUNT = 20
GATE = "cx"
FILE_NAME = "ising_one"


def _truncate(num, digits=5):
    a, dot, b = str(num).partition(".")
    return a + dot + b[:digits]


from ising.hamiltonian import parametrized_ising


def test_main():
    answers = {
        "META": {"QUBITS": str(QUBITS), "TIME": str(TIME), "GATE": str(GATE)},
        "taylor": {},
        "qdrift": {},
        "trotter1": {},
        "trotter2": {},
    }

    # Converting the logirithmic ranges to decimals
    error_points = [
        10**x for x in np.linspace(ERROR_RANGE[0], ERROR_RANGE[1], ERROR_COUNT)
    ]
    h_points = [10**x for x in np.linspace(H_RANGE[0], H_RANGE[1], H_COUNT)]
    min_error = min(error_points)

    for h in h_points:
        h_str = _truncate(h)
        hamiltonian = parametrized_ising(qubits=QUBITS, h=h)
        sorted_pairs = list(
            sorted(
                [(x, y.real) for (x, y) in zip(hamiltonian.paulis, hamiltonian.coeffs)],
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        )

        lambd = np.sum(np.abs(hamiltonian.coeffs))

        taylor_bench = TaylorBenchmarkTime(hamiltonian)
        answers["taylor"][h_str] = {}

        trotter_bench = TrotterBenchmarkTime(hamiltonian)
        answers["trotter1"][h_str] = {}

        qdrift_bench = QDriftBenchmarkTime(hamiltonian)
        answers["qdrift"][h_str] = {}

        ktrotter_bench = KTrotterBenchmarkTime(hamiltonian, order=ORDER)
        answers["trotter2"][h_str] = {}

        print("Calculating the alpha commutators")
        alpha_com_first = alpha_commutator_first_order(
            sorted_pairs, min_error, delta=0, cutoff_count=0
        )
        alpha_com_second = alpha_commutator_second_order(
            sorted_pairs, min_error, delta=0, cutoff_count=0
        )
        print("Completed the calculation")

        for error in error_points:
            error_str = _truncate(error)
            k = int(
                np.floor(
                    np.log(lambd * TIME / error) / np.log(np.log(lambd * TIME / error))
                )
            )

            count = taylor_bench.simulation_gate_count(TIME, k)
            print(f"Taylor:{count}")
            answers["taylor"][h_str][error_str] = _truncate(count.get(GATE, 0))

            trotter_rep = r_first_order(
                sorted_pairs, TIME, error, alpha_com=alpha_com_first
            )
            count = trotter_bench.simulation_gate_count(TIME, trotter_rep)
            print(f"Trotter:{count}")
            answers["trotter1"][h_str][error_str] = _truncate(count.get(GATE, 0))

            qdrift_rep = qdrift_count(lambd, TIME, error)
            count = qdrift_bench.simulation_gate_count(TIME, qdrift_rep)
            print(f"qDRIFT:{count}")
            answers["qdrift"][h_str][error_str] = _truncate(count.get(GATE, 0))

            ktrotter_reps = r_second_order(
                sorted_pairs, TIME, error, alpha_com=alpha_com_second
            )
            count = ktrotter_bench.simulation_gate_count(TIME, ktrotter_reps)
            print(f"kTrotter:{count}")
            answers["trotter2"][h_str][error_str] = _truncate(count.get(GATE, 0))

            print("-----------------")

    json_name = f"data/benchmark/heat/{FILE_NAME}_{GATE}_h_eps.json"

    print(answers)

    print(f"Saving results at: {json_name}")
    with open(json_name, "w") as file:
        json.dump(answers, file)

    print(f"Saving data at:{json_name}")


if __name__ == "__main__":
    test_main()
