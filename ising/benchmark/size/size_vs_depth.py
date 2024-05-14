import numpy as np
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

SIZE_RANGE = (3, 14)
SIZE_COUNT = 10
TIME = 1.0
TROTTER_ORDER = 2
H_VAL = 1.0
ERROR = 0.01

GATE = "cx"
FILE_NAME = "ising_one"


def _truncate(num, digits=5):
    a, dot, b = str(num).partition(".")
    return a + dot + b[:digits]


from ising.hamiltonian import parametrized_ising


def main():
    answers = {
        "META": {
            "SIZE_RANGE": f"{SIZE_RANGE[0]},{SIZE_RANGE[1]}",
            "TIME": str(TIME),
            "GATE": str(GATE),
        },
        "taylor": {},
        "qdrift": {},
        "trotter1": {},
        "trotter2": {},
    }

    size_points = [
        int(x) for x in np.linspace(SIZE_RANGE[0], SIZE_RANGE[1], SIZE_COUNT)
    ]

    for qubits in size_points:
        q_str = _truncate(qubits)
        hamiltonian = parametrized_ising(qubits=qubits, h=H_VAL)
        lambd = np.sum(np.abs(hamiltonian.coeffs))

        taylor_bench = TaylorBenchmarkTime(hamiltonian)
        trotter_bench = TrotterBenchmarkTime(hamiltonian)
        qdrift_bench = QDriftBenchmarkTime(hamiltonian)
        ktrotter_bench = KTrotterBenchmarkTime(hamiltonian, order=TROTTER_ORDER)

        print("Calculating the alpha commutators")
        alpha_com_second = alpha_commutator_second_order(hamiltonian.sparse_repr)
        alpha_com_first = alpha_commutator_first_order(hamiltonian.sparse_repr)
        print("Completed the calculation")

        k = int(
            np.floor(
                np.log(lambd * TIME / ERROR) / np.log(np.log(lambd * TIME / ERROR))
            )
        )

        count = taylor_bench.simulation_gate_count(TIME, k)
        print(f"Taylor:{count}")
        answers["taylor"][q_str] = _truncate(count.get(GATE, 0))

        trotter_rep = commutator_r_first_order(
            hamiltonian.sparse_repr, TIME, ERROR, alpha_com=alpha_com_first
        )
        count = trotter_bench.simulation_gate_count(TIME, trotter_rep)
        print(f"Trotter:{count}")
        answers["trotter1"][q_str] = _truncate(count.get(GATE, 0))

        qdrift_rep = qdrift_count(lambd, TIME, ERROR)
        count = qdrift_bench.simulation_gate_count(TIME, qdrift_rep)
        print(f"qDRIFT:{count}")
        answers["qdrift"][q_str] = _truncate(count.get(GATE, 0))

        ktrotter_reps = commutator_r_second_order(
            hamiltonian.sparse_repr, TIME, ERROR, alpha_com=alpha_com_second
        )
        count = ktrotter_bench.simulation_gate_count(TIME, ktrotter_reps)
        print(f"kTrotter:{count}")
        answers["trotter2"][q_str] = _truncate(count.get(GATE, 0))

        print("-----------------")

    json_name = f"data/benchmark/size/{FILE_NAME}_{GATE}_h_eps.json"

    print(answers)

    print(f"Saving results at: {json_name}")
    with open(json_name, "w") as file:
        json.dump(answers, file)

    print(f"Saving data at:{json_name}")


if __name__ == "__main__":
    main()
