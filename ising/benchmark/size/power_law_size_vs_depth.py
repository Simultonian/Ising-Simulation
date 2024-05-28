import numpy as np
from ising.benchmark.gates import (
    TaylorBenchmarkTime,
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

QUBITS = 8
POWER_RANGE = (1, 6)
POWER_COUNT = 5
ERROR = 0.1

## logarithmic range
ERR_RANGE = (-1, -3)
ERR_COUNT = 5

TIME = 1.0
TROTTER_ORDER = 2
H_VAL = 1.0

GATE = "cx"
FILE_NAME = "power_ising_one"


def _truncate(num, digits=5):
    a, dot, b = str(num).partition(".")
    return a + dot + b[:digits]


from ising.hamiltonian import parametrized_ising_power


def main():
    answers = {
        "META": {
            "ERR_RANGE": f"{ERR_RANGE[0]},{ERR_RANGE[1]}",
            "TIME": str(TIME),
            "GATE": str(GATE),
        },
        "taylor": {},
        "ktrotter": {},
    }

    power_points = [
        int(x) for x in np.linspace(POWER_RANGE[0], POWER_RANGE[1], POWER_COUNT)
    ]

    for power in power_points:
        power_str = _truncate(power)
        hamiltonian = parametrized_ising_power(qubits=QUBITS, h=H_VAL, power=power)
        lambd = np.sum(np.abs(hamiltonian.coeffs))

        taylor_bench = TaylorBenchmarkTime(hamiltonian)
        ktrotter_bench = KTrotterBenchmarkTime(hamiltonian, order=TROTTER_ORDER)

        print(f"Calculating the alpha commutators for power:{power}")
        alpha_com_second = alpha_commutator_second_order(hamiltonian.sparse_repr)

        k = int(
            np.floor(
                np.log(lambd * TIME / ERROR) / np.log(np.log(lambd * TIME / ERROR))
            )
        )

        count = taylor_bench.simulation_gate_count(TIME, k)
        print(f"Taylor:{count}")
        answers["taylor"][power_str] = _truncate(count.get(GATE, 0))

        ktrotter_reps = commutator_r_second_order(
            hamiltonian.sparse_repr, TIME, ERROR, alpha_com=alpha_com_second
        )
        count = ktrotter_bench.simulation_gate_count(TIME, ktrotter_reps)
        print(f"kTrotter:{count}")
        answers["ktrotter"][power_str] = _truncate(count.get(GATE, 0))

        print("-----------------")

    json_name = f"data/benchmark/size/{FILE_NAME}_{GATE}_power_err.json"

    print(answers)

    print(f"Saving results at: {json_name}")
    with open(json_name, "w") as file:
        json.dump(answers, file)

    print(f"Saving data at:{json_name}")


if __name__ == "__main__":
    main()
