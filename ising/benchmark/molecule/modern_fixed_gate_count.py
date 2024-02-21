from ising.hamiltonian import Hamiltonian, parse, parametrized_ising
from ising.benchmark.gates import trotter_gates, qdrift_gates, taylor_gates
import json

from typing import Callable

methods: dict[
    str, Callable[[Hamiltonian, float, float, float, float], dict[str, int]]
] = {
    "First Order Trotter": trotter_gates,
    "qDRIFT": qdrift_gates,
    "Truncated Taylor Series": taylor_gates,
}


def fixed_everything(
    name: str,
    molecule: Hamiltonian,
    obs_norm: float,
    eps: float,
    eeta: float,
    success: float,
):
    depth = {}
    for method, func in methods.items():
        depth[method] = func(molecule, eeta, eps, obs_norm, success)

    gate_sets = set()
    for _, count in depth.items():
        for gate in count.keys():
            gate_sets.add(gate)

    for method, val in depth.items():
        for gate in gate_sets:
            depth[method][gate] = depth[method].get(gate, 0)

    for method, val in depth.items():
        print(f"{method}:{val}")

    file_name = f"data/gatecount/{name}.json"
    print(f"Saving at {file_name}")
    with open(file_name, "w") as file:
        json.dump(depth, file)


def main():
    name = "ising"
    # molecule = parse(name)
    # print(name)
    molecule = parametrized_ising(5, 0.1)
    obs_norm = 1
    eps = 1e-1
    eeta = 0.8
    success = 0.9

    fixed_everything(name, molecule, obs_norm, eps, eeta, success)


def test_main():
    main()


if __name__ == "__main__":
    main()
