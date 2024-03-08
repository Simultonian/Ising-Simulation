import numpy as np
from ising.hamiltonian import Hamiltonian, parse, parametrized_ising
from ising.benchmark.gates import TrotterBenchmark, qDRIFTBenchmark, TaylorBenchmark
import json

methods = {
    "First Order Trotter": TrotterBenchmark,
    "qDRIFT": qDRIFTBenchmark,
    "Truncated Taylor Series": TaylorBenchmark,
}


def fixed_everything(
    name: str,
    molecule: Hamiltonian,
    obs_norm: float,
    eps_start: float,
    eps_end: float,
    eps_count: int,
    eeta: float,
    success: float,
):
    error_points = [10**x for x in np.linspace(eps_start, eps_end, eps_count)]
    
    synths = {}

    # __init__(ham: Hamiltonian, observable_norm: float, **kwargs):
    for method, synth in methods.items():
        synths[method] = synth(molecule, obs_norm, success=success, overlap=eeta, error = error_points[0])

    # dict[str, dict[float, int]]
    err_depth = {method: {} for method in methods.keys()}

    for method, synth in synths.items():
        for err in error_points:
            gate_count = synth.error_gate_count(err)
            err_depth[method][err] = gate_count["t"]


    for method, err_count in err_depth.items():
        print(f"{method}:{err_count}")

    file_name = f"data/tgatecount/{name}.json"
    print(f"Saving at {file_name}")
    with open(file_name, "w") as file:
        json.dump(err_depth, file)


def main():
    name = "ozone"
    molecule = parse(name)
    print(f"Running for {name}")
    obs_norm = 1
    eps_start, eps_end = -1, -4
    eps_count = 10
    eeta = 0.8
    success = 0.9

    fixed_everything(name, molecule, obs_norm, eps_start, eps_end, eps_count, eeta, success)


def test_main():
    main()


if __name__ == "__main__":
    main()
