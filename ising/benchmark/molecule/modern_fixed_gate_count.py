from ising.hamiltonian import Hamiltonian, parse, parametrized_ising
from ising.benchmark.gates.trotter import TrotterBenchmark
from ising.benchmark.gates.qdrift import qDRIFTBenchmark
from ising.benchmark.gates.taylor import TaylorBenchmark

from typing import Callable

colors: dict[str, str] = {
    "Truncated Taylor Series": "blue",
    "First Order Trotter": "purple",
    "qDRIFT": "yellow",
}


def trotter_gates(
    ham: Hamiltonian, obs_norm: float, overlap: float, error: float, success: float
) -> dict[str, int]:
    benchmarker = TrotterBenchmark(
        ham, obs_norm, overlap=overlap, error=error, success=success
    )

    return benchmarker.calculate_gates()


def qdrift_gates(
    ham: Hamiltonian, obs_norm: float, overlap: float, error: float, success: float
) -> dict[str, int]:
    benchmarker = qDRIFTBenchmark(
        ham, obs_norm, overlap=overlap, error=error, success=success
    )

    return benchmarker.calculate_gates()


def taylor_gates(
    ham: Hamiltonian, obs_norm: float, overlap: float, error: float, success: float
) -> dict[str, int]:
    benchmarker = TaylorBenchmark(
        ham, obs_norm, overlap=overlap, error=error, success=success
    )

    return benchmarker.calculate_gates()


methods: dict[
    str, Callable[[Hamiltonian, float, float, float, float], dict[str, int]]
] = {
    "First Order Trotter": trotter_gates,
    "qDRIFT": qdrift_gates,
    "taylor": taylor_gates,
}


def fixed_everything(
    molecule: Hamiltonian,
    obs_norm: float,
    eps: float,
    eeta: float,
    success: float,
):
    depth = {}
    for method, func in methods.items():
        depth[method] = func(molecule, eeta, eps, obs_norm, success)

    for method, val in depth.items():
        print(f"{method}:{val}")


def main():
    # name = "methane"
    # molecule = parse(name)
    # print(name)
    molecule = parametrized_ising(5, 0.1)
    obs_norm = 1
    eps = 1e-1
    eeta = 0.8
    success = 0.9

    fixed_everything(molecule, obs_norm, eps, eeta, success)


def test_main():
    main()


if __name__ == "__main__":
    main()
