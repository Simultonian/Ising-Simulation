import numpy as np

from ising.hamiltonian import Hamiltonian, parse
import ising.benchmark.groundstate_depth as groundstate_depth

from typing import Callable

methods: dict[str, Callable[[Hamiltonian, float, float, float], int]] = {
    "Ge et al": groundstate_depth.ge_et_al,
    "Phase Estimation and Amplitude Amplification": groundstate_depth.phase_estimation,
    "Poulin et al": groundstate_depth.poulin_et_al,
    "Lin et al": groundstate_depth.lin_et_al,
    "Truncated Taylor Series": groundstate_depth.truncated_taylor,
    "First Order Trotter": groundstate_depth.first_order_trotter,
    "qDRIFT": groundstate_depth.qdrift,
}

colors: dict[str, str] = {
    "Ge et al": "blue",
    "Phase Estimation and Amplitude Amplification": "red",
    "Poulin et al": "black",
    "Lin et al": "red",
    "Truncated Taylor Series": "blue",
    "First Order Trotter": "purple",
    "qDRIFT": "yellow",
}


def fixed_everything(
    molecule: Hamiltonian,
    obs_norm: float,
    eps: float,
    eeta: float,
):

    # ham: Hamiltonian, eeta: float, eps: float, obs_norm: float = 1
    depth = {}
    for method, func in methods.items():
        depth[method] = func(molecule, eeta, eps, obs_norm)

    print(depth)


def main():
    molecule = parse("methane")
    beta = np.sum(np.abs(molecule.coeffs))
    L = len(molecule.coeffs)
    gap = molecule.spectral_gap
    print(f"Terms: {L} \n Sum: {beta} \n gap: {gap}")
    obs_norm = 1
    eps = 1e-1
    eeta = 0.8

    fixed_everything(molecule, obs_norm, eps, eeta)


def test_main():
    main()


if __name__ == "__main__":
    main()
