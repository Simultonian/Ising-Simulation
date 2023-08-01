import numpy as np
from numpy.typing import NDArray

from ising.hamiltonian import Hamiltonian


class ExactSimulation:
    def __init__(self, ham: Hamiltonian):
        self.ham = ham

    def get_unitary(self, t: float) -> NDArray:
        return (
            self.ham.eig_vec
            @ np.diag(np.exp(complex(0, -1) * t * self.ham.eig_val))
            @ self.ham.eig_vec_inv
        )

    @property
    def ground_state(self) -> NDArray:
        return self.ham.ground_state

    def get_observations(
        self, rho_init: NDArray, observable: NDArray, times: list[float]
    ):
        results = []
        for time in times:
            unitary = self.get_unitary(time)
            rho_final = unitary @ rho_init @ unitary.conj().T
            result = np.trace(np.abs(observable @ rho_final))
            results.append(result)

        return results
