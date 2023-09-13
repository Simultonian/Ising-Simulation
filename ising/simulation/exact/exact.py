from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ising.hamiltonian import Hamiltonian
from qiskit.circuit import Parameter

from ising.hamiltonian.hamiltonian import substitute_parameter

class ExactSimulation:
    def __init__(self, ham: Hamiltonian, h: Parameter, error: float):
        self.ham = ham
        self.num_qubits = ham.sparse_repr.num_qubits
        self.ham_subbed: Optional[Hamiltonian] = None

        self.h = h

    @property
    def ground_state(self) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )
        return self.ham_subbed.ground_state

    def subsitute_h(self, h_val: float) -> None:
        self.ham_subbed = substitute_parameter(self.ham, self.h, h_val)

    def construct_parametrized_circuit(self) -> None:
        if self.ham_subbed is None:
            raise ValueError(
                "h value has not been substituted, qiskit does not support parametrized Hamiltonians."
            )

    def matrix(self, time: float) -> NDArray:
        if self.ham_subbed is None:
            raise ValueError("h value has not been substituted.")

        return (
            self.ham_subbed.eig_vec
            @ np.diag(np.exp(complex(0, -1) * time * self.ham_subbed.eig_val))
            @ self.ham_subbed.eig_vec_inv
        )

    def get_observations(
        self, rho_init: NDArray, observable: NDArray, times: list[float]
    ):
        results = []
        for time in times:
            unitary = self.matrix(time)
            rho_final = unitary @ rho_init @ unitary.conj().T
            result = np.trace(np.abs(observable @ rho_final))
            results.append(result)

        return results
