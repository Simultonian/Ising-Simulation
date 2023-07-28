from typing import Union
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from numpy.typing import NDArray


class ExactSimulation:
    def __init__(
        self,
        ham: Union[SparsePauliOp, NDArray],
        observable: Union[SparsePauliOp, NDArray],
    ):
        if isinstance(ham, SparsePauliOp):
            ham = ham.to_matrix()

        self.eig_val, self.eig_vec = np.linalg.eig(ham)
        self.eig_vec_inv = np.linalg.inv(self.eig_vec)

        if isinstance(observable, SparsePauliOp):
            self.observable = observable.to_matrix()
        else:
            self.observable = observable

    def get_unitary(self, t: float) -> NDArray:
        return (
            self.eig_vec
            @ np.diag(np.exp(complex(0, -1) * t * self.eig_val))
            @ self.eig_vec_inv
        )

    def get_observations(
        self,
        rho_init: NDArray,
        total_time: float = 0.0,
        division_count: int = 0,
        para_times: list[int] = [],
    ) -> list[float]:
        if division_count == 0:
            if len(para_times) == 0:
                raise ValueError("Both division_count and para_times not provided.")
            times = np.array(para_times)
        else:
            if len(para_times) != 0:
                raise ValueError("Both division_count and para_times provided.")
            times = np.linspace(0, total_time, division_count)

        results = []
        for time in times:
            unitary = self.get_unitary(time)
            rho_final = unitary @ rho_init @ unitary.conj().T
            result = np.trace(np.abs(self.observable @ rho_final))
            results.append(result)

        return results
