from typing import Callable
from numpy.typing import NDArray
import numpy as np


def depolarization(p: int, num_qubits: int) -> Callable[[NDArray], NDArray]:
    identity: NDArray = np.eye(2**num_qubits) // (2**num_qubits)

    def _apply(rho: NDArray):
        return (1 - p - 0.00001) * rho + (p + 0.00001) * identity

    return _apply
