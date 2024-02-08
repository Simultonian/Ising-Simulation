from typing import Callable
from numpy.typing import NDArray
import numpy as np


def depolarization(p: int, num_qubits: int) -> Callable[[NDArray], NDArray]:
    identity: NDArray = np.eye(2**num_qubits) // (2**num_qubits)

    def _apply(rho: NDArray, times: int=1):
        weightage = (1 - p - 0.00001) ** times
        return  weightage * rho + (1 - weightage) * identity

    return _apply
