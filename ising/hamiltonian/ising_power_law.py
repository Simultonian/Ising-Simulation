from typing import Union
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList
from .hamiltonian import Hamiltonian, sparse_spectral_gap
import numpy as np


def parametrized_ising_power(
    qubits: int,
    h: Union[Parameter, float],
    J: float = -1,
    power: float = 1,
    normalize: bool = True,
) -> Hamiltonian:
    """
    One dimensional Transverse-field Ising model parameterized by external
    field h with power-law decay. The Hamiltonian is represented by:

    :math:`j sum_{i neq j} 1/(i - j)^power Z_i Z_j + h sum_i X`

    Inputs:
        - qubits: Number of qubits for the Hamiltonian
        - j
        - h: External field

    Returns: Hamiltonian in SparsePauliOp
    """
    if qubits <= 0:
        raise ValueError("Invalid number of qubits.")

    i_n = ["I"] * qubits

    zz_terms = []
    zz_coeffs = []
    for i in range(qubits - 1):
        for j in range(i + 1, qubits):
            p_i = i_n
            p_i[i] = "Z"
            p_i[j] = "Z"
            zz_terms.append("".join(p_i))
            zz_coeffs.append(J / ((j - i) ** power))

            # Reset to avoid copying
            p_i[i] = "I"
            p_i[j] = "I"

    x_terms = []
    x_coeffs = np.array([h for _ in range(qubits)])

    for i in range(qubits):
        p_i = i_n
        p_i[i] = "X"
        x_terms.append("".join(p_i))

        # Reset to avoid copying
        p_i[i] = "I"

    return Hamiltonian(
        sparse_repr=SparsePauliOp(
            zz_terms + x_terms, np.concatenate([zz_coeffs, x_coeffs])
        ),
        normalized=normalize,
        _approx_spectral_gap=2 * abs(J) * abs(1 - h),
    )
