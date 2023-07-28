from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def parametrized_ising(qubits: int, h: Parameter) -> SparsePauliOp:
    """
    One dimensional Transverse-field Ising model parameterized by external
    field h. The Hamiltonian is represented by:

    :math:`-1 sum_{<i, j>} Z_i Z_j + h sum_i X`

    Inputs:
        - qubits: Number of qubits for the Hamiltonian
        - j: Energy prefactor
        - h: External field

    Returns: Hamiltonian in SparsePauliOp
    """
    if qubits <= 0:
        raise ValueError("Invalid number of qubits.")

    i_n = ["I"] * qubits

    zz_terms = []
    zz_coeffs = np.array([-1] * (qubits - 1))
    for i in range(qubits - 1):
        j = i + 1
        p_i = i_n
        p_i[i] = "Z"
        p_i[j] = "Z"
        zz_terms.append("".join(p_i))

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

    return SparsePauliOp(zz_terms + x_terms, np.concatenate([zz_coeffs, x_coeffs]))
