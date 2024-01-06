from typing import Union
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList
from .hamiltonian import Hamiltonian
import numpy as np


def trotter_reps_general(ham: SparsePauliOp, time: float, eps: float) -> int:
    """
    Calculate the Trotter error for general Hamiltonian
    """
    l = len(ham.coeffs)
    coeff_sum = sum(ham.coeffs)

    numr = np.abs(l * (coeff_sum * time) ** 2)
    dr = np.abs(2 * eps)
    final = int(np.ceil(numr / dr))
    if final < 0:
        raise ValueError("Incorrect calculation")

    final = max(final, 1)
    return final


def trotter_reps(num_qubits, h, time: float, eps: float) -> int:
    """
    Calculate the Trotter error for Ising model specifically
    """
    numr = np.abs(h) * num_qubits * (time**2)
    dr = np.abs(eps)
    final = int(np.ceil(numr / dr))
    if final < 0:
        raise ValueError("Incorrect calculation")

    final = max(final, 1)
    return final


def qdrift_count(lambd: float, time: float, eps: float) -> int:
    numr = np.abs(2 * (lambd * time) ** 2)
    dr = eps
    final = np.ceil(numr / dr).astype(int)
    # while final > 10000:
    #     final = final // 10

    final = int(final**0.8)

    final = max(final, 1)
    return final


def group_qdrift_count(lambd: np.float64, time: float, eps: float) -> int:
    numr = np.abs(2 * (lambd * time) ** 2)
    dr = eps
    final = int(np.ceil(numr / dr))
    return final


def general_grouping(ops: Union[list[Pauli], PauliList]) -> list[list[Pauli]]:
    g_x, g_z = [], []
    for op in ops:
        if "Z" in str(op):
            g_z.append(op)
        else:
            g_x.append(op)

    return [g_z, g_x]


def parametrized_ising(
    qubits: int, h: Union[Parameter, float], j: float = -1, normalize: bool = True
) -> Hamiltonian:
    """
    One dimensional Transverse-field Ising model parameterized by external
    field h. The Hamiltonian is represented by:

    :math:`j sum_{<i, j>} Z_i Z_j + h sum_i X`

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
    zz_coeffs = np.array([j] * (qubits - 1))
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

    return Hamiltonian(
        sparse_repr=SparsePauliOp(
            zz_terms + x_terms, np.concatenate([zz_coeffs, x_coeffs])
        ),
        normalized=normalize,
    )
