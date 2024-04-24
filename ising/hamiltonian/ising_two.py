from typing import Union
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList
from .hamiltonian import Hamiltonian
import numpy as np


def generate_matrix(side):
    return [[i + side * j for i in range(side)] for j in range(side)]


def generate_neighbours(side):
    NEIGHBOURS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def boundary(ni, nj):
        return 0 <= ni < side and 0 <= nj < side

    pairs = set()
    for i in range(side):
        for j in range(side):
            for di, dj in NEIGHBOURS:
                ni, nj = i + di, j + dj
                if not boundary(ni, nj):
                    continue

                pair = [(i, j), (ni, nj)]
                pair.sort()
                pair = tuple(pair)
                pairs.add(pair)

    return pairs


def parametrized_ising_two(
    side: int, h: Union[Parameter, float], J: float = -1, normalize: bool = True
) -> Hamiltonian:
    """
    Square two dimensional Transverse-field Ising model parameterized by external
    field h. The Hamiltonian is represented by:

    :math:`j sum_{<i, j>} Z_i Z_j + h sum_i X`

    --

    Unlike 1D Ising model, there is no easy way of approximating the spectral
    gap.

    Inputs:
        - qubits: Number of qubits for the Hamiltonian
        - j
        - h: External field

    Returns: Hamiltonian in SparsePauliOp
    """
    if side <= 0:
        raise ValueError("Invalid number of qubits.")

    qubits = side**2

    grid = generate_matrix(side)
    lattices = generate_neighbours(side)

    i_n = ["I"] * qubits
    zz_terms = []
    for point1, point2 in lattices:
        qubit1, qubit2 = grid[point1[0]][point1[1]], grid[point2[0]][point2[1]]
        i_n[qubit1] = "Z"
        i_n[qubit2] = "Z"

        zz_terms.append("".join(i_n))

        i_n[qubit1] = "I"
        i_n[qubit2] = "I"

    zz_coeffs = [J] * len(zz_terms)
    x_terms = []
    x_coeffs = np.array([h for _ in range(qubits)])

    for i in range(qubits):
        i_n[i] = "X"
        x_terms.append("".join(i_n))

        # Reset to avoid copying
        i_n[i] = "I"

    ham = Hamiltonian(
        sparse_repr=SparsePauliOp(
            zz_terms + x_terms, np.concatenate([zz_coeffs, x_coeffs])
        ),
        normalized=normalize,
    )
    return ham
